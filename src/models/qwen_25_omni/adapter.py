import torch
from typing import Any
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from src.models.model_adapter import ModelAdapter
from src.config_types.model import ModelConfig
from src.models.shared_logic.generation import GenerationState, get_eos_token_ids, move_inputs_to_device
from src.utils.logger import logger
from pathlib import Path
import src.models.shared_logic.qkv_extraction as qk_extraction 
from src.utils.utils import join_attentions
from src.models.qwen_25_omni.thinker import forward as thinker_forward
from types import MethodType

from qwen_omni_utils import process_mm_info
from datetime import datetime

from dataclasses import dataclass

@dataclass
class Video:
    path: str
    fps: float = 1.0
    use_audio: bool = False

@dataclass
class Message:
    audio_path: str = None
    image_path : str = None
    video: Video = None
    text: str = None



class QwenOmniAdapter(ModelAdapter):

    def __init__(self, config: ModelConfig, model_kwargs: dict = None, processor_kwargs: dict = None):
        super().__init__(config)

        model: Qwen2_5OmniForConditionalGeneration = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            config.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **(model_kwargs or {})
        )

        model.thinker.model.forward = MethodType(thinker_forward, model.thinker.model)
        model.eval()
        model = model.to(torch.device(config.device))

        processor_kwargs = processor_kwargs or {}
        processor = Qwen2_5OmniProcessor.from_pretrained(
            config.checkpoint,
            **processor_kwargs
        )

        # if you use full model instead of only thinker use return_audio=False for generate(...)
        self.hf_model = model
        self.hf_processor = processor


    def set_sliding_windows(self, first_n: int = 8, window_size: int = 720 * 4) :
        text_cfg = self.hf_model.config.thinker_config.text_config
        text_cfg.use_sliding_window = True
        text_cfg.sliding_window = window_size
        text_cfg.max_window_layers = first_n

        text_cfg.layer_types = [
            "sliding_attention" if i < first_n else "full_attention"
            for i in range(text_cfg.num_hidden_layers)
        ]


    def build_messages(self, user_input: Message | list[Message], keep_system_message: bool = True) :
        if type(user_input) == Message :
            user_input = [user_input]

        messages: list[dict] = []

        for message in user_input :
            content = []
            if message.image_path is not None :
                content.append({"type": "image", "image": message.image_path})
            if message.audio_path is not None :
                content.append({"type": "audio", "audio": message.audio_path})
            if message.video is not None :
                content.append({"type": "video", "video": message.video.path, "fps": message.video.fps})
            if message.text is not None :
                content.append({"type": "text", "text": message.text})

            messages.append({
                "role": "user", 
                "content": content
            })

        text = self.hf_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not keep_system_message :
            index = text.find("<|im_start|>user")
            text = text[index:]

        return messages, text


    def prepare_inputs(
        self, 
        messages: list[dict],
        input_text: str,
        use_audio_in_video: bool = False, 
        position_offset: int = 0
    ) :

        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=use_audio_in_video,
        )

        inputs: dict = self.hf_processor(
            text=[input_text],
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=False,
            use_audio_in_video=use_audio_in_video,
        )

        position_ids, rope_deltas = self.hf_model.thinker.get_rope_index(
            input_ids=inputs["input_ids"],
            video_grid_thw=inputs.get("video_grid_thw", None),
            attention_mask=inputs["attention_mask"],
            second_per_grids=inputs.get("video_second_per_grid", None) 
        )
        position_ids = position_ids + position_offset
        inputs["position_ids"] = position_ids


        return inputs


    @property
    def device(self):
        return self.hf_model.device


    @torch.inference_mode()
    def prefill(
        self,
        inputs: dict[str, Any],
        state: GenerationState | None = None,
        press=None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> GenerationState:
        model_kwargs = dict(model_kwargs or {})
        inputs = move_inputs_to_device(inputs, self.device)
        
        if state is None:
            state = GenerationState(
                past_key_values=DynamicCache(config=self.hf_model.config),
                eos_token_ids=get_eos_token_ids(self.hf_model),
            )

        if state.finished:
            raise ValueError("Cannot append prefill inputs to a finished state.")

        # remove kv pairs that are not deemed context
        if state.context_len < state.past_key_values.get_seq_length() :
            state.past_key_values.crop(state.context_len)
    
        kwargs = {
            **inputs,
            "past_key_values": state.past_key_values,
            "use_cache": True,
            "return_dict": True,
            **model_kwargs,
        }

        # Qwen Omni does implement forward not directly --> thinker does
        if press is None :
            outputs = self.hf_model.thinker(**kwargs)
        else :
            with press(self.hf_model):
                outputs = self.hf_model.thinker(**kwargs)

        input_length = inputs["input_ids"].shape[1]
        state.context_len += input_length
        state.context_logical_len += input_length

        return state


    @torch.inference_mode()
    def generate(
        self,
        inputs: dict[str, Any],
        state: GenerationState | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) :
        if state is None :
            state = GenerationState(
                past_key_values=DynamicCache(config=self.hf_model.config),
                eos_token_ids=get_eos_token_ids(self.hf_model),
            )

        inputs = move_inputs_to_device(inputs, self.device)
        
        # crop(-100) --> remove last 100 pairs || crop(100) --> only keep first 100
        if state.context_len < state.past_key_values.get_seq_length() :
            state.past_key_values.crop(state.context_len)

        # I don't know why but generate needs cache positions. forward doesn't for some reason
        cache_position = torch.arange(
            0,
            inputs["input_ids"].shape[1], 
            device=inputs["input_ids"].device, 
            dtype=torch.long
        ) + state.past_key_values.get_seq_length()

        logger.info(f"cache position: {cache_position}")

        # generate has no parameter "return_dict"
        kwargs = {
            **inputs,
            "use_cache": True,
            "past_key_values": state.past_key_values,
            "return_dict_in_generate": True,
            "cache_position": cache_position,
            "return_audio": False,
            **(model_kwargs or {})
        }
        text_ids = self.hf_model.generate(**kwargs)
        
        logger.info(f"post generate KV Cache seq length: {state.past_key_values.get_seq_length()}")

        # Qwen returns all tokens so remove input tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids: torch.Tensor = text_ids[:, input_length:]  # new generated ids

        text: str = self.hf_processor.batch_decode(
            generated_ids,
            # skip_special_tokens=True,
            # clean_up_tokenization_spaces=False,
        )

        return generated_ids, text, state


    def ask_question(self, entry: dict, attn_save_dir:Path = None, return_attentions = False) :
        if return_attentions :
            print("attn_implementation must be eager")

        if attn_save_dir is not None :
            attn_save_dir = attn_save_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            captured, handles = qk_extraction.register_hooks(self.hf_model.thinker.model.layers)

        inputs = self.prepare_inputs(entry, use_audio_in_video=False)
        input_length = inputs["input_ids"].shape[1]
        # generated_ids, text, state = model.generate(inputs)
        inputs = move_inputs_to_device(inputs, self.device) 
        kwargs = inputs | { 
            "return_audio": False, 
            "output_attentions": True, 
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        with torch.no_grad():
            outputs = self.hf_model.generate(**kwargs)

        if attn_save_dir is not None :
            # Remove hooks
            for handle in handles:
                handle.remove()

            qk_extraction.save_queries_and_keys(captured, outputs.past_key_values, attn_save_dir / "qk")

            video_indices = torch.where(inputs["input_ids"][0] == 151656)[0]  # video_token_id
            first, last = video_indices[0].item(), video_indices[-1].item() 
            qk_extraction.calculate_attention_scores(
                attn_save_dir,
                self.hf_model.thinker.model.layers,
                self.hf_model.thinker.config.text_config.rope_parameters["mrope_section"],
                media_boundaries=[first, last]
            )

        generated_ids: torch.Tensor = outputs.sequences[:, input_length:]  # new generated ids
        text: str = self.hf_processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        if return_attentions :
            return text, join_attentions(outputs.attentions)

        return text