import torch
from typing import Any
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from src.models.model_adapter import ModelAdapter
from src.config_types.model import ModelConfig
from src.models.qwen_25_omni.generation import media_encoder, prefill_validate_inputs
from src.models.shared_logic.load_model import load_model
from src.models.shared_logic.generation import GenerationState, get_eos_token_ids, move_inputs_to_device
from src.utils.logger import logger

from src.models.qwen_25_omni.thinker import forward as thinker_forward

from types import MethodType


class QwenOmniAdapter(ModelAdapter):

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        model: Qwen2_5OmniForConditionalGeneration = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            config.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # **model_kwargs
        )
        # model.thinker.model = ThinkerTextModel._from_config(
        #     model.config.thinker_config.text_config,
        #     torch_dtype=torch.bfloat16,
        #     # device_map="auto",
        # )
        model.thinker.model.forward = MethodType(thinker_forward, model.thinker.model)
        model.eval()
        model = model.to(torch.device(config.device))

        processor = Qwen2_5OmniProcessor.from_pretrained(
            config.checkpoint,
            # **processor_kwargs
        )

        # if you use full model instead of only thinker use return_audio=False for generate(...)
        self.hf_model = model
        self.hf_processor = processor


    def prepare_inputs(
        self, 
        entry: dict, 
        use_audio: bool = True, 
        use_image: bool = True, 
        use_video: bool = True, 
        use_audio_in_video: bool = True, 
        use_text: bool = True,
        remove_system_message: bool = False
    ) :
        # build messages
        content = []
        if "audio_path" in entry and use_audio:
            content.append({"type": "audio", "audio": entry["audio_path"]})
        if "image_path" in entry and use_image:
            content.append({"type": "image", "image": entry["image_path"]})
        if "video_path" in entry and use_video:
            content.append({"type": "video", "video": entry["video_path"], "fps": 1.0})
        if "text" in entry and use_text:
            content.append({"type": "text", "text": entry["text"]})

        if len(content) == 0 :
            raise ValueError("no content provided. what now?")

        messages = [{
            "role": "user", 
            "content": content
        }]

        logger.info("messages:")
        logger.info(str(messages))

        audio_in_video = "video_path" in entry and use_video and use_audio_in_video
        inputs = media_encoder(self.hf_processor, messages, audio_in_video, remove_system_message)
        return inputs


    @property
    def device(self):
        return self.hf_model.device


    def get_positions(
        self,
        state: GenerationState,
        inputs: dict[str, torch.Tensor],
    ):
        """
        Returns:
            cache_position: [seq_len]
            attention_mask: [batch, total_seq_len]
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        logger.info(device)

        # Append the new attention mask to the existing mask.
        input_attentions = inputs.get("attention_mask", None)
        if input_attentions is None:
            input_attentions = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        attention_mask = torch.cat([ state.attention_mask.to(device), input_attentions.to(device) ], dim=-1)

        # don't use position ids for media inputs because they use rope 
        media_key = ["feature_attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "video_second_per_grid"]
        if any( key in inputs for key in media_key ) :
            return None, attention_mask

        # token position = seqeuence position + rope delta
        position_ids = (
            torch.arange(0, seq_len, device=device, dtype=torch.long)
            + state.context_len
            + state.rope_deltas
        )

        return position_ids, attention_mask


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
        input_ids: torch.Tensor = inputs["input_ids"]

        prefill_validate_inputs(input_ids, model_kwargs)
        
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
            # attention mask and rope_deltas correspond to context len
    
        position_ids, attention_mask = self.get_positions(state, inputs)

        kwargs = {
            **inputs,
            "past_key_values": state.past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
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
        # output keys: ['logits', 'past_key_values']

        input_length = inputs["input_ids"].shape[1]
        state.context_len += input_length
        state.context_logical_len += input_length
        state.attention_mask = attention_mask
        state.rope_deltas = outputs.rope_deltas

        return state


    @torch.inference_mode()
    def generate(
        self,
        state: GenerationState | None = None,
        inputs: dict[str, Any] | None = None,
        max_new_tokens: int = 128, # TODO: constant
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

        position_ids, attention_mask = self.get_positions(state, inputs)

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
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "return_audio": False,
            **(model_kwargs or {})
        }
 
        text_ids = self.hf_model.generate(**kwargs)
        logger.info(f"post generate KV Cache seq length: {state.past_key_values.get_seq_length()}")

        # Qwen returns all tokens so remove input tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = text_ids[:, input_length:]  # new generated ids

        text = self.hf_processor.batch_decode(
            generated_ids,
            # skip_special_tokens=True,
            # clean_up_tokenization_spaces=False,
        )

        return generated_ids, text, state


# forward returns: ['logits', 'past_key_values', 'rope_deltas']
# generate returns: text_ids