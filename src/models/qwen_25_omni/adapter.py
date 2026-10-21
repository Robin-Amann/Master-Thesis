import torch
from typing import Any, Optional
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

from src.models.model_adapter import ModelAdapter
from src.config_types.model import ModelConfig
from src.models.qwen_25_omni.generation import media_encoder, prefill_validate_inputs
from src.models.shared_logic.load_model import load_model
from src.models.shared_logic.generation import GenerationState, get_eos_token_ids, move_inputs_to_device, get_press_context, decode
from src.utils.logger import logger


class QwenOmniAdapter(ModelAdapter):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        model, processor = load_model(
            model_name=config.checkpoint, 
            model_class=Qwen2_5OmniForConditionalGeneration,
            torch_dtype=torch.bfloat16, 
            device=torch.device(config.device),
            model_kwargs={}, 
            processor_kwargs={}
        )
        # if you use full model instead of only thinker use return_audio=False for generate(...)
        self.hf_model: Qwen2_5OmniForConditionalGeneration = model
        self.hf_processor: Qwen2_5OmniProcessor = processor


    def prepare_inputs(
        self, 
        entry: dict, 
        use_image: bool = True, 
        use_video: bool = True, 
        use_audio_in_video: bool = True, 
        use_audio: bool = True, 
        use_text: bool = True,
        remove_system_message: bool = False
    ) :
        # build messages
        content = []
        if "image_path" in entry and use_image:
            content.append({"type": "image", "image": entry["image_path"]})
        if "video_path" in entry and use_video:
            content.append({"type": "video", "video": entry["video_path"], "fps": 1.0})
        if "audio_path" in entry and use_audio:
            content.append({"type": "audio", "audio": entry["audio_path"]})
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


    def get_positions(self, state: GenerationState, input_ids: torch.Tensor) :
        # indicate the positions of inputs in the sequence
        # https://huggingface.co/docs/transformers/v4.47.1/kv_cache
        cache_position = torch.arange(
            state.context_len,
            state.context_len + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )

        # token’s position in the list of tokens
        # https://huggingface.co/docs/transformers/glossary
        position_ids = torch.arange(
            state.context_logical_len,
            state.context_logical_len + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        return cache_position, position_ids


    @torch.inference_mode()
    def prefill(
        self,
        inputs: dict[str, Any],
        state: Optional[GenerationState] = None,
        press=None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> GenerationState:
        model_kwargs = dict(model_kwargs or {})
        inputs = move_inputs_to_device(inputs, self.device)
        input_ids: torch.Tensor = inputs["input_ids"]

        prefill_validate_inputs(input_ids, model_kwargs)
        
        if state is None:
            state = GenerationState(
                past_key_values=DynamicCache(config=self.hf_model.config),
                context_len=0,
                context_logical_len=0,
                eos_token_ids=get_eos_token_ids(self.hf_model),
                finished=False
            )

        if state.finished:
            raise ValueError("Cannot append prefill inputs to a finished state.")

        # remove kv pairs that are not deemed context
        if state.context_len < state.past_key_values.get_seq_length() :
            state.past_key_values.crop(state.context_len)

        cache_position, position_ids = self.get_positions(state, input_ids)

        kwargs = {
            **inputs,
            "past_key_values": state.past_key_values, 
            "cache_position": cache_position, # the logical positions of the new input
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

        state.context_len = outputs.past_key_values.get_seq_length()
        state.context_logical_len = state.context_logical_len + input_ids.shape[1]

        return state


    @torch.inference_mode()
    def generate(
        self,
        state: Optional[GenerationState] = None,
        inputs: Optional[dict[str, Any]] = None,
        max_new_tokens: int = 128, # TODO: constant
        model_kwargs: Optional[dict[str, Any]] = None,
    ) :

        if state is None :
            state = GenerationState(
                past_key_values=DynamicCache(config=self.hf_model.config),
                context_len=0,
                context_logical_len=0,
                eos_token_ids=get_eos_token_ids(self.hf_model),
                finished=False
            )

        inputs = move_inputs_to_device(inputs, self.device)
        
        # crop(-100) --> remove last 100 pairs || crop(100) --> only keep first 100
        state.past_key_values.crop(state.context_len)
        cache_position, position_ids = self.get_positions(state, inputs["input_ids"])
        # generate has no parameter "return_dict"
        logger.info(str(cache_position))
        logger.info(str(position_ids))
        text_ids = self.hf_model.generate(
            **inputs,
            use_cache=True,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            return_audio=False,
            **(model_kwargs or {})
        )

        logger.info(f"{state.past_key_values.get_seq_length()}")

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