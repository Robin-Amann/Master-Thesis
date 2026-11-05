import torch
from typing import Any
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

from src.models.model_adapter import ModelAdapter
from src.config_types.model import ModelConfig
from src.models.qwen_25_omni.generation import media_encoder, prefill_validate_inputs
from src.models.shared_logic.load_model import load_model
from src.models.shared_logic.generation import GenerationState, get_eos_token_ids, move_inputs_to_device
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

        # Physical KV-cache positions.
        cache_position = torch.arange(
            state.context_len,
            state.context_len + seq_len,
            device=device,
            dtype=torch.long,
        )

        # position ids do save as rope delta so don't use them because model keeps track of delta
        # N -> [1, N] | [1, N] + [B, 1] -> [B, N]
        # text_positions = cache_position.unsqueeze(0) + state.rope_deltas.to(device)
        # # [B, N] -> [1, B, N] | [1, B, N] -> [3, B, N]
        # position_ids = text_positions.unsqueeze(0).expand(3, batch_size, seq_len)

        return cache_position, attention_mask  # , position_ids


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
            
        cache_position, attention_mask = self.get_positions(state, inputs)

        # TODO: check how rope delta actually behaves in implementation. 
        # seems like it does not matter what i set rope delta to because it 
        # depends on attention mask and other things. 
        kwargs = {
            **inputs,
            "attention_mask": attention_mask,
            "past_key_values": state.past_key_values,
            "cache_position": cache_position,
            "rope_deltas": state.rope_deltas,
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

        cache_position, full_attention_mask = self.get_positions(state, inputs)

        # generate has no parameter "return_dict"
        kwargs = {
            **inputs,
            "use_cache": True,
            "past_key_values": state.past_key_values,
            "cache_position": cache_position,
            "attention_mask": full_attention_mask,
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