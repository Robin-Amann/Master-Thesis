import torch
from typing import Any, Mapping, Optional
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

from src.models.model_adapter import ModelAdapter
from src.config_types.model import ModelConfig
from src.models.qwen_25_omni.generation import media_encoder, prefill_validate_inputs
from src.models.shared_logic.load_model import load_model
from src.models.shared_logic.generation import GenerationState, get_eos_token_ids, move_inputs_to_device, get_press_context, decode

import logging

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

        logging.info("messages:\n", messages)

        audio_in_video = "video_path" in entry and use_video and use_audio_in_video
        inputs = media_encoder(self.hf_processor, messages, audio_in_video, remove_system_message)
        return inputs


    @property
    def device(self):
        return self.hf_model.device


    @torch.inference_mode()
    def forward(
        self,
        inputs: dict[str, Any],
        state: Optional[GenerationState] = None,
        press=None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> GenerationState:
        """
        Append processor-produced inputs to an optional existing generation
        state, apply the KV-cache strategy, and return the updated state.

        Args:
            inputs:
                Processor output for the NEW input segment only.
            state:
                Existing generation state, or None for initial prefill.
            press:
                Optional context-manager factory, e.g. press(model).
            model_kwargs:
                Additional model arguments. These must not override
                internally managed cache arguments.

        Returns:
            A new GenerationState.
        """

        model_kwargs = dict(model_kwargs or {})
        inputs = move_inputs_to_device(inputs, self.device)
        input_ids: torch.Tensor = inputs["input_ids"]

        prefill_validate_inputs(input_ids, model_kwargs)

        # Stage 1: establish the previous state and logical offset.
        if state is None:
            cache = DynamicCache(config=self.hf_model.config)
            logical_start = 0
            prompt_ids = input_ids
            eos_token_ids = get_eos_token_ids(self.hf_model)
        else:
            if state.finished:
                raise ValueError(
                    "Cannot append prefill inputs to a finished state."
                )

            cache = state.past_key_values
            logical_start = state.logical_length

            if state.prompt_ids.shape[0] != input_ids.shape[0]:
                raise ValueError("Input and state batch sizes must match.")

            prompt_ids = torch.cat(
                [state.prompt_ids, input_ids],
                dim=1,
            )

            if len(state.generated_ids) > 0 :
                raise ValueError("Model already generated ids. Can't use past state for prefill.")
                
            eos_token_ids = state.eos_token_ids

        input_length = input_ids.shape[1]
        logical_end = logical_start + input_length

        # Stage 2: establish cache positions.
        # cache_position tracks the logical positions of the new tokens.
        # This is distinct from the number of KV entries physically
        # retained after eviction.
        cache_position = torch.arange(
            logical_start,
            logical_end,
            device=input_ids.device,
            dtype=torch.long,
        ) # [start, start+1, start+2, ..., end-2, end-1]

        # Respect explicitly supplied position_ids. Otherwise, let the
        # model construct model-specific positions where supported.
        inputs.pop("cache_position", None)

        # Stage 3: append the new inputs and apply the KV strategy.
        model = self.hf_model.thinker
        if press is not None :
            with press(model):
                outputs = model(
                    **inputs,
                    past_key_values=cache,
                    cache_position=cache_position, # the logical positions of the new input
                    use_cache=True,
                    return_dict=True,
                    **model_kwargs,
                )
        else :
            outputs = model(
                **inputs,
                past_key_values=cache,
                cache_position=cache_position, # the logical positions of the new input
                use_cache=True,
                return_dict=True,
                **model_kwargs,
            )
        # Stage 4: construct the updated generation state.
        return GenerationState(
            past_key_values=outputs.past_key_values,
            next_logits=outputs.logits[:, -1, :],
            logical_length=logical_end,
            prompt_ids=prompt_ids,
            generated_ids=[],
            eos_token_ids=eos_token_ids,
            finished=False,
        )


    @torch.inference_mode()
    def old_generate(self, state: GenerationState) -> Optional[int]:
        """Generate one greedy token and advance the KV cache by one position. Alters the state in place"""
        if state is None:
            raise RuntimeError("Call prefill() before generate().")

        if state.finished:
            return None

        # Greedy decoding.
        token_id = int(state.next_logits.argmax(dim=-1).item())
        state.generated_ids.append(token_id)

        if token_id in state.eos_token_ids:
            state.finished = True
            return token_id

        token = torch.tensor(
            [[token_id]],
            device=self.device,
            dtype=torch.long,
        )

        position = torch.tensor(
            [[state.logical_length]],
            device=self.device,
            dtype=torch.long,
        )
        cache_position = torch.tensor(
            [state.logical_length],
            device=self.device,
            dtype=torch.long,
        )

        outputs = self.hf_model.thinker(
            input_ids=token,
            past_key_values=state.past_key_values,
            position_ids=position,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

        state.past_key_values = outputs.past_key_values
        state.next_logits = outputs.logits[:, -1, :]
        state.logical_length += 1

        return token_id


    @torch.inference_mode()
    def generate(
        self,
        prefill_state: Optional[GenerationState] = None,
        inputs: Optional[dict[str, Any]] = None,
        max_new_tokens: int = 128, # TODO: constant
        press=None,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> str | tuple[str, GenerationState]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        if prefill_state is None and inputs is None :
            raise ValueError("need to provide past model state or inputs to predict new tokens")

        if prefill_state.finished:
            raise ValueError("Model is already finished")


        start = len(prefill_state.generated_ids)


        outputs = self.hf_model.thinker.generate(
            input_ids=inputs["input_ids"],
            past_key_values=prefill_state.past_key_values,
            max_new_tokens=max_new_tokens,
        )

        generated_text = decode(self.hf_processor, prefill_state, start)

        return generated_text
