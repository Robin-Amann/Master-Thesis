import torch
from typing import Any, Mapping, Optional
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

from src.models.model_adapter import ModelAdapter
from src.config_types.model import ModelConfig
from .generation import media_encoder, prefill_validate_inputs
from ..shared_logic.load_model import load_model
from ..shared_logic.generation import GenerationState, get_eos_token_ids, move_inputs_to_device, get_press_context, decode


class QwenOmniAdapter(ModelAdapter):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        model, processor = load_model(
            model_name=config.checkpoint, 
            torch_dtype=torch.bfloat16, 
            device=torch.device(config.device),
            model_kwargs={}, 
            processor_kwargs={}
        )
        self.hf_model: Qwen2_5OmniForConditionalGeneration = model
        self.hf_processor: Qwen2_5OmniProcessor = processor

    def prepare_inputs(self, entry: dict, use_audio_in_video: bool = True) :
        if "video_path" in entry :
            kind, key = "video", "video_path"
        elif "audio_path" in entry :
            kind, key = "audio", "audio_path"
        elif "image_path" in entry :
            kind, key = "image", "image_path"
        elif "text" in entry:
            kind, key = "text", "text"
        else :
            raise ValueError("no content provided. what now?")

        messages = [{
            "role": "user", 
            "content": [{
                "type": kind, 
                kind: entry[key]
            }]
        }]

        inputs = media_encoder(self.hf_processor, messages, use_audio_in_video and kind == "video")
        return inputs


    @property
    def device(self):
        return self.hf_model.device


    @torch.inference_mode()
    def prefill(
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
            generated_ids = []
            eos_token_ids = get_eos_token_ids(self.hf_model)
            finished = False
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

            generated_ids = list(state.generated_ids)
            eos_token_ids = set(state.eos_token_ids)
            finished = state.finished

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
        )

        # Respect explicitly supplied position_ids. Otherwise, let the
        # model construct model-specific positions where supported.
        inputs.pop("cache_position", None)

        # Stage 3: append the new inputs and apply the KV strategy.
        context = get_press_context(press)

        with context(self.hf_model):
            outputs = self.hf_model(
                **inputs,
                past_key_values=cache,
                cache_position=cache_position,
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
            generated_ids=generated_ids,
            eos_token_ids=eos_token_ids,
            finished=finished,
        )


    @torch.inference_mode()
    def generate(self, state: GenerationState) -> Optional[int]:
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

        outputs = self.hf_model(
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
    def predict(
        self,
        state: Optional[GenerationState] = None,
        inputs: Optional[dict[str, Any]] = None,
        max_new_tokens: int = 128, # TODO: constant
        press=None,
        model_kwargs: Optional[dict[str, Any]] = None,
        return_state: bool = False,
    ) -> str | tuple[str, GenerationState]:
        """
        Generate a response from an optional existing GenerationState.

                Returns:
            Generated text, optionally paired with the updated state.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        # Resolve the call signature.
        if state is None and inputs is None :
            raise ValueError("need to provide past model state or inputs to predict new tokens")

        # Extend the existing prompt/cache before decoding.
        if inputs is not None:
            state = self.prefill(
                inputs=inputs,
                state=state,
                press=press,
                model_kwargs=model_kwargs,
            )

        # Track only tokens generated during this predict() call.
        start = len(state.generated_ids)

        for _ in range(max_new_tokens):
            if state.finished:
                break

            token_id = self.generate(state)

            if token_id is None:
                break

            if state.finished:
                break


        generated_text = decode(self.hf_processor, state, start)

        if return_state:
            return generated_text, state

        return generated_text
