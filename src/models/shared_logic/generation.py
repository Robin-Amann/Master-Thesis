from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, DynamicCache
from typing import Any
from contextlib import nullcontext

@dataclass
class GenerationState:
    # KV cache after processing the prompt and generated tokens so far.
    past_key_values: DynamicCache

    # Logits predicting the next token.
    next_logits: torch.Tensor

    # Number of tokens processed by the model
    logical_length: int

    prompt_ids: torch.Tensor
    generated_ids: list[int]

    eos_token_ids: set[int]
    finished: bool = False


def decode(processor: Any, state: GenerationState, generated_ids_offset: int = 0) -> str:
    ids = torch.tensor([state.generated_ids[generated_ids_offset:]], dtype=torch.long)
    return processor.batch_decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]



def get_eos_token_ids(model: PreTrainedModel) -> set[int]:
    eos = model.generation_config.eos_token_id

    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}

    return set(eos)


def move_inputs_to_device(
    inputs: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move processor-produced tensors to the model device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def get_press_context(press):
    """Construct the KV eviction/compression context."""
    if press is None:
        return nullcontext()

    return press
