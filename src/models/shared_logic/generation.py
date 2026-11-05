from dataclasses import dataclass, field
import torch
from transformers import PreTrainedModel, DynamicCache
from typing import Any
from contextlib import nullcontext

@dataclass
class GenerationState:
    past_key_values: DynamicCache   # KV cache after processing the prompt and generated tokens so far.
    context_len: int                # number of kv entries that belong to context
    context_logical_len: int        # actual number of tokens that belong to context (longer than context_len if kv eviction took place)
    eos_token_ids: set[int]         # tokens that idicate that generation is over
    finished: bool                  # is generation with past_key_values over?

    # attention mask and rope delta correspond to context len
    # so even if model is used and past_key_values is updated, 
    # don't update those 2
    attention_mask: torch.Tensor        # Full attention mask: [batch, cached_sequence_length].
    rope_deltas: torch.Tensor           # Qwen multimodal RoPE offset: [batch, 1].

    def __init__(
        self,
        past_key_values: DynamicCache,
        eos_token_ids: set[int] = field(default_factory=set),
        context_len: int = 0,
        context_logical_len: int = 0,
        finished: bool = False,
        attention_mask: torch.Tensor | None = None,
        rope_deltas: torch.Tensor | None = None,
    ):
        self.past_key_values = past_key_values
        self.context_len = context_len
        self.context_logical_len = context_logical_len
        self.eos_token_ids = eos_token_ids
        self.finished = finished
        self.attention_mask = attention_mask
        self.rope_deltas = rope_deltas

        if self.rope_deltas is None:
            self.rope_deltas = torch.zeros((1, 1), dtype=torch.long)

        if self.attention_mask is None:
            self.attention_mask = torch.ones((1, self.context_len), dtype=torch.long)

    def __str__(self):
        return (
            f"GenerationState(\n"
            f"  past_key_values_seq_length={self.past_key_values.get_seq_length()}, \n"
            f"  context_len={self.context_len}, \n"
            f"  context_logical_len={self.context_logical_len}, \n"
            f"  eos_token_ids={self.eos_token_ids}, \n"
            f"  finished={self.finished}, \n"
            f"  attention_mask_shape={self.attention_mask.shape if self.attention_mask is not None else None}, \n"
            f"  rope_deltas={self.rope_deltas}\n"
            f")"
        )

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
