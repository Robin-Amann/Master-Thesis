
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.base_press import BasePress

# similar to ComposedPress

@dataclass
class LayerSplitPress(BasePress):
    """
    Apply press_1 to the first n_first_layers layers
    and press_2 to all subsequent layers.

    Layer indexing is zero-based:
        layers [0, ..., N-1] -> press_1
        layers [N, ..., L-1] -> press_2
    """

    press_1: BasePress
    press_2: BasePress
    n_first_layers: int

    def __post_init__(self):
        if self.n_first_layers < 0:
            raise ValueError("n_first_layers must be non-negative")

    def post_init_from_model(self, model: PreTrainedModel):
        """Initialize both underlying presses."""

        self.press_1.post_init_from_model(model)
        self.press_2.post_init_from_model(model)

        # Validate the requested layer split.
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )

        num_layers = len(language_model.layers)

        if self.n_first_layers > num_layers:
            raise ValueError(
                f"n_first_layers={self.n_first_layers} exceeds "
                f"the number of model layers ({num_layers})"
            )

    # def compress(
    #     self,
    #     module: nn.Module,
    #     hidden_states: torch.Tensor,
    #     keys: torch.Tensor,
    #     values: torch.Tensor,
    #     attentions: torch.Tensor,
    #     kwargs: dict,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Dispatch compression to the appropriate press.
    #     """

    #     layer_idx = module.layer_idx

    #     if layer_idx < self.n_first_layers:
    #         press = self.press_1
    #     else:
    #         press = self.press_2

    #     return press.compress(
    #         module=module,
    #         hidden_states=hidden_states,
    #         keys=keys,
    #         values=values,
    #         attentions=attentions,
    #         kwargs=kwargs,
    #     )


    def forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict,
        output: list,
    ):
        # Select the press according to the current layer.
        if module.layer_idx < self.n_first_layers:
            press = self.press_1
        else:
            press = self.press_2

        # Delegate the complete hook operation.
        return press.forward_hook(
            module, input, kwargs, output
        )



# from kvpress import KnormPress, SnapKVPress
# press = LayerSplitPress(
#     press_1=KnormPress(compression_ratio=0.2),
#     press_2=SnapKVPress(compression_ratio=0.5),
#     n_first_layers=8,
# )

# with press(model):
#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=128,
#     )
