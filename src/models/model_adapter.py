from typing import Any
from transformers import PreTrainedModel
from src.config_types.model import ModelConfig
import torch
from .shared_logic.generation import GenerationState
from typing import Optional


class ModelAdapter:

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hf_model: PreTrainedModel = None
    
    def prepare_inputs(self, entry: dict, use_audio_in_video: bool = True) :
        # entry > chat form > processor.apply_chat_template
        raise NotImplementedError()

    @torch.inference_mode()
    def prefill( 
        self,
        inputs: dict[str, Any],
        state: Optional[GenerationState] = None,
        press=None,
        model_kwargs: Optional[dict[str, Any]] = None
    ) -> GenerationState: 
        raise NotImplementedError()

    @torch.inference_mode()
    def generate(self, state: GenerationState) -> Optional[int]:
        # multiple generate
        raise NotImplementedError()

    def predict(
        self,
        state: Optional[GenerationState] = None,
        inputs: Optional[dict[str, Any]] = None,
        max_new_tokens: int = 128, # TODO: constant
        press=None,
        model_kwargs: Optional[dict[str, Any]] = None,
        return_state: bool = False,
    ) -> str | tuple[str, GenerationState]:
        raise NotImplementedError()


