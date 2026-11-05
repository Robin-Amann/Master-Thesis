from typing import Any
from transformers import PreTrainedModel, AutoProcessor
from src.config_types.model import ModelConfig
import torch
from src.models.shared_logic.generation import GenerationState
from abc import ABC, abstractmethod

class ModelAdapter:

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hf_model: PreTrainedModel = None

    
    @abstractmethod
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
        # entry > chat form > processor.apply_chat_template
        raise NotImplementedError()


    @torch.inference_mode()
    @abstractmethod
    def prefill( 
        self,
        inputs: dict[str, Any],
        state: GenerationState | None = None,
        press=None,
        model_kwargs: dict[str, Any] | None = None
    ) -> GenerationState: 
        raise NotImplementedError()


    @torch.inference_mode()
    @abstractmethod
    def generate(self, state: GenerationState) -> int | None:
        # multiple generate
        raise NotImplementedError()


    @abstractmethod
    def predict(
        self,
        state: GenerationState | None = None,
        inputs: dict[str, Any] | None = None,
        max_new_tokens: int = 128, # TODO: constant
        press=None,
        model_kwargs: dict[str, Any] | None = None,
        return_state: bool = False,
    ) -> str | tuple[str, GenerationState]:
        raise NotImplementedError()


