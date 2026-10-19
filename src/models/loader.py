from .model_adapter import ModelAdapter
from .qwen_25_omni.adapter import QwenOmniAdapter
from .qwen_25_vl.adapter import QwenVLAdapter
from src.config_types.model import ModelConfig
# can't put the model in the ModelAdapter file due to circulat imports
# Model --> Adapter 
# loader --> Adapter, Model

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "Qwen2_5_Omni_7B": QwenOmniAdapter,
    "Qwen2.5-VL-7B": QwenVLAdapter,
}

def load_model(config: ModelConfig) -> ModelAdapter :
    try:
        adapter_class = ADAPTERS[config.name]
    except KeyError:
        raise ValueError(f"Unknown model: {config.name}") from None

    adapter = adapter_class(config)

    return adapter
