from transformers import AutoProcessor, PreTrainedModel
import torch

def load_model(
    model_name: str, 
    model_class: type[PreTrainedModel], 
    torch_dtype: torch.dtype,
    device: torch.device,
    model_kwargs: dict, 
    processor_kwargs: dict
) -> tuple[PreTrainedModel, AutoProcessor]: 

    print(f"load {model_name}")
    model: PreTrainedModel = model_class.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        **model_kwargs
    )
    model.eval()
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        **processor_kwargs
    )

    return model, processor