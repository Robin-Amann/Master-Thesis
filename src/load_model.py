from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from src.constants import TORCH_DTYPE

def load_model(
    model_name: str, 
    attn_implementation: str, 
    min_pixels: int,
    max_pixels: int,
) -> tuple[Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor]: 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return model, processor
