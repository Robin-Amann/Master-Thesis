from src.utils.logger import logger
from src.config_types.benchmark import ModelConfig
from src.models.qwen_25_omni.adapter import QwenOmniAdapter, Video, Message
import torch
from src.models.shared_logic.qkv_extraction import calculate_attention_scores
from pathlib import Path
from src.visualization.attention import plot_attention_grid, plot_single_head
from src.data.loader import load_MVBench
from transformers import DynamicCache, Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration


def load_model() :
    model_config = ModelConfig(
    name="Qwen2_5_Omni",
    checkpoint="Qwen/Qwen2.5-Omni-3B",
    device="cuda",
    precision="float16"
    )

    model = QwenOmniAdapter(
        model_config, 
        # model_kwargs={ "attn_implementation": "eager" },
        processor_kwargs={
        # "min_pixels": 128*28*28, 
        # "max_pixels": 512 * 28 * 28
        }
    )

    return model


# model = load_model()
# dataset = load_MVBench()
# text = model.ask_question(dataset[0], Path("./data/MVBench/0"))
# print(text)

# Query normalization:
# Check whether your installed implementation uses q_norm or another transformation 
# between q_proj and RoPE. If it does, the captured query projection must undergo 
# that transformation as well. Otherwise, your reconstructed attention will differ 
# from the model's actual attention.

from src.press.guidance_press import GuidancePress
from src.models.shared_logic.generation import move_inputs_to_device

press = GuidancePress(
    compression_ratio=0.5,
    prompt_type="guidance",
    prompt_len=0,
    layer_cutoff=8,
)

model = load_model()
dataset = load_MVBench()
entry = dataset[0]

# guild context inputs
messages, text = model.build_messages(Message(video=Video(path=entry["video_path"])))
context_inputs = model.prepare_inputs(messages, text)
context_inputs = move_inputs_to_device(context_inputs, model.device)

# tensor([168, 168, 168])
t, h, w = context_inputs["position_ids"][:, 0, -1].tolist()
assert t == h == w
offset = t + 1

# build guidance inputs
guidance_inputs = model.hf_processor(text=["<|im_start|>user\nWhat happend in this video?<|im_end|>"], return_tensors="pt")
position_ids, _ = model.hf_model.thinker.get_rope_index(
    input_ids=guidance_inputs["input_ids"],
    attention_mask=guidance_inputs["attention_mask"]
)
position_ids = position_ids + offset 
guidance_inputs["position_ids"] = position_ids
guidance_inputs = move_inputs_to_device(guidance_inputs, model.device)

# build query inputs
messages, text = model.build_messages(Message(text=entry["text"]), keep_system_message=False)
query_inputs = model.prepare_inputs(messages, text, position_offset=offset)

initial_input = {
    "input_ids": torch.cat([context_inputs["input_ids"], guidance_inputs["input_ids"]], dim=-1),
    "attention_mask": torch.cat([context_inputs["attention_mask"], guidance_inputs["attention_mask"]], dim=-1),
    "pixel_values_videos": context_inputs["pixel_values_videos"],
    "video_grid_thw": context_inputs["video_grid_thw"],
    "video_second_per_grid": context_inputs["video_second_per_grid"],
    "position_ids": torch.cat([context_inputs["position_ids"], guidance_inputs["position_ids"]], dim=-1),
}

hf_model: Qwen2_5OmniForConditionalGeneration = model.hf_model

press.prompt_type = "guidance"
press.prompt_len = guidance_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    with press(hf_model.thinker):
        outputs = hf_model.thinker(
            **initial_input,
            use_cache=True,
            return_dict=True,
        )

    cache1 = outputs.past_key_values

del outputs


# press.prompt_type = "query"
# press.prompt_len = query_inputs["input_ids"].shape[-1]

# generation_kwargs = query_inputs | {
#     # "use_cache": True, ?
#     "past_key_values": cache1,
#     "use_cache": True,
#     "return_audio": False,
#     # "max_new_tokens": 128,
#     # "do_sample": False,
# }

# with torch.inference_mode():
#     with press(hf_model):
#         generated_ids = hf_model.generate(**generation_kwargs)

# answer_ids = generated_ids[:, query_inputs["input_ids"].shape[-1]:]
# answer = model.hf_processor.batch_decode(
#     answer_ids,
#     skip_special_tokens=True,
# )[0]

# print("Question:", entry["text"])
# print("Answer:", answer)


