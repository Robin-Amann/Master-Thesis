from src.utils.logger import logger
from src.config_types.benchmark import ModelConfig
from src.models.qwen_25_omni.adapter import QwenOmniAdapter
import torch
from src.models.shared_logic.qkv_extraction import calculate_attention_scores
from pathlib import Path
from src.visualization.attention import plot_attention_grid, plot_single_head
from src.data.loader import load_MVBench

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


model = load_model()
dataset = load_MVBench()
text = model.ask_question(dataset[0], Path("./data/MVBench/0"))
print(text)



# 3. Query normalization
# Check whether your installed implementation uses q_norm or another transformation 
# between q_proj and RoPE. If it does, the captured query projection must undergo 
# that transformation as well. Otherwise, your reconstructed attention will differ 
# from the model's actual attention.

from src.press.guidance_press import GuidancePress


press = GuidancePress(
    ratio=0.5,
    prompt_type="guidance",
    prompt_len=0,
    layer_cutoff=8,

)

input = ...
guidance_prompt = ...

press.prompt_type = "guidance"
press.prompt_len = len(guidance_prompt)

# only compresses first half of the context and removes last prompt_len kv pairs
with press(model) :
    cache1 = model([input, guidance_prompt])

query = ...

press.prompt_type = "query"
press.prompt_len = len(query)

with press(model) :
    cache2 = model.generate(cache1, query)


query2 = ...
press.prompt_type = "query"
press.query_prompt_len = len(query2)
with press(model) :
    cache2 = model.generate(cache2, query2)



