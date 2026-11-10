from src.utils.logger import logger
from src.data.loader import load_hf_dataset, download_hf_dataset_files
from src.config_types.benchmark import load_config, BenchmarkConfig
from src.procedures.eviction.loader import load_strategy
from src.models.qwen_25_omni.adapter import QwenOmniAdapter
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info

config = load_config("./configs/benchmark.yaml")
model = QwenOmniAdapter(config.model)

def load_entry() :
    def p(x) :
        return x["level"] == "1" and x["group_type"] == "relevance" and x["second_head"] == "Frame-Only"

    # 312 questions over 78 videos | 112 has about 4 minutes [ids: 12-15]
    # /export/data1/ramann/huggingface/hub/Video-MME-v2/xxx.mp4
    dataset = load_hf_dataset(config.datasets[3].source).filter(lambda x: p(x), batched=False)
    # dataset = dataset.add_column("video_path", [ f"/export/data1/ramann/huggingface/hub/Video-MME-v2/{x}.mp4" for x in dataset["video_id"]])
    dataset = dataset.map(lambda item: {
        'video_path': f"/export/data1/ramann/huggingface/hub/Video-MME-v2/{item["video_id"]}.mp4",
        'text': f"{item["question"]}\noptions:\n{item["options"]}"
    }, remove_columns=["url", "group_type", "group_structure", "level", "second_head", "third_head"])
    # print(len(set(dataset["video_id"])))

    entry = dataset[12]
    return entry

entry = load_entry()

inputs = model.prepare_inputs(entry, use_audio_in_video=False)
N = inputs["input_ids"].shape[1]
t, h, w = inputs["video_grid_thw"][0].tolist() # 29, 40, 72
N_video = t * h * w / 4
N_text = N - N_video
video_start = 16 # maybe starts at 15?
token_per_frame = N_video / t

n_frames = 2
n = int(video_start + n_frames * token_per_frame)
new_inputs = {
    "input_ids": inputs["input_ids"][:, :n],
    "attention_mask": inputs["attention_mask"][:, :n],
    "pixel_values_videos": inputs["pixel_values_videos"][: h * w * n_frames, :],
    "video_grid_thw": inputs["video_grid_thw"][:, :],
    "video_second_per_grid": inputs["video_second_per_grid"][:]
}

new_inputs["video_grid_thw"][0][0] = n_frames

print(new_inputs)
text_ids, text, state = model.generate(inputs=new_inputs)
logger.info(str(state))
logger.info(text)

# media_inputs = model.prepare_inputs(entry, use_text=False)
# text_inputs = model.prepare_inputs(entry, use_video=False)

# logger.info(f"media inputs: {media_inputs["input_ids"].shape}")
# logger.info(f"text inputs: {text_inputs["input_ids"].shape}")
# state = model.prefill(inputs=media_inputs)
# logger.info(str(state))
# text_ids, text, state = model.generate(state=state, inputs=text_inputs)
# logger.info(text)
# logger.info(str(state))
