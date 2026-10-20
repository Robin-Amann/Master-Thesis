

# load data example
# load model
# (load eviction strategy)
# prefill model
# save state (indices)
# ask question
# calculate metric

from src.data.loader import load_hf_dataset
from src.config_types.benchmark import load_config
from src.models.loader import load_model
from src.procedures.eviction.loader import load_strategy

config = load_config("./configs/benchmark.yaml")

# video_audio
model = load_model(config.model)
strategy = config.eviction.strategies[0]
args = strategy.args or {}
eviction_strategy = load_strategy(strategy.name, args | {"compression_ratio": 0.5} )  

source = config.datasets[4].source
print(source)
dataset = load_hf_dataset(source, limit=10)
print(dataset)
# Dataset({
#     features: ['video_id', 'url', 'group_type', 'group_structure', 'question_id', 'question', 'options', 'answer', 'level', 'second_head', 'third_head'],
#     num_rows: 100
# })
# dataset = dataset.select_columns(['video_id', 'url', 'question_id', 'question', 'options'])


# for video_id in dataset.unique("video_id"):
#     video_questions = dataset.filter(
#         lambda row: row["video_id"] == video_id
#     )

#     video_url = video_questions[0]["url"]

    # media_inputs = model.prepare_inputs(media_entry["media"])

    # for entry in video_questions :
    #     pass

from datasets import Video
base = "/export/data1/ramann/hub/datasets--OpenGVLab--MVBench/snapshots/230a2d4fac8900333c61754641c7a13e069ac9c6/video/perception/videos/"

dataset = dataset.cast_column("video", Video(decode=False)).map(
    lambda row: {"video_path": row["video"]["path"], "text": row["question"]},
    remove_columns=["video", "question"]
)

entry = dataset[0]
entry["video_path"] = base + entry["video_path"]
print(entry)
media_inputs = model.prepare_inputs(entry, use_audio_in_video=False, use_text=False)
text_inputs = model.prepare_inputs(entry, use_video=False, remove_system_message=True)

state = model.prefill(inputs=media_inputs, press=eviction_strategy)
print(state.past_key_values.get_seq_length())
# generated_text, output_state = model.predict(state=state, inputs=text_inputs, return_state=True)
# print(output_state.past_key_values.get_seq_length())


# print(generated_text)
# print(output_state)

# 'input_ids':
# 'attention_mask':
# 'pixel_values_videos':
# 'video_grid_thw': tensor([[15, 40, 72]]), 
# 'video_second_per_grid': tensor([1.])


# from huggingface_hub import snapshot_download

# local_dir = snapshot_download(
#     repo_id="OpenGVLab/MVBench",
#     repo_type="dataset",
#     allow_patterns="video/*",
# )

# print(local_dir)


