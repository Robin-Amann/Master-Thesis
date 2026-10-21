from src.utils.logger import logger
from src.data.loader import load_hf_dataset
from src.config_types.benchmark import load_config, BenchmarkConfig
from src.procedures.eviction.loader import load_strategy
from src.models.qwen_25_omni.adapter import QwenOmniAdapter
import torch

config = load_config("./configs/benchmark.yaml")

def load_entry(config: BenchmarkConfig) :
    source = config.datasets[4].source

    logger.info("data source:")
    logger.info(source)

    dataset = load_hf_dataset(source, limit=10)
    logger.info("dataset:")
    logger.info(dataset)

    base = "/export/data1/ramann/hub/datasets--OpenGVLab--MVBench/snapshots/230a2d4fac8900333c61754641c7a13e069ac9c6/video/perception/videos/"
    # video_6480.mp4 is 15 sec.
    logger.info("video base:")
    logger.info(base)

    dataset = dataset.map(
        lambda row: {"video_path": base + row["video"], "text": row["question"]},
        remove_columns=["video", "question"]
    )

    entry = dataset[0]

    logger.info("\nentry:")
    logger.info(entry)

    return entry

def load_eviction_strategy(config: BenchmarkConfig) :
    strategy = config.eviction.strategies[0]
    args = strategy.args or {}
    eviction_strategy = load_strategy(strategy.name, args | {"compression_ratio": 0.5} )  
    return eviction_strategy


model = QwenOmniAdapter(config.model)
eviction_strategy = load_eviction_strategy(config)
entry = load_entry(config)


media_inputs = model.prepare_inputs(
    entry, 
    use_audio_in_video=False,
    use_text=False
)

text_inputs = model.prepare_inputs(
    entry, 
    use_video=False,
    remove_system_message=False
)

# inputs = model.prepare_inputs(
#     entry, 
#     use_audio_in_video=False,
# )
state = model.prefill(inputs=media_inputs)
logger.info(str([state.past_key_values.get_seq_length(), state.context_len, state.context_logical_len, state.eos_token_ids, state.finished]))

text_ids, text, state = model.generate(state=state, inputs=text_inputs)
# text_ids, text, state = model.generate(inputs=inputs)

logger.info(f"text ids {text_ids}, {text_ids.shape}")
logger.info(f"generated text: {text[0]}")
logger.info(str([state.past_key_values.get_seq_length(), state.context_len, state.context_logical_len, state.eos_token_ids, state.finished]))


# logger.info("\n\n### Prepare media inputs:")
# media_inputs = model.prepare_inputs(entry, use_audio_in_video=False, use_text=False)
# logger.info("\n\n### Prepare text inputs:")
# text_inputs = model.prepare_inputs(entry, use_video=False, remove_system_message=True)

# logger.info("\n\n### Prefill:")
# state = model.prefill(inputs=media_inputs, press=eviction_strategy)
# logger.info(f"seq_length: {state.past_key_values.get_seq_length()}")

# logger.info("\n\n### Predict:")
# generated_text, output_state = model.predict(state=state, inputs=text_inputs, return_state=True)
# logger.info(f"old seq_length: {state.past_key_values.get_seq_length()}")
# logger.info(f"new seq_length: {output_state.past_key_values.get_seq_length()}")
# logger.info(f"generated_text: {generated_text}")
