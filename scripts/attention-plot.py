from utils.constants import *
from inference.load_model import load_model
from models.qwen_25_vl.video_encoder import encode_video
from inference.inference import ask_video
from plots.plots import plot_attention_grid
from torch.nn.functional import max_pool2d
from kvpress import KVSquaredPress

press = KVSquaredPress(
    compression_ratio=0.9,
    chunk_size=4096,
)

# model
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ATTN_IMP = "eager"
CHUNK_SIZE = 128 # should not go under 16

# video
VIDEO_FPS = 1.0
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 360 * 420
MAX_NEW_TOKENS = 64

# how many pixel to use for the prefill section of the attention blocks
# generation is mapped 1:1

model, processor = load_model(
    model_name=MODEL_NAME,
    attn_implementation=ATTN_IMP,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS
)

# generate video embeddings / load cached embeddings
video_cache = encode_video(
    model=model,
    processor=processor,
    video_path=VIDEO_PATH, 
    cache_file=VIDEO_CACHE_DIR / "video_embeddings.pt",
    fps=VIDEO_FPS,
    max_pixels=MAX_PIXELS
)

result = ask_video(
    model=model,
    processor=processor,
    chunk_size=CHUNK_SIZE,
    video_cache=video_cache,
    query="What is the person taking from the shelf?",
    max_new_tokens=MAX_NEW_TOKENS,
    save_attention=(ATTN_IMP == "eager") and True,

)

plot_attention_grid(
    data_dir=Path(result["query_dir"]),
    output_file=PLOT_DIR / f"{result["query_hash"]}_attention.png",
    # max_plot_tokens=1000, # for long videos
)
