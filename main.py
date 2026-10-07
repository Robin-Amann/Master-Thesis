from src.constants import *
from src.load_model import load_model
from src.video_encoder import encode_video
from src.helper_functions import ask_video
from src.plots import plot_attention_grid


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

# print()
# print("Video encoding complete.")
# print("Cached video tokens:", video_embeds.shape[0])
# print("Embedding dimension:", video_embeds.shape[1])
# print()
# print("Saved:", cache_file)

VIDEO_EMBEDS = video_cache["video_embeds"]
VIDEO_GRID_THW = video_cache["video_grid_thw"]
SECOND_PER_GRID_TS = video_cache["second_per_grid_ts"]

print("Cached video embedding shape:", VIDEO_EMBEDS.shape)
print("Video grid:", VIDEO_GRID_THW)
# prefill
# compress cache (optional)
# generate

result = ask_video(
    model=model,
    processor=processor,
    chunk_size=CHUNK_SIZE,
    attn_implementation=ATTN_IMP,
    video_cache=video_cache,
    question="What is the person taking from the shelf?",
    max_new_tokens=MAX_NEW_TOKENS,
)


plot_attention_grid(
    result, # TODO: load attention from src and compress
    output_file=PLOT_DIR / "question_1_attention.png",
    # max_plot_tokens=1000, # for long videos
)
