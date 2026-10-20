from pathlib import Path
import numpy as np
import torch

WORK_DIR = Path("/content/qwen_video_cache")
PLOT_DIR = WORK_DIR / "plots"

# model
ATTENTION_DTYPE = np.float16
TORCH_DTYPE = torch.bfloat16

# video
VIDEO_PATH = "/content/output.mkv"
VIDEO_CACHE_DIR = WORK_DIR / "video"
ATTN_DIR = WORK_DIR / "attention"


# VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# ATTN_DIR.mkdir(parents=True, exist_ok=True)
# PLOT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available() :
    print("GPU:", torch.cuda.get_device_name(0))
else :
    print("CPU")