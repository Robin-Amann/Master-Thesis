from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
import torch
from src.utils.logger import logger
from pathlib import Path
import json

# inputs:
# input_ids                 [batch, input tokens]   Token IDs representing the formatted conversation.
# attention_mask            [batch, input tokens]   Indicates which text tokens are valid and which are padding.
# input_features                                    Audio features extracted from input audio.
# feature_attention_mask                            Mask identifying valid audio feature positions.
# pixel_values                                      Processed image tensors.
# image_grid_thw                                    Image temporal, height, and width grid metadata.
# pixel_values_videos       [T*H*W, d_token]        Processed video-frame tensors.
# video_grid_thw            [num_videos, 3]         Video grid metadata. 3 --> t, h, w
# video_second_per_grid	    [num_videos]            Temporal information used to align video features. Represents how many seconds of video correspond to a single temporal grid unit.
#  
# input tokens = token_video + token_text
# token_video = T * H/2 * W/2
# d_token = 1176 

# input ids: (T: some token)
# ['<|im_start|>', ..., '<|im_end|>', ..., '<|im_start|>', ..., '<|vision_bos|>', '<|VIDEO|>', ..., '<|VIDEO|>', '<|vision_eos|>', '<|im_end|>', 'Ċ'] 
# ['<|im_start|>', ..., '<|im_end|>', ..., '<|im_start|>', ..., '<|vision_bos|>', '<|VIDEO|>', ... (video), '<|VIDEO|>', '<|vision_eos|>', ... (question), '<|im_end|>', 'Ċ']

def chunk_inputs(inputs: dict, n_frames: int) :
    logger.info("chunking:")
    logger.info(n_frames)
    N = inputs["input_ids"].shape[1]
    t, h, w = inputs["video_grid_thw"][0].tolist() # 29, 40, 72
    logger.info(f"t: {t}")
    n_frames = min(n_frames, t)
    N_video = t * h * w / 4
    N_text = N - N_video
    video_start = inputs["input_ids"][0][:20].tolist().index(151656) # first occurrence of <|VIDEO|>. should be position 15
    token_per_frame = N_video / t
    n = int(n_frames * token_per_frame) # 1455 = 15 + 2 * 720

    start = 0
    end = int(video_start) + n
    frame_start = 0
    frame_end = n_frames
    result = []
    while start < N :
        logger.info(f"{frame_start} to {frame_end}")
        if frame_end >= t :
            end = N
        result.append({
            "input_ids": inputs["input_ids"][:, start:end],
            "attention_mask": inputs["attention_mask"][:, start:end],
            "pixel_values_videos": inputs["pixel_values_videos"][h * w * frame_start: h * w * frame_end, :],
            "video_grid_thw": inputs["video_grid_thw"].clone(),
            "video_second_per_grid": inputs["video_second_per_grid"],
            "position_ids": inputs["position_ids"][:, :, start:end]
        })
        result[-1]["video_grid_thw"][0][0] = frame_end - frame_start
        start = end
        end = min(end + int(n_frames * token_per_frame), N)
        frame_start = frame_end
        frame_end = min(frame_end + n_frames, t)

    return result

def write_result(file_path: Path, new_entry: dict) :
    with file_path.open("r") as file:
        content = file.read()

    data: list[dict] = json.loads(content) if len(content.strip()) >= 2 else []
    data.append(new_entry)

    with file_path.open("w") as file:
        json.dump(data, file, indent=4)