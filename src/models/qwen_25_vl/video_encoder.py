from utils.constants import *
from qwen_vl_utils import process_vision_info
import gc
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from torch import device, FloatTensor, LongTensor
import logging
from utils.utils import tensor_size, to_qwen_chat_messages

def __encode_video(
    processor: Qwen2_5_VLProcessor, 
    visual_device: device,
    messages: list[dict]
): 
    # Obtain the actual video tensors.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    pixel_values_videos = inputs["pixel_values_videos"].to(visual_device)
    video_grid_thw = inputs["video_grid_thw"].to(visual_device)
    second_per_grid_ts = inputs.get("second_per_grid_ts", None)

    if second_per_grid_ts is not None:
        second_per_grid_ts = second_per_grid_ts.detach().cpu()
    
    # free space
    del inputs

    print("Video tensor:", tuple(pixel_values_videos.shape))
    print("Video grid:", video_grid_thw)

    return pixel_values_videos, video_grid_thw, second_per_grid_ts


def __run_video_encoder(
    model: Qwen2_5_VLForConditionalGeneration, 
    pixel_values_videos: FloatTensor, 
    video_grid_thw: LongTensor
):
    # Run the Qwen vision encoder ONCE.
    with torch.inference_mode():
        video_features = model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    # get_video_features() returns one pooler_output per video.
    # For one video we take the first element.
    video_embeds = video_features.pooler_output[0]

    # free space
    del video_features

    # Move the cached representation to CPU.
    video_embeds_cpu = video_embeds.detach().cpu().to(torch.float16)
    video_grid_thw_cpu = video_grid_thw.detach().cpu()


    return video_embeds_cpu, video_grid_thw_cpu


def encode_video(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    video_path: Path,
    cache_file: Path,
    fps: float,
    max_pixels: int,
):
    """
    Decode and vision-encode the video once.

    The resulting cache contains:
        video_embeds
        video_grid_thw
        second_per_grid_ts
        video_token_id
        metadata
    """
    logging.info("Encode Video...")
    
    # check if the cache already exists
    if cache_file.exists():
        print(f"Loading existing video cache:  {cache_file}")
        cache = torch.load(cache_file, map_location="cpu")
        return cache

    video_path = Path(video_path)
    assert video_path.exists(), f"Video does not exist: {video_path}"

    # encode video
    VISUAL_DEVICE = next(model.model.visual.parameters()).device
    logging.info(f"Visual Device: {VISUAL_DEVICE}")

    messages = to_qwen_chat_messages({'path': video_path, "fps": fps, "max_pixels": max_pixels})
    pixel_values_videos, video_grid_thw, second_per_grid_ts = __encode_video(
        processor, 
        VISUAL_DEVICE,
        messages
    )
    logging.info(f"Pixel Values: {pixel_values_videos}")
    logging.info(f"Video Grid: {video_grid_thw} [T, H, W] {tensor_size(video_grid_thw)} (store in cache)")
    if second_per_grid_ts is not None: 
        logging.info(f"Second Per Grid: {second_per_grid_ts} [T, S] {tensor_size(second_per_grid_ts)} (store in cache)")

    # run encoder
    logging.info(f"Encoder...")
    video_embeds_cpu, video_grid_thw_cpu = __run_video_encoder(
        model, 
        pixel_values_videos, 
        video_grid_thw
    )
    del pixel_values_videos
    logging.info(f"Video Embeds: {video_embeds_cpu} {tensor_size(video_embeds_cpu)} (store in cache)")

    cache = {
        "video_embeds": video_embeds_cpu,
        "video_grid_thw": video_grid_thw_cpu,
        "video_token_id": int(model.config.video_token_id),
        "model_name": model.config.name_or_path,
        "video_path": str(video_path),
        "fps": fps,
        "max_pixels": max_pixels,
    }

    if second_per_grid_ts is not None: 
        cache["second_per_grid_ts"] = second_per_grid_ts

    logging.info("Save to Cache...")
    torch.save(cache, cache_file)

    logging.info("Garbage Collection...")
    gc.collect()
    torch.cuda.empty_cache()

    return cache


