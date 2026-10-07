from constants import *
from qwen_vl_utils import process_vision_info
from src.qwen_utils import to_chat_messages
import gc
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from torch import device, FloatTensor, LongTensor

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
    video_embeds = video_embeds.detach().cpu().to(torch.float16)
    video_grid_thw = video_grid_thw.detach().cpu()


    return video_embeds, video_grid_thw


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

    # check if the cache already exists
    if cache_file.exists():
        print(f"Loading existing video cache:  {cache_file}")
        cache = torch.load(cache_file, map_location="cpu")
        return cache

    video_path = Path(video_path)
    assert video_path.exists(), f"Video does not exist: {video_path}"

    # encode video
    VISUAL_DEVICE = next(model.model.visual.parameters()).device
    messages = to_chat_messages({'path': video_path, "fps": fps, "max_pixels": max_pixels})
    pixel_values_videos, video_grid_thw, second_per_grid_ts = __encode_video(
        processor, 
        VISUAL_DEVICE,
        messages
    )

    # run encoder
    video_embeds, video_grid_thw = __run_video_encoder(
        model, 
        pixel_values_videos, 
        video_grid_thw
    )
    del pixel_values_videos

    cache = {
        "video_embeds": video_embeds,
        "video_grid_thw": video_grid_thw,
        "video_token_id": int(model.config.video_token_id),
        "model_name": MODEL_NAME, # TODO get from model directly
        "video_path": str(video_path),
        "fps": fps,
        "max_pixels": max_pixels,
    }

    if second_per_grid_ts is not None: 
        cache["second_per_grid_ts"] = second_per_grid_ts

    torch.save(cache, cache_file)

    gc.collect()
    torch.cuda.empty_cache()

    return cache


