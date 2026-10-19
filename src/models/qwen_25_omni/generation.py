from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
import torch

def media_encoder(processor: Qwen2_5OmniProcessor, messages, use_audio_in_video=True):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=use_audio_in_video,
    )

    return processor(
        text=[text],
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    # input_ids	                Token IDs representing the formatted conversation.
    # attention_mask	        Indicates which text tokens are valid and which are padding.
    # input_features	        Audio features extracted from input audio.
    # feature_attention_mask    Mask identifying valid audio feature positions.
    # pixel_values	            Processed image tensors.
    # image_grid_thw	        Image temporal, height, and width grid metadata.
    # pixel_values_videos	    Processed video-frame tensors.
    # video_grid_thw	        Video grid metadata.
    # video_second_per_grid	    Temporal information used to align video features.


def prefill_validate_inputs(input_ids: torch.Tensor, model_kwargs: dict) :
    if input_ids.ndim != 2:
        raise ValueError(
            f"Expected input_ids with shape (batch, seq_len), "
            f"got {tuple(input_ids.shape)}."
        )

    if input_ids.shape[1] == 0:
        raise ValueError("Cannot prefill an empty input sequence.")

    # These arguments are managed by this function to ensure that
    # the old state and new inputs are combined consistently.
    reserved = {
        "past_key_values",
        "use_cache",
        "return_dict",
    }

    conflicts = reserved.intersection(model_kwargs)
    if conflicts:
        raise ValueError(
            f"model_kwargs cannot override managed arguments: "
            f"{sorted(conflicts)}"
        )

