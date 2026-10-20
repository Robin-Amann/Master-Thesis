from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from torch import device
from src.utils.utils import to_qwen_chat_messages


def build_query_inputs(
    model: Qwen2_5_VLForConditionalGeneration, 
    processor: Qwen2_5_VLProcessor,
    query: str, 
    video_cache: dict
):
    """
    Construct the Qwen multimodal token sequence using cached
    video embeddings rather than processing the video again.
    """

    video_token_id = int(model.config.video_token_id)
    messages = to_qwen_chat_messages(text=query)

    # Generate the textual chat template without decoding the video.
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize only the textual template.
    tokenized = processor.tokenizer(
        chat_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = tokenized["input_ids"]

    # The tokenizer produces one video placeholder.
    # Replace it by one token per cached video embedding.
    video_mask = input_ids == video_token_id

    num_placeholder_tokens = int(video_mask.sum())

    if num_placeholder_tokens != 1:
        raise RuntimeError(
            f"Expected exactly one video placeholder, "
            f"found {num_placeholder_tokens}."
        )

    n_video_tokens = video_cache["video_embeds"].shape[0]

    new_ids = []

    for token in input_ids[0]:
        if int(token) == video_token_id:
            new_ids.extend([video_token_id] * n_video_tokens)
        else:
            new_ids.append(int(token))

    torch.Tensor()
    input_ids = torch.tensor(
        [new_ids],
        dtype=torch.long,
    )

    # Token type:
    #   0 = text
    #   1 = image
    #   2 = video
    #
    # This is required for Qwen's 3-D multimodal RoPE.
    mm_token_type_ids = torch.zeros_like(input_ids)

    mm_token_type_ids[input_ids == video_token_id] = 2

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "video_grid_thw": video_cache["video_grid_thw"].clone(),
        "second_per_grid_ts": (
            video_cache["second_per_grid_ts"].clone()
            if "second_per_grid_ts" in video_cache
            else None
        ),
    }


def move_inputs_to_lm_device(inputs: dict, lm_device: device):
    """
    Move tensors required by the language model to the LM device.
    """
    result = {}

    for k, v in inputs.items():
        if v is None:
            result[k] = None
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(lm_device)
        else:
            result[k] = v

    return result


def construct_embeddings(
        model: Qwen2_5_VLForConditionalGeneration, 
        input_ids: torch.Tensor, 
        video_cache: dict,
        lm_device: device,
        inputs: dict
    ):

    mm_token_type_ids = inputs["mm_token_type_ids"]
    video_grid_thw = inputs["video_grid_thw"]
    second_per_grid_ts = inputs["second_per_grid_ts"]
    attention_mask = inputs["attention_mask"]

    text_embeds: torch.Tensor = model.get_input_embeddings()(input_ids)

    # Cached video embeddings -> LM device and dtype.
    video_embeds: torch.Tensor = video_cache["video_embeds"].to(
        device=lm_device,
        dtype=text_embeds.dtype,
    )

    video_positions = input_ids[0] == model.config.video_token_id

    if int(video_positions.sum()) != video_embeds.shape[0]:
        raise RuntimeError(
            "Number of video placeholder tokens does not match "
            "the number of cached video embeddings."
        )

    # Replace video placeholder embeddings with cached visual embeddings.
    inputs_embeds = text_embeds.clone()
    inputs_embeds[0, video_positions, :] = video_embeds

    # Calculate multimodal RoPE positions.
    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
    )

    # Cache the RoPE offset for autoregressive generation.
    model.model.rope_deltas = rope_deltas

    return inputs_embeds, position_ids

