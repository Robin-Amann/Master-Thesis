from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from torch import device, Tensor
import numpy as np
import gc
import hashlib
from pathlib import Path

from src.qwen_utils import to_chat_messages
from src.constants import ATTENTION_DTYPE, ATTN_DIR


def _build_query_inputs(
    model: Qwen2_5_VLForConditionalGeneration, 
    processor: Qwen2_5_VLProcessor,
    question: str, 
    video_cache: dict
):
    """
    Construct the Qwen multimodal token sequence using cached
    video embeddings rather than processing the video again.
    """

    video_token_id = int(model.config.video_token_id)
    messages = to_chat_messages(text=question)

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

    input_ids = Tensor(
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


def _move_inputs_to_lm_device(inputs: dict, lm_device: device):
    """
    Move tensors required by the language model to the LM device.
    """
    result = {}

    for k, v in inputs.items():
        if v is None:
            result[k] = None
        elif isinstance(v, Tensor):
            result[k] = v.to(lm_device)
        else:
            result[k] = v

    return result


def _construct_embeddings(
        model: Qwen2_5_VLForConditionalGeneration, 
        input_ids: Tensor, 
        video_cache: dict,
        lm_device: device,
        inputs: dict
    ):

    mm_token_type_ids = inputs["mm_token_type_ids"]
    video_grid_thw = inputs["video_grid_thw"]
    second_per_grid_ts = inputs["second_per_grid_ts"]
    attention_mask = inputs["attention_mask"]

    text_embeds: Tensor = model.get_input_embeddings()(input_ids)

    # Cached video embeddings -> LM device and dtype.
    video_embeds: Tensor = video_cache["video_embeds"].to(
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


def _save_attention(attentions: list[Tensor], save_dir: Path, filename: str) :
    for layer_idx, a in enumerate(attentions):
        layer_chunk_dir = save_dir / f"layer_{layer_idx:02d}"
        layer_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        matrix = a[0].detach().float().cpu().numpy().astype(ATTENTION_DTYPE)
        np.save(layer_chunk_dir / filename, matrix)
        del matrix


def _prefill(
    chunk_size: int,
    model: Qwen2_5_VLForConditionalGeneration,
    input_ids: Tensor,
    inputs_embeds: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    attention_dir: Path # implies that attn_implementation == "eager" and save_attention == True
):   
    prefill_len = input_ids.shape[1]

    past_key_values = None # kv cache
    last_logits = None  
    for start_idx in range(0, prefill_len, chunk_size):
        chunk_num = start_idx // chunk_size
        print(f"Chunk: {chunk_num}")
        end_idx = min(start_idx + chunk_size, prefill_len)

        chunk_input_ids = input_ids[:, start_idx:end_idx]
        chunk_inputs_embeds = inputs_embeds[:, start_idx:end_idx, :]
        chunk_attention_mask = attention_mask[:, :end_idx]
        chunk_position_ids = position_ids[:, :, start_idx:end_idx]

        outputs = model(
            input_ids=chunk_input_ids,
            inputs_embeds=chunk_inputs_embeds,
            attention_mask=chunk_attention_mask,
            position_ids=chunk_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=(attention_dir is not None),
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :] # [batch_size, sequence_length, vocabulary_size]

        # attentions: 
        #   layers x (batch_size, num_attention_heads, query_length, key_length) 
        #   layers x [1, num_heads, query_length, key_length]
        if attention_dir is not None and outputs.attentions is not None:
            _save_attention(outputs.attentions, attention_dir, f"prefill_chunk_{chunk_num:04d}.npy")

        # Force free GPU memory of current outputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    # Last prefill logits determine the first generated token.
    next_token_id = torch.argmax(                
        last_logits, # next_token_logits
        dim=-1,
        keepdim=True,
    )

    return {
        "past_key_values": past_key_values, # list of tuples of (key, value) for each layer
        "prefill_len": prefill_len, # int
        "new_token_id": next_token_id, # Tensor
    }


def _generate(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    prefill_state: dict,
    max_new_tokens: int,
    lm_device: device,
    attention_dir: Path
) :
    generated_ids = []
    eos_token_id = model.config.eos_token_id
    past_key_values = prefill_state["past_key_values"]
    next_token_id = prefill_state["next_token_id"]
    for step in range(max_new_tokens):
        # next embedding
        token_id = int(next_token_id[0, 0])
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        token_embed = model.get_input_embeddings()(next_token_id)

        # attention mask
        current_len = prefill_state["prefill_len"] + len(generated_ids)
        new_attention_mask = torch.ones(
            (1, current_len),
            dtype=torch.long,
            device=lm_device,
        )

        outputs = model(
            input_ids=next_token_id,
            inputs_embeds=token_embed,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=(attention_dir is not None),
            return_dict=True,
        )

        if (attention_dir is not None and outputs.attentions is not None):
            _save_attention(outputs.attentions, attention_dir, f"generation_step_{step:04d}.npy")

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        if (step + 1) % 10 == 0:
            print(f"  generated {step + 1} tokens")

    generated_ids = Tensor(
        generated_ids,
        dtype=torch.long,
    )

    answer = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return {
        "answer": answer,
        "generated_ids": generated_ids,
    }


@torch.inference_mode()
def ask_video(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    chunk_size: int,
    video_cache: dict,
    question: str,
    max_new_tokens: int,
    save_attention=True,
):
    """
    Ask a question about the cached video with chunked prefill
    and streaming disk-saves for attention matrices to conserve RAM/VRAM.

    Returns:
        answer
        generated_ids
        prefill_attentions (mocked or loaded if needed)
        generation_attentions
    """

    lm_device = model.get_input_embeddings().weight.device

    # input_ids | attention_mask | mm_token_type_ids | video_grid_thw | (second_per_grid_ts)

    inputs_cpu = _build_query_inputs(model, processor, question, video_cache)
    inputs = _move_inputs_to_lm_device(inputs_cpu, lm_device)

    input_ids:  Tensor = inputs["input_ids"]
    inputs_embeds, position_ids = _construct_embeddings(
        model=model,
        input_ids=input_ids,
        video_cache=video_cache,
        lm_device=lm_device,
        inputs=inputs,
    )

    # prepare directories
    if save_attention:
        query_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        query_dir = ATTN_DIR / query_hash
        query_dir.mkdir(parents=True, exist_ok=True)
    else :
        query_dir = None

    attention_mask = inputs["attention_mask"]

    # past_key_values, prefill_len, new_token_id
    prefill_state = _prefill(
        chunk_size=chunk_size,
        model=model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        attention_dir=query_dir
    )

    # answer, generated_ids
    result = _generate(
        model=model,
        processor=processor,
        prefill_state=prefill_state,
        max_new_tokens=max_new_tokens,
        lm_device=lm_device,
        attention_dir=query_dir
    )

    result = {
        "question": question,
        **prefill_state,
        "answer": result["answer"],
        "generated_ids": result["generated_ids"],

    }

    if query_dir is not None:
        result["attention_dtype"] = str(ATTENTION_DTYPE)

