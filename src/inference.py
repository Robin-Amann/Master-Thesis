from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from torch import device
import gc
import hashlib
from pathlib import Path

from src.qwen_utils import to_chat_messages
from src.constants import ATTENTION_DTYPE, ATTN_DIR
from src.utils import tensor_size

import logging
import math


def _build_query_inputs(
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
    messages = to_chat_messages(text=query)

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


def _move_inputs_to_lm_device(inputs: dict, lm_device: device):
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


def _construct_embeddings(
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


def _prefill(
    chunk_size: int,
    model: Qwen2_5_VLForConditionalGeneration,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    attention_dir: Path, # implies that attn_implementation == "eager" and save_attention == True
):   
    logging.info("Prefill...")
    prefill_len = input_ids.shape[1]
    logging.info(f"Prefill len: {prefill_len}")
    logging.info(f"Chunk Size: {chunk_size}")

    past_key_values = None # kv cache
    last_logits = None

    # L x H x px
    n_layers = model.config.text_config.num_hidden_layers
    n_heads = model.config.text_config.num_attention_heads
    n_px = math.ceil(prefill_len/chunk_size)
    attention = torch.zeros((n_layers, n_heads, n_px, n_px), device="cpu", dtype=torch.float16)
    logging.info(f"Attention: {attention} [L, H, px, px] {tensor_size(attention)}")

    for start_idx in range(0, prefill_len, chunk_size):
        chunk_num = start_idx // chunk_size
        logging.info(f"Chunk: {chunk_num}")
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
            logging.info(f"Chunk Attentions: L x {outputs.attentions[0]} [H, px, px] {tensor_size(outputs.attentions[0], "mb")}")
            
            # a: [1, num_heads, query_length, key_length]
            for layer_idx, a in enumerate(outputs.attentions):
                matrix = torch.nn.functional.max_pool2d(a[0], kernel_size=chunk_size, stride=chunk_size, ceil_mode=True).squeeze()
                # matrix: [num_heads, 1, chunk_num+1]
                attention[layer_idx, :, chunk_num, :chunk_num+1] = matrix.detach().float().cpu()
                del matrix

        # Force free GPU memory of current outputs
        del outputs
        logging.info(f"Garbage Collection...")
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(attention, attention_dir / "prefill.pt")
    del attention
    logging.info(f"Garbage Collection...")
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
        "prefill_px": n_px,
        "chunk_size": chunk_size,
    }


def _generate(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    prefill_state: dict,
    max_new_tokens: int,
    lm_device: device,
    attention_dir: Path,
) :

    n_layers = model.config.text_config.num_hidden_layers
    n_heads = model.config.text_config.num_attention_heads
    n_px = prefill_state["prefill_px"]
    prefill_len = prefill_state["prefill_len"]
    attention = {
        "prefill": torch.zeros((n_layers, n_heads, max_new_tokens, n_px), device="cpu", dtype=torch.float16),
        "generate": torch.zeros((n_layers, n_heads, max_new_tokens, max_new_tokens), device="cpu", dtype=torch.float16)
    }

    generated_ids = []
    eos_token_id = model.config.eos_token_id
    past_key_values = prefill_state["past_key_values"]
    next_token_id = prefill_state["next_token_id"]
    for step in range(max_new_tokens):
        print(f"generate token {step+1}")
        # next embedding
        token_id = int(next_token_id[0, 0])
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        token_embed = model.get_input_embeddings()(next_token_id)

        # attention mask
        current_len = prefill_len + len(generated_ids)
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
            # a: [1, num_heads, 1, key_length]
            for layer_idx, a in enumerate(outputs.attentions):
                # [num_heads, 1, prefill_len], [num_heads, 1, step+1]
                prefill, generation = torch.split(a[0], [prefill_len, a.shape[-1] - prefill_len], dim=-1)

                # [num_heads, 1, n_px]
                prefill_ = torch.nn.functional.max_pool2d(a[0], kernel_size=(1, prefill_state["chunk_size"]), stride=(1, prefill_state["chunk_size"]), ceil_mode=True).squeeze()
                attention["prefill"][layer_idx, :, step, :] = prefill_.detach().float().cpu()
                del prefill
                del prefill_

                attention["generation"][layer_idx, :, step, :step+1] = generation.detach().float().cpu()
                del generation
                
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    steps = len(generated_ids)
    attention["prefill"] = attention["prefill"][:, :, :steps, :steps]
    attention["generation"] = attention["generation"][:, :, :steps, :steps]

    torch.save(attention["prefill"], attention_dir / "gen_prefill.pt")
    torch.save(attention["generation"], attention_dir / "gen_generation.pt")
    del attention
    gc.collect()
    torch.cuda.empty_cache()

    generated_ids = torch.tensor(
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
    query: str,
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
    logging.info("Ask Video...")
    lm_device = model.get_input_embeddings().weight.device
    logging.info(f"lm device: {lm_device}")

    # input_ids | attention_mask | mm_token_type_ids | video_grid_thw | (second_per_grid_ts)

    inputs_cpu = _build_query_inputs(model, processor, query, video_cache)
    inputs = _move_inputs_to_lm_device(inputs_cpu, lm_device)

    
    logging.info("Construct Embeddings...")
    input_ids: torch.Tensor = inputs["input_ids"]
    logging.info(f"Input IDs: {input_ids} {tensor_size(input_ids)}")
    inputs_embeds, position_ids = _construct_embeddings(
        model=model,
        input_ids=input_ids,
        video_cache=video_cache,
        lm_device=lm_device,
        inputs=inputs,
    )
    logging.info(f"Inputs Embeds: {inputs_embeds} {tensor_size(inputs_embeds)}")
    logging.info(f"Position Ids: {position_ids} {tensor_size(position_ids)}")

    # prepare directories
    if save_attention:
        query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
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
        attention_dir=query_dir,
    )

    # answer, generated_ids
    result = _generate(
        model=model,
        processor=processor,
        prefill_state=prefill_state,
        max_new_tokens=max_new_tokens,
        lm_device=lm_device,
        attention_dir=query_dir,
    )

    result = {
        "query": query,
        **prefill_state,
        "answer": result["answer"],
        "generated_ids": result["generated_ids"],
    }

    if query_dir is not None:
        result["attention_dtype"] = str(ATTENTION_DTYPE)
        result["query_dir"] = query_dir.as_posix()
        result["query_hash"] = query_hash 

