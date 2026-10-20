from typing import Any
import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from pathlib import Path
from src.models.model_adapter import ModelAdapter
from torch import device
import gc
import hashlib

from src.utils.constants import ATTENTION_DTYPE, ATTN_DIR
from src.utils.utils import tensor_size
from src.models.qwen_25_vl.inference import build_query_inputs, construct_embeddings, move_inputs_to_lm_device
import logging
import math
from src.models.shared_logic.load_model import load_model

class QwenVLAdapter(ModelAdapter):

    def __init__(self, checkpoint, device):
        super().__init__(checkpoint, device)
        model, processor = load_model(
            model_name=checkpoint, 
            torch_dtype=torch.bfloat16, 
            device=device,
            model_kwargs={"attn_implementation": "eager"}, 
            processor_kwargs={"min_pixels": ..., "max_pixels": ...}
        )
        self.hf_model: Qwen2_5_VLForConditionalGeneration = model
        self.hf_processor: Qwen2_5_VLProcessor = processor

    
    def prefill(
        self,
        chunk_size: int,
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
        n_layers = self.hf_model.config.text_config.num_hidden_layers
        n_heads = self.hf_model.config.text_config.num_attention_heads
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

            outputs = self.hf_model(
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


    def generate(
        self,
        prefill_state: dict,
        max_new_tokens: int,
        lm_device: device,
        attention_dir: Path,
    ) :

        n_layers = self.hf_model.config.text_config.num_hidden_layers
        n_heads = self.hf_model.config.text_config.num_attention_heads
        n_px = prefill_state["prefill_px"]
        prefill_len = prefill_state["prefill_len"]
        attention = {
            "prefill": torch.zeros((n_layers, n_heads, max_new_tokens, n_px), device="cpu", dtype=torch.float16),
            "generate": torch.zeros((n_layers, n_heads, max_new_tokens, max_new_tokens), device="cpu", dtype=torch.float16)
        }

        generated_ids = []
        eos_token_id = self.hf_model.config.eos_token_id
        past_key_values = prefill_state["past_key_values"]
        next_token_id = prefill_state["next_token_id"]
        for step in range(max_new_tokens):
            print(f"generate token {step+1}")
            # next embedding
            token_id = int(next_token_id[0, 0])
            generated_ids.append(token_id)

            if eos_token_id is not None and token_id == eos_token_id:
                break

            token_embed = self.hf_model.get_input_embeddings()(next_token_id)

            # attention mask
            current_len = prefill_len + len(generated_ids)
            new_attention_mask = torch.ones(
                (1, current_len),
                dtype=torch.long,
                device=lm_device,
            )

            outputs = self.hf_model(
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

        answer = self.hf_processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return {
            "answer": answer,
            "generated_ids": generated_ids,
        }


    def predict(self, batch: Any) -> Any:
        return self.model(batch)


    @torch.inference_mode()
    def ask_video(
        self,
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
        lm_device = self.hf_model.get_input_embeddings().weight.device
        logging.info(f"lm device: {lm_device}")

        # input_ids | attention_mask | mm_token_type_ids | video_grid_thw | (second_per_grid_ts)

        inputs_cpu = build_query_inputs(query, video_cache)
        inputs = move_inputs_to_lm_device(inputs_cpu, lm_device)

        
        logging.info("Construct Embeddings...")
        input_ids: torch.Tensor = inputs["input_ids"]
        logging.info(f"Input IDs: {input_ids} {tensor_size(input_ids)}")
        inputs_embeds, position_ids = construct_embeddings(
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
        prefill_state = self.prefill(
            chunk_size=chunk_size,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attention_dir=query_dir,
        )

        # answer, generated_ids
        result = self.generate(
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

