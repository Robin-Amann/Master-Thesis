from torch import nn
import torch
from pathlib import Path
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb
import numpy as np
from transformers import DynamicCache
from src.utils.utils import shrink_attention_tensor
import os
from src.visualization.attention import plot_attention_grid

def register_hooks(layers: nn.ModuleList) :
    n = len(layers)

    # Store Q, cosine and sine tensors on CPU.
    captured = { i: {"q": [], "cos": [], "sin": []} for i in range(n) }
    handles = []

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        # Capture Q projection (before RoPE).
        def make_q_hook(idx):
            def hook(module, inputs, output):
                captured[idx]["q"].append(output.detach().cpu())
            return hook

        handles.append(attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))

        # Capture rotary embeddings passed to attention.
        def make_attn_pre_hook(idx):
            def hook(module, args, kwargs):

                position_embeddings = kwargs.get("position_embeddings")

                if position_embeddings is not None:
                    cos, sin = position_embeddings

                    captured[idx]["cos"].append(cos.detach().cpu())
                    captured[idx]["sin"].append(sin.detach().cpu())
            return hook

        handles.append(
            attn.register_forward_pre_hook(
                make_attn_pre_hook(layer_idx),
                with_kwargs=True
            )
        )

    return captured, handles


def save_queries_and_keys(
    captured: dict[int, dict[str, list[torch.Tensor]]], 
    cache: DynamicCache,
    save_dir: Path
) :
    for i, layer in enumerate(cache.layers):
        d = save_dir / f"layer_{i}"
        d.mkdir(parents=True, exist_ok=True)
        # q:        [B, N', n_query_heads * head_dim] a tensor for each generation step
        # sin, cos: [3, B, N', head_dim]

        for name in ("q", "cos", "sin"):
            tensors = captured[i][name]

            if tensors:
                tensor = torch.cat(tensors, dim=-2)
                torch.save(tensor, d / f"{name}.pt")

        # keys: [B, n_kv_heads, N, head_dim]
        torch.save(layer.keys.detach().cpu(), save_dir / f"layer_{i}" / "k.pt")
        # torch.save(value.detach().cpu(), save_dir / f"layer_{i}_value.pt")


def calculate_attention_scores(
    save_dir: Path, 
    layers: nn.ModuleList, 
    mrope_section: list[int],
    media_boundaries: list[int]
) :
    attn_dir = save_dir / "attention"
    attn_dir.mkdir(exist_ok=True)

    device = "cuda"
    a, b, c = 0, 0, 0
    first, last = media_boundaries
    num_layers = len(layers)

    def load_tensor(name: str, layer_idx: int) -> torch.Tensor :
        return torch.load(
            save_dir / "qk" / f"layer_{layer_idx}" / f"{name}.pt",
            map_location=device,
            weights_only=True
        )

    # Process each layer independently
    for layer_idx, layer in enumerate(layers):
        print(f"Processing layer {layer_idx}/{num_layers}")
        attn: nn.Module = layer.self_attn

        q = load_tensor("q", layer_idx)     # [B, N, H * head_dim]
        k = load_tensor("k", layer_idx)   # [B, n_kv_heads, N, head_dim]
        cos = load_tensor("cos", layer_idx) # [3, B, N, head_dim]
        sin = load_tensor("sin", layer_idx) # [3, B, N, head_dim]

        batch_size, seq_len, _ = q.shape
        num_heads = attn.num_heads
        num_kv_heads = attn.num_key_value_heads
        head_dim = attn.head_dim

        # Reshape Q.
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE to Q.
        # K from the cache is already RoPE-transformed.
        out = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        q: torch.Tensor = out[0] # [B, H, N, d_head]

        # Expand KV heads for grouped-query attention.
        if num_heads != num_kv_heads:
            repeats = num_heads // num_kv_heads # 8
            k = k.repeat_interleave(repeats, dim=1)  # [B, H, N, d_head]

        N = q.shape[2] 

        # Attention output: [B, Hq, Nq, Nk]
        attention = np.lib.format.open_memmap(
            attn_dir / f"layer_{layer_idx}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(batch_size, num_heads, N, N)
        )

        chunk_size = 32
        scale = attn.scaling

        k_transposed = k.transpose(-2, -1)

        # For cached generation, Q positions may start after
        # the cached prefix. Adjust this if processing chunks
        # from an existing KV cache.
        key_positions = torch.arange(N, device=device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            q_chunk = q[:, :, start:end, :]

            scores = torch.matmul(q_chunk.float(), k_transposed.float()) * scale
            query_positions = torch.arange(start, end, device=device)

            causal_mask = key_positions[None, :] <= query_positions[:, None]
            scores.masked_fill_(~causal_mask[None, None, :, :], float("-inf"))
            scores = torch.softmax(scores, dim=-1)

            attention[:, :, start:end, :] = scores.cpu().numpy()

            del scores, q_chunk

        attention.flush()
        del q, k, cos, sin, k_transposed, attention
        torch.cuda.empty_cache()

        attention = torch.tensor(np.load(attn_dir / f"layer_{layer_idx}.npy"))

        a, b, c = first, last+1-(first), N-(last+1)
        for reduction in ('mean', 'max', 'sum') :
            t = shrink_attention_tensor(attention, a, b, c , 36, reduction)
            torch.save(t, attn_dir / f"layer_{layer_idx}_{reduction}.pt")

        os.remove(attn_dir / f"layer_{layer_idx}.npy")


    for reduction in ('mean', 'max', 'sum') :
        attentions = torch.cat( [ torch.load(attn_dir / f"layer_{i}_{reduction}.pt") for i in range(num_layers) ], dim=0)
        torch.save(attentions, attn_dir / f"qk_attentions_{reduction}.pt")
        plot_attention_grid(output_file=attn_dir / f"{reduction}.jpeg", attentions=attentions, boundaries=[a, a + b / 36])

    for layer in range(num_layers) :
        for reduction in ('mean', 'max', 'sum') :
            os.remove(attn_dir / f"layer_{layer}_{reduction}.pt")
