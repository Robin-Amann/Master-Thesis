
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from kvpress.presses.base_press import BasePress, is_prefilling
from kvpress.utils import compute_n_kept
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb
from transformers import DynamicCache

@dataclass
class GuidancePress(BasePress):

    compression_ratio: float = 0.5
    prompt_type: Literal["guidance", "query"] = "guidance"
    prompt_len: int = 0
    layer_cutoff: int = 8

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1
        assert self.prompt_len >= 0
        assert self.layer_cutoff >= 0
        assert self.prompt_type in ("guidance", "query")

    def _reconstruct_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,    # [B, P, d]
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        B, P, _ = hidden_states.shape
        D = module.head_dim     # 128 (d_head)
        Hq = module.num_heads   # 16 (H)

        q: torch.Tensor = module.q_proj(hidden_states)    
        q = q.view(B, P, Hq, D).transpose(1, 2)     # [B, H, P, d_head]

        cos, sin = position_embeddings              # [3, B, P, d_head]

        # Qwen2.5-Omni multimodal RoPE
        # Dummy K; only the rotated Q is needed
        q, _ = apply_multimodal_rotary_pos_emb(q, q, cos, sin, module.rope_parameters["mrope_section"])

        return q    # [B, H, P, d_head]


    def _get_attention_scores(
        self,
        module: nn.Module,
        queries: torch.Tensor,      # [B, H, P, d_head]
        keys: torch.Tensor,         # [B, kv_heads, C, d_head]
    ) -> torch.Tensor:

        B, Hkv, C, D = keys.shape 
        Hq = queries.shape[1]

        if Hq % Hkv != 0:
            raise ValueError(
                f"Query heads ({Hq}) must be divisible by "
                f"KV heads ({Hkv})."
            )

        groups = Hq // Hkv

        # Keys have already received the correct M-RoPE transformation
        # during the original attention forward pass.
        logits = torch.matmul(
            queries.float(),                                                              
            keys.float().repeat_interleave(groups, dim=1).transpose(-1, -2),     # [B, H, d_head, C]
        ) * getattr(module, "scaling", D ** -0.5)       # [B, H, P, C]
        # scaling or D**-0.5

        # All context tokens precede the guidance/query tokens. Therefore, no additional causal mask is necessary.
        attention = torch.softmax(logits, dim=-1)

        # Aggregate query heads belonging to each KV head,
        # and aggregate over guidance/query tokens.
        # [B, H, P, C] --> [B, kv_heads, H / kv_heads, P, C] --> [B, kv_heads, C]
        scores = attention.reshape(B, Hkv, groups, queries.shape[2], C).sum(dim=(2, 3))  

        return scores # [B, kv_heads, C]


    def _select_tokens(
        self,
        keys: torch.Tensor,         # [B, kv_heads, C, d_head]
        values: torch.Tensor,       # [B, kv_heads, C, d_head]
        scores: torch.Tensor,       # [B, kv_heads, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, C, D = keys.shape

        if self.compression_ratio == 0:
            return keys, values

        n_kept = compute_n_kept(C, self.compression_ratio)

        if n_kept >= C:
            return keys, values

        indices = scores.topk(n_kept, dim=-1).indices           # [B, kv_heads, C']         C' = C * compression_ratio
        indices = indices.sort(dim=-1).values                   # [B, kv_heads, C']
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)   # [B, kv_heads, C', d_head]

        keys = keys.gather(dim=2, index=indices).contiguous()
        values = values.gather(dim=2, index=indices).contiguous()
        return keys, values


    def compress(
        self,
        module: nn.Module,              # Qwen2_5OmniAttention [Q, K, V, O projection + rotary_emb]
        hidden_states: torch.Tensor,    # [B, N', d]        N' only new positions
        keys: torch.Tensor,             # [B, kv_heads, N, d_head]
        values: torch.Tensor,           # [B, kv_heads, N, d_head]
        attentions: torch.Tensor,       # None
        kwargs: dict,                   
    ) -> tuple[torch.Tensor, torch.Tensor]:

        layer_idx: int = module.layer_idx
        q_len: int = hidden_states.shape[1]

        if self.prompt_type == "guidance":
            # Cache contains context || guidance.
            context_len = keys.shape[2] - self.prompt_len   # N - P         | N = C + P

            if context_len <= 0:
                raise ValueError("Invalid prompt_len: no context tokens remain.")

            context_keys = keys[:, :, :context_len, :]
            context_values = values[:, :, :context_len, :]

            # Only the first layers use guidance-based pruning. just remove guidance prompt
            if layer_idx >= self.layer_cutoff:
                return context_keys, context_values

            prompt_positions = tuple( x[:, :, context_len:, :] for x in kwargs["position_embeddings"] )
            prompt_queries = self._reconstruct_queries(module, hidden_states[:, context_len:, :], prompt_positions)   # [B, H, P, d_head]
            scores = self._get_attention_scores(module, prompt_queries, context_keys) # [B, kv_heads, C]
            return self._select_tokens(context_keys, context_values, scores)

        elif self.prompt_type == "query":
            # Cache contains compressed/uncompressed context || query.
            context_len = keys.shape[2] - q_len   # N - P         | N = C + P

            if context_len < 0:
                raise ValueError("Query compression requires a non-empty cached context.")

            context_keys = keys[:, :, :-q_len, :]       # [B, kv_heads, C', d_head]
            context_values = values[:, :, :-q_len, :]   # [B, kv_heads, C', d_head]

            query_keys = keys[:, :, -q_len:, :]         # [B, kv_heads, P, d_head]
            query_values = values[:, :, -q_len:, :]     # [B, kv_heads, P, d_head]

            # The first layers were already compressed using guidance.
            if layer_idx < self.layer_cutoff:
                return keys, values

            prompt_positions = tuple( x[:, :, -q_len:, :] for x in kwargs["position_embeddings"] )
            prompt_queries = self._reconstruct_queries(module, hidden_states[:, -q_len:, :], prompt_positions)   # [B, H, P, d_head]
            scores = self._get_attention_scores(module, prompt_queries, context_keys)
            compressed_keys, compressed_values = self._select_tokens(context_keys, context_values, scores)

            # Preserve the current query tokens in the KV cache.
            return (
                torch.cat([compressed_keys, query_keys], dim=2),
                torch.cat([compressed_values, query_values], dim=2),
            )

        raise ValueError(f"Unknown prompt_type: {self.prompt_type}")


    def forward_hook(
        self,
        module: nn.Module,      # Qwen2_5OmniAttention [Q, K, V, O projection + rotary_emb]
        input,                  # empty
        kwargs: dict,           # ['hidden_states', 'attention_mask', 'position_ids', 'past_key_values', 'output_attentions', 'use_cache', 'cache_position', 'position_embeddings']
        output,                 # tuple<tensor[[B, N (5069), 2048]], NoneType>  (hidden states output, attentions)
    ):
        # kwargs
        # - hidden_states       [B, N', d]
        # - attention_mask      None
        # - position_ids        None
        # - past_key_values     DynamicCache
        # - output_attentions   bool
        # - use_cache           bool
        # - cache_position      [N]
        # - position_embeddings <tensor[[3, B, N', d_head]], tensor[[3, B, N', d_head]]>   <sin, cos>

        hidden_states: torch.Tensor = kwargs["hidden_states"]   # [B, N, d]     d = H * d_head
        cache: DynamicCache = kwargs["past_key_values"]

        if cache is None:
            raise ValueError("GuidancePress requires a mutable KV cache.")

        layer_idx = module.layer_idx

        keys: torch.Tensor = cache.layers[layer_idx].keys       # [B, kv_heads, N, d_head]
        values: torch.Tensor = cache.layers[layer_idx].values   # [B, kv_heads, N, d_head]


        if (keys is None or values is None) and self.prompt_type == "query" : 
            raise ValueError("KV cache is empty. Need to prefill with guidance prompt before query")

        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        # Update the cache in-place.
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values

        return output
