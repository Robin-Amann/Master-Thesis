
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from kvpress.presses.base_press import BasePress, is_prefilling
from kvpress.utils import compute_n_kept
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb

@dataclass
class GuidancePress(BasePress):
    """
    Two-stage, query-conditioned KV cache compression.

    Stage 1 (guidance):
        Compress context KV in the first `layer_cutoff` layers using
        guidance-to-context attention. Remove guidance KV in all layers.

    Stage 2 (query):
        Keep the first `layer_cutoff` layers unchanged.
        Compress context KV in the remaining layers using
        query-to-context attention.

    Qwen2.5-Omni M-RoPE is handled using the position_embeddings
    supplied by the attention module. Keys are already RoPE-rotated.
    """

    compression_ratio: float = 0.5
    prompt_type: Literal["guidance", "query"] = "guidance"
    prompt_len: int = 0
    layer_cutoff: int = 8

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1
        assert self.prompt_len >= 0
        assert self.layer_cutoff >= 0
        assert self.prompt_type in ("guidance", "query")

    # ---------------------------------------------------------
    # Attention scoring

    def _get_query_states(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:

        B, Q, _ = hidden_states.shape
        D = module.head_dim
        Hq = module.num_heads

        q = module.q_proj(hidden_states)
        q = q.view(B, Q, Hq, D).transpose(1, 2)

        cos, sin = kwargs["position_embeddings"]

        # Qwen2.5-Omni multimodal RoPE
        q, _ = apply_multimodal_rotary_pos_emb(
            q,
            q,  # Dummy K; only the rotated Q is needed
            cos,
            sin,
            module.rope_parameters["mrope_section"],
        )

        return q


    def _get_attention_scores(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        context_keys: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Calculate query-to-context attention.

        Args:
            hidden_states: [B, Q, hidden_size]
            context_keys: [B, Hkv, C, D]

        Returns:
            Importance scores: [B, Hkv, C]
        """
        q = self._get_query_states(module, hidden_states, kwargs)

        B, Hkv, C, D = context_keys.shape
        Hq = q.shape[1]

        if Hq % Hkv != 0:
            raise ValueError(
                f"Query heads ({Hq}) must be divisible by "
                f"KV heads ({Hkv})."
            )

        groups = Hq // Hkv

        # Keys have already received the correct M-RoPE transformation
        # during the original attention forward pass.
        logits = torch.matmul(
            q.float(),
            context_keys.float().transpose(-1, -2),
        ) * getattr(module, "scaling", D ** -0.5)

        # All context tokens precede the guidance/query tokens.
        # Therefore, no additional causal mask is necessary.
        attention = torch.softmax(logits, dim=-1)

        # Aggregate query heads belonging to each KV head,
        # and aggregate over guidance/query tokens.
        scores = attention.reshape(
            B, Hkv, groups, q.shape[2], C
        ).mean(dim=(2, 3))

        return scores

    # ---------------------------------------------------------
    # Token selection
    # ---------------------------------------------------------

    def _select_tokens(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the highest-scoring context tokens independently
        for each KV head.

        Args:
            keys, values: [B, Hkv, C, D]
            scores: [B, Hkv, C]
        """
        B, Hkv, C, D = keys.shape

        if self.compression_ratio == 0:
            return keys, values

        n_kept = compute_n_kept(C, self.compression_ratio)

        if n_kept >= C:
            return keys, values

        # Select the n_kept highest-scoring tokens.
        indices = scores.topk(n_kept, dim=-1).indices

        # Restore original token ordering.
        indices = indices.sort(dim=-1).values

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)

        return (
            keys.gather(dim=2, index=indices).contiguous(),
            values.gather(dim=2, index=indices).contiguous(),
        )
    # ---------------------------------------------------------
    # Main compression
    # ---------------------------------------------------------

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        layer_idx = module.layer_idx
        q_len = hidden_states.shape[1]

        if self.prompt_type == "guidance":

            # Cache contains context || guidance.
            context_len = keys.shape[2] - self.prompt_len

            if context_len <= 0:
                raise ValueError(
                    "Invalid prompt_len: no context tokens remain."
                )

            context_keys = keys[:, :, :context_len]
            context_values = values[:, :, :context_len]

            # Only the first layers use guidance-based pruning.
            if layer_idx >= self.layer_cutoff:
                return context_keys, context_values

            scores = self._get_attention_scores(
                module,
                hidden_states[:, context_len:, :],
                context_keys,
                {
                    **kwargs,
                    "position_embeddings": (
                        kwargs["position_embeddings"][0][:, context_len:],
                        kwargs["position_embeddings"][1][:, context_len:],
                    ),
                },
            )

            return self._select_tokens(
                context_keys, context_values, scores
            )

        elif self.prompt_type == "query":

            # Cache contains compressed/uncompressed context || query.
            context_len = keys.shape[2] - q_len

            if context_len <= 0:
                raise ValueError(
                    "Query compression requires a non-empty cached context."
                )

            context_keys = keys[:, :, :context_len]
            context_values = values[:, :, :context_len]

            query_keys = keys[:, :, context_len:]
            query_values = values[:, :, context_len:]

            # The first layers were already compressed using guidance.
            if layer_idx < self.layer_cutoff:
                return keys, values

            scores = self._get_attention_scores(
                module,
                hidden_states,
                context_keys,
                kwargs,
            )

            compressed_keys, compressed_values = self._select_tokens(
                context_keys,
                context_values,
                scores,
            )

            # Preserve the current query tokens in the KV cache.
            return (
                torch.cat([compressed_keys, query_keys], dim=2),
                torch.cat([compressed_values, query_values], dim=2),
            )

        raise ValueError(
            f"Unknown prompt_type: {self.prompt_type}"
        )

    # ---------------------------------------------------------
    # Custom forward hook
    # ---------------------------------------------------------

    def forward_hook(
        self,
        module: nn.Module,
        input,
        kwargs: dict,
        output,
    ):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]

        if cache is None:
            raise ValueError(
                "GuidancePress requires a mutable KV cache."
            )

        q_len = hidden_states.shape[1]
        prefilling = is_prefilling(
            kwargs["cache_position"], q_len
        )

        if self.prompt_type == "guidance" and not prefilling:
            return output

        if self.prompt_type == "query" and prefilling:
            raise ValueError(
                "Query mode requires a pre-existing guidance cache. "
                "Pass the context cache and query tokens together."
            )

        layer_idx = module.layer_idx

        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values

        if keys is None or values is None:
            raise ValueError("KV cache is empty.")

        keys, values = self.compress(
            module,
            hidden_states,
            keys,
            values,
            output[1],
            kwargs,
        )

        # Update the cache in-place.
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values

        return output
