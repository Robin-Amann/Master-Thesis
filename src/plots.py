import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch


def _set_labels(ax, display_matrix, cmap, boundary, layer_idx, head_idx, n_layers) :
    im = ax.imshow(
        display_matrix.detach().cpu().numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # Prefill -> generation boundary.
    ax.axvline(
        boundary -0.25,
        linewidth=0.5,
        color="red",
    )

    ax.axhline(
        boundary -0.25,
        linewidth=0.5,
        color="red",
    )

    ax.set_title(
        f"L{layer_idx} H{head_idx}",
        fontsize=7,
    )

    # Only label outer axes.
    if layer_idx == n_layers - 1:
        ax.set_xlabel(
            "Key position",
            fontsize=7,
        )

    if head_idx == 0:
        ax.set_ylabel(
            "Query position",
            fontsize=7,
        )

    ax.tick_params(
        axis="both",
        labelsize=5,
    )


def plot_attention_grid(
    data_dir: Path,
    output_file: Path,
    cmap="viridis",
    figsize_scale=2.5,
):
    """
    Plot an H x L grid.

    Rows    = transformer layers
    Columns = attention heads

    Each subplot contains:

        prefill + generation attention

    A red line marks the boundary between the prefill
    context and generated tokens.
    """

    print("load data")
    attention_prefill: torch.Tensor = torch.load(data_dir / 'prefill.pt')                 # [L, H, N, N]
    attention_gen_prefill: torch.Tensor = torch.load(data_dir / 'gen_prefill.pt')         # [L, H, M, N]
    attention_gen_generation: torch.Tensor = torch.load(data_dir / 'gen_generation.pt')   # [L, H, M, M]

    n_layers, n_heads, n_gen, n_prefill = attention_gen_prefill.shape

    top_right = torch.zeros(n_layers, n_heads, n_prefill, n_gen, device="cpu", dtype=attention_prefill.dtype)

    attentions = torch.cat([
        torch.cat([attention_prefill, top_right], dim=-1),                      # [L, H, N, N+M]
        torch.cat([attention_gen_prefill, attention_gen_generation], dim=-1),   # [L, H, M, N+M]
    ], dim=-2)                                                                  # [L, H, N+M, N+M]

    # Figure
    print("create plot")
    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(
            figsize_scale * n_heads,
            figsize_scale * n_layers,
        ),
        squeeze=False,
    )

    print("add matricies")
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            display_matrix, boundary = attentions[layer_idx, head_idx, :, :], n_prefill
            ax = axes[layer_idx, head_idx]
            _set_labels(ax, display_matrix, cmap, boundary, layer_idx, head_idx, n_layers)

    # Global labels
    fig.suptitle(
        "Qwen2.5-VL-3B attention: layers × heads",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.01,
        "Red line = end of prefill / beginning of generation",
        ha="center",
        fontsize=11,
    )

    plt.tight_layout(
        rect=[0, 0.025, 1, 0.97]
    )

    print("save plot")
    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    # plt.show()