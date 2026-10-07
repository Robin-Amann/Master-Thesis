import matplotlib as plt
import numpy as np
from pathlib import Path


def _build_full_matrix(total_len, prefill_len, prefill_head, head_idx, generation_steps, max_plot_tokens, plot_total_len, plot_prefill_len) :
    # Full square attention matrix.
    matrix = np.full(
        (total_len, total_len),
        np.nan,
        dtype=np.float32,
    )

    # Prefill block
    matrix[:prefill_len, :prefill_len] = prefill_head

    # Generation rows
    for step_idx, step_attention in enumerate(generation_steps):
        row = prefill_len + step_idx
        values = step_attention[head_idx, 0]

        matrix[row, :len(values)] = values

    # Optional visualization downsampling.
    if max_plot_tokens is not None and total_len > max_plot_tokens:
        row_indices = np.linspace(
            0,
            total_len - 1,
            plot_total_len,
        ).astype(int)

        col_indices = row_indices
        display_matrix = matrix[np.ix_(row_indices, col_indices)]
        boundary = plot_prefill_len

    else:
        display_matrix = matrix
        boundary = prefill_len

    return display_matrix, boundary


def _set_labels(ax, display_matrix, cmap, boundary, layer_idx, head_idx, n_layers) :
    im = ax.imshow(
        display_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # Prefill -> generation boundary.
    ax.axvline(
        boundary - 0.5,
        linewidth=1.2,
        color="red",
    )

    ax.axhline(
        boundary - 0.5,
        linewidth=1.2,
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
    prefill_attentions,
    generation_attentions,
    prefill_len,
    n_generated,
    plot_dir: Path,
    max_plot_tokens=None,
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

    n_layers = len(prefill_attentions)
    n_heads = prefill_attentions[0].shape[0]
    total_len = prefill_len + n_generated

    # Optional downsampling for visualization only.
    if max_plot_tokens is not None and total_len > max_plot_tokens:
        scale = max_plot_tokens / total_len
        plot_prefill_len = max(1, int(prefill_len * scale))
        plot_total_len = max_plot_tokens

        print(f"downsampling from {total_len} to {plot_total_len}")
    else:
        plot_prefill_len = prefill_len
        plot_total_len = total_len

    # Figure
    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(
            figsize_scale * n_heads,
            figsize_scale * n_layers,
        ),
        squeeze=False,
    )

    for layer_idx in range(n_layers):
        prefill = prefill_attentions[layer_idx]
        generation_steps = generation_attentions[layer_idx]

        for head_idx in range(n_heads):
            display_matrix, boundary = _build_full_matrix(total_len, prefill_len, prefill[head_idx], head_idx, generation_steps, max_plot_tokens, plot_total_len, plot_prefill_len)

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


    output_file = plot_dir / "attention_grid.png"

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print("Saved plot:")
    print(output_file)

    return output_file
