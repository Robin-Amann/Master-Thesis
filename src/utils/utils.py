import torch
from pathlib import Path
import yaml
import subprocess
from typing import Literal

def tensor_size(tensor: torch.Tensor, unit: str = "gb", output: str = "str", percision: int=3) -> str | float:
    """
    Return the memory size of a PyTorch tensor in the requested unit.

    Units:
        byte, kb, mb, gb

    Uses binary units:
        1 kb = 10^3 bytes
        1 mb = 10^6 bytes
        1 gb = 10^8 bytes

    Output:
        str: e.g. 1.5 gb
        float: e.g. 1.5
    """
    units = {
        "byte": 1,
        "kb": 10**3,
        "mb": 10**6,
        "gb": 10**9,
    }

    unit = unit.lower()

    if unit not in units:
        raise ValueError(f"Unknown unit '{unit}'. Choose from: {list(units)}")

    nbytes = tensor.numel() * tensor.element_size()
    result = nbytes / units[unit]
    if output == "str" :
        return f"{round(result, percision)} {unit}"

    return result


def make_uniform_causal_tensor(L, H, N, device=None, dtype=torch.float32):
    """
    Returns a tensor of shape [L, H, N, N] where each [N, N] slice is:

        [[1,    0,    0,   ...],
         [1/2,  1/2,  0,   ...],
         [1/3,  1/3,  1/3, ...],
         ...
        ]

    All [L, H] slices are identical.
    """
    i = torch.arange(N, device=device)
    mask = i[:, None] >= i[None, :]
    
    matrix = mask.to(dtype) / (i[:, None] + 1).to(dtype)
    
    return matrix.unsqueeze(0).unsqueeze(0).expand(L, H, N, N).clone()


def make_random_row_stochastic_tensor(
    L, H, M, N, device=None, dtype=torch.float32
):
    """
    Returns a tensor of shape [L, H, M, N].

    Each [M, N] matrix contains random nonnegative values,
    with every row summing to 1.
    """
    x = torch.rand(L, H, M, N, device=device, dtype=dtype)
    x = x / x.sum(dim=-1, keepdim=True)
    return x


def to_qwen_chat_messages(video: dict | None = None, text: str = "Describe this video."):
    content = []
    if video is None:
        content.append({
            "type": "video",
            "video": "PLACEHOLDER",
        })
    else :
        content.append({
            "type": "video",
            "video": f"file://{video['path'].resolve()}",
            "fps": video["fps"],
            "max_pixels": video["max_pixels"],
        })

    content.append({
        "type": "text",
        "text": text,
    })

    return [ {
        "role": "user",
        "content": content
    } ]


def load_config(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def save_config(config: dict, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def get_video_length(path: str) :
    duration = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])

    duration = float(duration)
    return duration


def nested_type(obj):
    if isinstance(obj, torch.Tensor):
        return f"tensor[{list(obj.shape)}]"

    if isinstance(obj, list):
        if not obj:
            return "list<>"
        return f"list<{nested_type(obj[0])}>"

    if isinstance(obj, tuple):
        if not obj:
            return "tuple<>"
        return f"tuple<{', '.join(nested_type(x) for x in obj)}>"

    if isinstance(obj, dict):
        if not obj:
            return "dict<>"
        return "dict<" + ", ".join(
            f"{nested_type(k)}: {nested_type(v)}"
            for k, v in obj.items()
        ) + ">"

    return type(obj).__name__


def shrink_attention_tensor(x: torch.Tensor, n_text_start: int, n_video: int, n_text_end: int, factor: int, reduction: Literal["mean", "max", "sum"] = "mean"):

    assert n_video % factor == 0, "b must be divisible by n"
    _, _, H, W = x.shape
    assert H == W == n_text_start + n_video + n_text_end, "Input tensor has wrong shape"

    sizes = [n_text_start, n_video, n_text_end]
    rows = torch.split(x, sizes, dim=-2)
    blocks = [ torch.split(row, sizes, dim=-1) for row in rows ]

    def shrink(cell: torch.Tensor, dim: int) :
        if reduction == "mean":
            cell = cell.mean(dim=dim)
        elif reduction == "max":
            cell = cell.max(dim=dim).values
        elif reduction == "sum":
            cell = cell.sum(dim=dim)
        return cell

    result = []
    for y, row in enumerate(blocks) :
        result_row = []
        for x, cell in enumerate(row) :
            if x != 1 and y != 1 :
                result_row.append(cell)
                continue

            print(cell.shape)
            d1, d2, d3, d4 = cell.shape

            if x == 1:
                cell = cell.reshape(d1, d2, d3, n_video // factor, factor)
                cell = shrink(cell, 4)
                d1, d2, d3, d4 = cell.shape

            if y == 1:
                cell = cell.reshape(d1, d2, n_video // factor, factor, d4)
                cell = shrink(cell, 3)

            result_row.append(cell)
        result.append(torch.cat(result_row, dim=3))    
    return torch.cat(result, dim=2)



def join_attentions(attentions) :
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]

    # Total sequence length
    seq_len = attentions[0][0].shape[-1] + len(attentions) - 1

    full_attentions = torch.zeros(
        num_layers,
        num_heads,
        seq_len,
        seq_len,
        dtype=torch.float32,
    )

    for layer_idx in range(num_layers):

        # Prefill attention: [1, H, S, S]
        prefill = attentions[0][layer_idx][0].float()

        prompt_len = prefill.shape[-1]

        full_attentions[
            layer_idx, :, :prompt_len, :prompt_len
        ] = prefill

        # Decoding attention
        for step_idx, step in enumerate(attentions[1:], start=1):

            attn = step[layer_idx][0, :, 0, :].float()

            query_pos = prompt_len + step_idx - 1
            key_len = attn.shape[-1]

            full_attentions[
                layer_idx, :, query_pos, :key_len
            ] = attn

    return full_attentions