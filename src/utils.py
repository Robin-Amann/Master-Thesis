import torch


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
