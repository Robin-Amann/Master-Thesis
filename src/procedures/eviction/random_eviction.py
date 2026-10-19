# procedures/eviction/random.py

import torch


class RandomEviction:
    name = "random"

    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)

    def evict(self, cache, ratio: float):
        # Determine eligible cache positions.
        # Sample positions to evict.
        # Apply eviction to the cache.
        # Return eviction indices.
        ...