# procedures/eviction/base.py

from typing import Protocol


class EvictionStrategy(Protocol):
    name: str

    def evict(self, cache, ratio: float):
        """Return the modified cache and/or eviction indices."""
        ...