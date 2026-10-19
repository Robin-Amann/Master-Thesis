import kvpress
from typing import Optional, Any

def load_strategy(name: str, args: dict[str, Any]) :
    """args can be a empty dict if no arguments are necessary"""
    
    metric_map = {
        "StreamingLLM": kvpress.StreamingLLMPress,
        "KVzip": kvpress.KVzipPress,
        "KeyDiff": kvpress.KeyDiffPress,
        "KVzap": kvpress.KVzapPress,
        # "KVsquared": 
        # "HybridKV": 
        # "TGV-KV"
    }

    if args is None :
        args = {}

    if name not in metric_map :
        raise ValueError("eviction strategy not found")

    return metric_map[name](**args)
