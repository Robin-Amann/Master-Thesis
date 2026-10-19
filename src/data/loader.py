from src.config_types.benchmark import DatasetSourceConfig
from datasets import load_dataset, Dataset

def load_hf_dataset(config: DatasetSourceConfig, limit: int | None = None) -> Dataset:
    """Load a configured HF dataset; verify access and schema first."""
    
    kwargs = {
        "path": config.repo,
        "split": config.split,
    }
    if config.config is not None :
        kwargs["name"] = config.config

    ds: Dataset = load_dataset(**kwargs)

    if limit is not None and limit > 0 :
        ds = ds.select(range(min(limit, len(ds))))

    return ds


def group_by_media(dataset: Dataset) :
    return dataset