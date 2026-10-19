
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict
from .model import ModelConfig
from typing import Optional, Any

class ExperimentConfig(BaseModel):
    name: str
    seed: int
    output_dir: Path


Modality = Literal["audio", "image", "video", "video_audio"]

class DatasetSourceConfig(BaseModel):
    repo: str
    config: str | None
    split: str

class DatasetConfig(BaseModel):
    name: str
    source: DatasetSourceConfig
    modality: Modality

class EvictionStrategy(BaseModel) :
    name: str
    args: Optional[dict[str, Any]]

class EvictionConfig(BaseModel):
    strategies: list[EvictionStrategy]
    ratios: list[float]

class GenerationConfig(BaseModel):
    max_new_tokens: int
    do_sample: bool


# ---------- Root config ----------

class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    model: ModelConfig
    datasets: list[DatasetConfig]
    eviction: EvictionConfig
    generation: GenerationConfig
    evaluation_metrics: dict[Modality, str]


def load_config(path: str | Path) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return BenchmarkConfig.model_validate(raw)
