from pydantic import BaseModel
from typing import Literal


class ModelConfig(BaseModel):
    name: str
    checkpoint: str
    device: str
    precision: Literal["float16", "bfloat16", "float32"]