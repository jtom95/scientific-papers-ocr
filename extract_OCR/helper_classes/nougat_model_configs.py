from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


class ModelType(Enum):
    SMALL = "checkpoint_small_model"
    BASE = "checkpoint_base_model"


class LocalSettings:
    def __init__(self, model_directory: Path = ".", model_type: ModelType = "small"):
        if isinstance(model_type, str):
            if model_type.lower() == "small":
                model_type = ModelType.SMALL
            elif model_type.lower() == "base":
                model_type = ModelType.BASE
            else:
                raise ValueError("model_type must be either 'small' or 'base'")

        self.model_type = model_type
        self._checkpoint_base_directory = Path(model_directory)
        self.checkpoint_directory = self._checkpoint_base_directory / self.model_type.value


@dataclass
class Configs:
    full_precision: bool = False
    batch_size: Optional[int] = None
    allow_skipping: bool = False
    markdown: bool = True
