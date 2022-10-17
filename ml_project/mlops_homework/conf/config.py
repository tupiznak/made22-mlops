from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class SplitConfig:
    size: float


@dataclass
class ModelConfig:
    name: str
    test_split: SplitConfig


@dataclass
class PreprocessingConfig:
    categorical_features: str


@dataclass
class Config:
    model: ModelConfig
    preprocessing: PreprocessingConfig
    random_state: int


ConfigStore.instance().store(name="base_config", node=Config)
