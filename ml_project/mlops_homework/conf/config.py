from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    name: str
    test_split_size: float


@dataclass
class Config:
    model: ModelConfig
    random_state: int


ConfigStore.instance().store(name="base_config", node=Config)
