import os
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
from hydra.core.config_store import ConfigStore

load_dotenv(find_dotenv())
PROJECT_PATH = os.environ.get('PROJECT_PATH', None)
if PROJECT_PATH is None:
    raise ModuleNotFoundError('need set PROJECT_PATH env')


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
    relative_path_to_data_raw: str
    relative_path_to_data_raw_csv: str
    relative_path_to_data_processed_csv: str
    relative_path_to_reports_eda: str
    relative_path_to_model_encoder: str
    relative_path_to_model: str


ConfigStore.instance().store(name="base_config", node=Config)
