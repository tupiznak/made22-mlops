import os

import hydra
from dotenv import load_dotenv, find_dotenv
from mlflow import set_tracking_uri, log_param

from mlops_homework.conf.config import Config
from .baseline.train_baseline_model import train_model

import dvc.api


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    load_dotenv(find_dotenv())
    set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    dvc_params = dvc.api.params_show()
    [setattr(cfg, k, v) for k, v in dvc_params.items()]
    log_param('test_split_size', cfg.model.test_split.size)
    log_param('random_state', cfg.random_state)
    log_param('encode_strategy', cfg.preprocessing.categorical_features)
    if cfg.model.name == 'baseline':
        train_model(test_split_size=cfg.model.test_split.size, random_state=cfg.random_state,
                    data_path=cfg.relative_path_to_data_processed_csv, model_path=cfg.relative_path_to_model)
