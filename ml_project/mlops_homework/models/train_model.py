import hydra
from mlflow import set_tracking_uri, log_param

from mlops_homework.conf.config import Config
from .baseline.train_baseline_model import train_model

import dvc.api


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    set_tracking_uri('postgresql+psycopg2://mlflow:mlflow@mlflow-database:5432/mlflow')
    dvc_params = dvc.api.params_show()
    [setattr(cfg, k, v) for k, v in dvc_params.items()]
    log_param('test_split_size', cfg.model.test_split.size)
    log_param('random_state', cfg.random_state)
    log_param('encode_strategy', cfg.preprocessing.categorical_features)
    if cfg.model.name == 'baseline':
        train_model(test_split_size=cfg.model.test_split.size, random_state=cfg.random_state)
