import hydra

from mlops_homework.conf.config import Config
from .baseline.train_baseline_model import train_model

import dvc.api


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    dvc_params = dvc.api.params_show()
    [setattr(cfg, k, v) for k, v in dvc_params.items()]
    print(cfg)
    if cfg.model.name == 'baseline':
        train_model(test_split_size=cfg.model.test_split_size, random_state=cfg.random_state)
