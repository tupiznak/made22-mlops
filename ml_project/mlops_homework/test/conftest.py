from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from mlops_homework.conf.config import PROJECT_PATH


@pytest.fixture
def features_file():
    with NamedTemporaryFile() as features_file:
        yield features_file


@pytest.fixture
def targets_file():
    with NamedTemporaryFile() as targets_file:
        yield targets_file


@pytest.fixture
def config():
    initialize(version_base=None, config_path='../conf')
    cfg = compose(config_name="config")
    yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture
def input_data(config) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROJECT_PATH + config.relative_path_to_data_raw_csv)
    x_input = df.drop(columns=['condition'])
    targets = df['condition']
    return x_input, targets
