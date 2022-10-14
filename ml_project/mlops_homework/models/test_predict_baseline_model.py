from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from mlops_homework.models.predict_baseline_model import predict

from mlops_homework.features.test_build_features import input_data

_ = input_data


@pytest.fixture
def features_file():
    with NamedTemporaryFile() as features_file:
        yield features_file


@pytest.fixture
def targets_file():
    with NamedTemporaryFile() as targets_file:
        yield targets_file


def test_model(features_file, targets_file, input_data):
    objects_count = 250
    input_data[0].head(n=objects_count).to_csv(features_file.name, index=False)
    features_file.seek(0)
    predict(features_file=features_file.name, targets_file=targets_file.name)
    targets_file.seek(0)
    targets = [int(s) for s in targets_file.readlines()]
    assert len(targets) == objects_count
    assert f1_score(input_data[1][:objects_count], targets) > 0.7
