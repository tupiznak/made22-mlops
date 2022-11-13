from sklearn.metrics import f1_score

from mlops_homework.conf.config import PROJECT_PATH
from mlops_homework.models.baseline.predict_baseline_model import predict


def test_model(features_file, targets_file, input_data, config):
    objects_count = 250
    input_data[0].head(n=objects_count).to_csv(features_file.name, index=False)
    features_file.seek(0)
    predict(features_file=features_file.name, targets_file=targets_file.name,
            encoder_path=PROJECT_PATH + config.relative_path_to_model_encoder,
            model_path=PROJECT_PATH + config.relative_path_to_model)
    targets_file.seek(0)
    targets = [int(s) for s in targets_file.readlines()]
    assert len(targets) == objects_count
    assert f1_score(input_data[1][:objects_count], targets) > 0.7
