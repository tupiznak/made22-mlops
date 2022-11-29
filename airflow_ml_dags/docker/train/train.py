import json
import os
import pickle
from pathlib import Path

import click
import mlflow
import numpy as np
from mlflow import log_metric
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

os.environ["AWS_ACCESS_KEY_ID"] = "11111111"
os.environ["AWS_SECRET_ACCESS_KEY"] = "22222222"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://0.0.0.0:9000/"
os.environ["MLFLOW_TRACKING_URI"] = f"http://0.0.0.0:5000/"


@click.command()
@click.option('--features-file-train', help='Input file for read features.', required=True)
@click.option('--targets-file-train', help='Input file for read targets.', required=True)
@click.option('--features-file-test', help='Input file for read features.', required=True)
@click.option('--targets-file-test', help='Input file for read targets.', required=True)
@click.option('--file-metrics-validate', help='Metrics file', required=True)
@click.option('--model-path', help='Model path.', required=True)
def main(features_file_train: str, targets_file_train: str,
         features_file_test: str, targets_file_test: str,
         file_metrics_validate: str, model_path: str):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(file_metrics_validate).parent.mkdir(parents=True, exist_ok=True)

    X_train = np.genfromtxt(features_file_train, delimiter=',', dtype=float)
    y_train = np.genfromtxt(targets_file_train, delimiter=',', dtype=int)
    X_test = np.genfromtxt(features_file_test, delimiter=',', dtype=float)
    y_test = np.genfromtxt(targets_file_test, delimiter=',', dtype=int)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)

    f1 = f1_score(y_test, clf.predict(X_test))
    json.dump(dict(f1=f1), open(file_metrics_validate, 'w'))

    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)

    log_metric("f1", f1)
    mlflow.sklearn.log_model(sk_model=clf, artifact_path='model', registered_model_name='baseline')


if __name__ == '__main__':
    main()
