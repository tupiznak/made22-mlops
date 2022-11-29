import os
from pathlib import Path

import click
import mlflow
import numpy as np

os.environ["AWS_ACCESS_KEY_ID"] = "11111111"
os.environ["AWS_SECRET_ACCESS_KEY"] = "22222222"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://0.0.0.0:9000/"
os.environ["MLFLOW_TRACKING_URI"] = f"http://0.0.0.0:5000/"


@click.command()
@click.option('--features-file', help='Input file for read features.', required=True)
@click.option('--targets-file', help='Output file for save targets.', required=True)
@click.option('--model-path', help='Model path.', required=True)
def main(features_file: str, targets_file: str, model_path: str):
    Path(targets_file).parent.mkdir(parents=True, exist_ok=True)

    X = np.genfromtxt(features_file, delimiter=',', dtype=float)

    # clf = pickle.load(open(model_path, 'rb'))
    try:
        clf = mlflow.sklearn.load_model('models:/baseline/latest')
    except Exception as e:
        print('model not found !!!')
        raise e

    predict = clf.predict(X)
    np.savetxt(targets_file, predict, fmt='%i', delimiter=',')


if __name__ == '__main__':
    main()
