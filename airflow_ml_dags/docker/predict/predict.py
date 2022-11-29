import pickle
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option('--features-file', help='Input file for read features.', required=True)
@click.option('--targets-file', help='Output file for save targets.', required=True)
@click.option('--model-path', help='Model path.', required=True)
def main(features_file: str, targets_file: str, model_path: str):
    Path(targets_file).parent.mkdir(parents=True, exist_ok=True)

    X = np.genfromtxt(features_file, delimiter=',', dtype=float)
    clf = pickle.load(open(model_path, 'rb'))

    predict = clf.predict(X)
    np.savetxt(targets_file, predict, fmt='%i', delimiter=',')


if __name__ == '__main__':
    main()
