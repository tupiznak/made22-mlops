from pathlib import Path

import click
import numpy as np
from sklearn.datasets import make_classification


@click.command()
@click.option('--features-file', help='Output file for save features.', required=True)
@click.option('--targets-file', help='Output file for save targets.', required=True)
def main(features_file: str, targets_file: str):
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    Path(features_file).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(features_file, X, fmt='%f', delimiter=",")
    np.savetxt(targets_file, y, fmt='%i', delimiter=',')


if __name__ == '__main__':
    main()
