from pathlib import Path

import click
import numpy as np
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--features-file-preprocess', help='Input file for read features.', required=True)
@click.option('--targets-file-preprocess', help='Input file for read targets.', required=True)
@click.option('--features-file-train', help='Output file for save features.', required=True)
@click.option('--targets-file-train', help='Output file for save targets.', required=True)
@click.option('--features-file-test', help='Output file for save features.', required=True)
@click.option('--targets-file-test', help='Output file for save targets.', required=True)
def main(features_file_preprocess: str, targets_file_preprocess: str,
         features_file_train: str, targets_file_train: str,
         features_file_test: str, targets_file_test: str):
    Path(features_file_train).parent.mkdir(parents=True, exist_ok=True)

    X = np.genfromtxt(features_file_preprocess, delimiter=',', dtype=float)
    y = np.genfromtxt(targets_file_preprocess, delimiter=',', dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    np.savetxt(features_file_train, X_train, fmt='%f', delimiter=',')
    np.savetxt(targets_file_train, y_train, fmt='%i', delimiter=',')
    np.savetxt(features_file_test, X_test, fmt='%f', delimiter=',')
    np.savetxt(targets_file_test, y_test, fmt='%i', delimiter=',')


if __name__ == '__main__':
    main()
