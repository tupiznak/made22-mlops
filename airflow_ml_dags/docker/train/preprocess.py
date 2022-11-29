from pathlib import Path

import click
import numpy as np


@click.command()
@click.option('--features-file-raw', help='Input file for read features.', required=True)
@click.option('--targets-file-raw', help='Input file for read targets.', required=True)
@click.option('--features-file', help='Output file for save features.', required=True)
@click.option('--targets-file', help='Output file for save targets.', required=True)
def main(features_file_raw: str, targets_file_raw: str, features_file: str, targets_file: str):
    Path(features_file).parent.mkdir(parents=True, exist_ok=True)

    X = np.genfromtxt(features_file_raw, delimiter=',', dtype=np.float)
    y = np.genfromtxt(targets_file_raw, delimiter=',', dtype=np.int)
    np.savetxt(features_file, X, fmt='%f', delimiter=',')
    np.savetxt(targets_file, y, fmt='%i', delimiter=',')


if __name__ == '__main__':
    main()
