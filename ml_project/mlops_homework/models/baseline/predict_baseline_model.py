import logging
import pickle

import click
import pandas as pd
from hydra import compose, initialize

from mlops_homework.features.build_features import DataTransformer
from mlops_homework.models.baseline.model import BaselineModel


@click.command()
@click.option('--features-file', help='Input file with features in csv.', required=True)
@click.option('--targets-file', help='Output file for save targets.', required=True)
def main(features_file: str, targets_file: str):
    initialize(version_base=None, config_path='../../conf')
    cfg = compose(config_name="config")
    predict(features_file, targets_file, cfg.relative_path_to_model_encoder, cfg.relative_path_to_model)


def predict(features_file: str, targets_file: str, encoder_path: str, model_path: str):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger('use model')

    logger.info('Load encoder...')
    with open(encoder_path, 'rb') as file:
        encoder: DataTransformer = pickle.load(file)

    logger.info('Load model...')
    with open(model_path, 'rb') as file:
        model: BaselineModel = pickle.load(file)

    logger.info('Load batch...')
    x_batch = pd.read_csv(features_file)

    logger.info('Transform batch...')
    x_batch = encoder.transform(x_batch.to_numpy())

    logger.info('Predict data...')
    targets = model.predict(x_batch)

    logger.info('Save data...')
    with open(targets_file, 'w') as file:
        file.writelines([f'{y}\r\n' for y in targets])

    logger.info('Model predicted')


if __name__ == '__main__':
    main()
