import logging
import pickle

import click
import pandas as pd

from mlops_homework.data import MODEL_PATH
from mlops_homework.features.build_features import DataTransformer
from mlops_homework.models.train_baseline_model import BaselineModel


@click.command
@click.option('--features-file', help='Input file with features in csv.')
@click.option('--targets-file', help='Output file for save targets.')
def predict(features_file: str, targets_file: str):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger('use model')

    logger.info('Load encoder...')
    with open(MODEL_PATH.joinpath('encoder_baseline.pkl'), 'rb') as file:
        encoder: DataTransformer = pickle.load(file)

    logger.info('Load model...')
    with open(MODEL_PATH.joinpath('baseline.pkl'), 'rb') as file:
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
    predict()
