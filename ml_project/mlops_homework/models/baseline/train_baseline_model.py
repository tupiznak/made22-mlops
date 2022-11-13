import logging
import pickle

import mlflow
import pandas as pd
from mlflow import log_metric
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from mlops_homework.conf.config import ModelConfig
from mlops_homework.models.baseline.model import BaselineModel


def train_model(config_model: ModelConfig, data_path: str, model_path: str):
    test_split_size = config_model.test_split
    random_state = config_model.random_state
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger('train model')

    logger.info('Prepare model...')

    df = pd.read_csv(data_path)
    x_input = df.drop(columns=["condition"])
    target = df["condition"]

    logger.info('Split data...')
    x_train, x_test, y_train, y_test = train_test_split(
        x_input, target, test_size=test_split_size, random_state=random_state)
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    cv = StratifiedKFold(n_splits=config_model.fold_splits)
    model = BaselineModel(penalty=config_model.penalty, cv=cv,
                          max_iter=config_model.max_iter, tol=config_model.tol)

    logger.info('Training model...')
    model.fit(x_train, y_train)
    y_predict_train = model.predict(x_train)
    logger.info(f'Model f1 score train {f1_score(y_train, y_predict_train)}')
    logger.info(f'Model f1 score test {f1_score(y_test, model.predict(x_test))}')
    log_metric("f1", f1_score(y_test, model.predict(x_test)))

    logger.info('Save model...')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    mlflow.sklearn.log_model(sk_model=model, artifact_path='model',
                             registered_model_name='baseline')

    logger.info('Model trained')
