import logging

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle

from mlops_homework.data import DATA_PATH, MODEL_PATH


def main():
    logger = logging.getLogger('train model')

    logger.info('Prepare model...')

    df = pd.read_csv(DATA_PATH.joinpath('processed/heart_cleveland_upload.csv'))
    x_input = df.drop(columns=["condition"])
    target = df["condition"]

    logger.info('Split data...')
    x_train, x_test, y_train, y_test = train_test_split(x_input, target, test_size=0.85)

    cv = StratifiedKFold(n_splits=5)
    model = LogisticRegressionCV(penalty="l2", cv=cv, max_iter=10000, tol=0.01)

    logger.info('Training model...')
    model.fit(x_train, y_train)
    y_predict_train = model.predict(x_train)
    f1_score(y_train, y_predict_train)

    logger.info('Save model...')
    with open(MODEL_PATH.joinpath('baseline.pkl'), 'wb') as file:
        pickle.dump(model, file)

    logger.info('Model trained')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()