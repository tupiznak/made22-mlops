import logging
import pickle

import dvc.api
import hydra
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_homework.conf.config import Config
from mlops_homework.data import DATA_PATH, MODEL_PATH

CAT_FEATURES_ONE_HOT = ['sex', 'cp', 'restecg', 'thal']
CAT_FEATURES_LABEL = ['fbs', 'exang', 'slope', 'ca']


class DataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()
        self.categorical_features_idx: list[int] = []
        self.categorical_features: list[str] = []
        self.real_features_idx: list[int] = []
        self.columns: list[str] = []

    def fit_categorical(self, x_data: pd.DataFrame, categorical_features):
        self.encoder.fit(x_data[categorical_features])
        self.categorical_features_idx = [x_data.columns.get_loc(f) for f in categorical_features]
        self.real_features_idx = [f for f in range(x_data.shape[1])
                                  if f not in self.categorical_features_idx]
        self.categorical_features = categorical_features
        categorical_features = set(categorical_features)
        self.columns = [col for col in x_data.columns if col not in categorical_features] + \
                       list(self.encoder.get_feature_names_out())
        return self.encoder

    def fit_scaler(self, x_data: np.ndarray):
        self.scaler.fit(x_data)
        return self.scaler

    def fit(self, x_data: pd.DataFrame, categorical_features):
        self.fit_categorical(x_data, categorical_features)
        self.fit_scaler(self.transform_categorical(x_data.to_numpy()))
        return self

    def get_columns(self):
        return self.columns

    def transform_categorical(self, x_batch: np.ndarray):
        data = pd.DataFrame(data=x_batch[:, list(self.categorical_features_idx)],
                            columns=self.categorical_features)
        cat_data = self.encoder.transform(data).toarray()
        return np.concatenate((x_batch[:, list(self.real_features_idx)], cat_data), axis=1)

    def transform_scaler(self, x_batch: np.ndarray):
        return self.scaler.transform(x_batch)

    def transform(self, x_batch: np.ndarray):
        x_batch = self.transform_categorical(x_batch)
        x_batch = self.transform_scaler(x_batch)
        return x_batch


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    try:
        dvc_params = dvc.api.params_show()
        [setattr(cfg, k, v) for k, v in dvc_params.items()]
    except FileNotFoundError:
        pass
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger('process data')

    logger.info('Begin preprocessing...')

    df = pd.read_csv(DATA_PATH.joinpath('raw/heart_cleveland_upload.csv'))
    x_input = df.drop(columns=['condition'])
    targets = df['condition']

    if cfg.preprocessing.categorical_features == 'all':
        categorical_features = CAT_FEATURES_ONE_HOT + CAT_FEATURES_LABEL
    else:
        categorical_features = CAT_FEATURES_ONE_HOT
    encoder = DataTransformer()
    encoder.fit(x_data=x_input, categorical_features=categorical_features)

    logger.info('Save encoder...')
    with open(MODEL_PATH.joinpath('encoder_baseline.pkl'), 'wb') as file:
        pickle.dump(encoder, file)

    x_transform = pd.concat(
        (
            pd.DataFrame(
                data=encoder.transform(x_input.to_numpy()),
                columns=encoder.get_columns(),
            ),
            targets
        ),
        axis=1,
    )
    x_transform.to_csv(DATA_PATH.joinpath('processed/heart_cleveland_upload.csv'), index=False)

    logger.info('Preprocess finished')


if __name__ == '__main__':
    main()
