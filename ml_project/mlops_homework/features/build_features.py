import logging
import pickle

import dvc.api
import hydra
import pandas as pd

from mlops_homework.conf.config import Config
from mlops_homework.data import DATA_PATH, MODEL_PATH
from .data_transformer import DataTransformer

CAT_FEATURES_ONE_HOT = ['sex', 'cp', 'restecg', 'thal']
CAT_FEATURES_LABEL = ['fbs', 'exang', 'slope', 'ca']


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    # FIXME need help
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
