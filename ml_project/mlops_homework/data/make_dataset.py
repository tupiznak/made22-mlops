import logging

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import OneHotEncoder
from mlops_homework.data import DATA_PATH

CAT_FEATURES_ONE_HOT = ['sex', 'cp', 'restecg', 'thal']
CAT_FEATURES_LABEL = ['fbs', 'exang', 'slope', 'ca']


def main():
    logger = logging.getLogger('process data')

    logger.info('Begin preprocessing...')

    df = pd.read_csv(DATA_PATH.joinpath('raw/heart_cleveland_upload.csv'))
    x_input = df.drop(columns=['condition'])
    targets = df['condition']

    categorical_features = CAT_FEATURES_ONE_HOT
    encoder = OneHotEncoder()
    encoder.fit(x_input[categorical_features])
    x_transform = pd.concat(
        (
            x_input.drop(categorical_features, axis=1),
            pd.DataFrame(
                data=encoder.transform(x_input[categorical_features]).toarray(),
                columns=encoder.get_feature_names_out(),
            ),
            targets
        ),
        axis=1,
    )
    x_transform.to_csv(DATA_PATH.joinpath('processed/heart_cleveland_upload.csv'))

    logger.info('Preprocess finished')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()
