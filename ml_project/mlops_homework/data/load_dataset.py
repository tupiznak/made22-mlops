import logging

import hydra
from dotenv import find_dotenv, load_dotenv

from mlops_homework.conf.config import PROJECT_PATH, Config


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    logger = logging.getLogger('download dataset')

    logger.info('Try kaggle auth...')
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except OSError:
        logger.error('KAGGLE_KEY and KAGGLE_USERNAME not found in environment. Exit.')
        return

    logger.info('Downloading dataset...')
    api.dataset_download_files('cherngs/heart-disease-cleveland-uci',
                               path=PROJECT_PATH + cfg.relative_path_to_data_raw,
                               unzip=True)
    logger.info('Dataset downloaded')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()
