import hydra
import pandas as pd
from pandas_profiling import ProfileReport

from mlops_homework.conf.config import PROJECT_PATH, Config


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    df = pd.read_csv(PROJECT_PATH + cfg.relative_path_to_data_raw_csv)
    profile = ProfileReport(df, title="EDA Report")
    profile.to_file(PROJECT_PATH + cfg.relative_path_to_reports_eda)


if __name__ == '__main__':
    main()
