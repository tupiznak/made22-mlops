import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

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
        self.columns = [col for col in x_data.columns
                        if col not in categorical_features] + list(self.encoder.get_feature_names_out())
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
