from faker import Faker
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from mlops_homework.features.build_features import DataTransformer


def test_transform(input_data):
    fake = Faker()
    transform = DataTransformer()

    x_input = input_data[0]
    ages = [fake.random_int(min=10, max=90) for _ in range(x_input.shape[0])]
    x_input.iloc[:, 0] = ages

    transform.fit(x_input, ['sex'])
    x = x_input.to_numpy()

    size = 20
    start_idx = 3
    test_data = x[start_idx:start_idx + size]
    need_data = pd.concat((x_input.drop(['sex'], axis=1),
                           pd.DataFrame(data=x_input['sex'] == 0),
                           pd.DataFrame(data=x_input['sex'] == 1)
                           ), axis=1).to_numpy()
    assert np.all(transform.transform_categorical(test_data) == need_data[start_idx:start_idx + size])
    need_data = scale(need_data)
    assert np.all(np.isclose(transform.transform(test_data), need_data[start_idx:start_idx + size]))
