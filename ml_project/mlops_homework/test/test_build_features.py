import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from mlops_homework.features.build_features import DataTransformer


def test_transform(input_data):
    transform = DataTransformer()
    x_input = input_data[0]
    transform.fit(x_input, ['sex'])
    x = x_input.to_numpy()
    need_data = pd.concat((x_input.drop(['sex'], axis=1),
                           pd.DataFrame(data=x_input['sex'] == 0),
                           pd.DataFrame(data=x_input['sex'] == 1)
                           ), axis=1).to_numpy()
    assert np.all(transform.transform_categorical(x[3:5]) == need_data[3:5])
    need_data = scale(need_data)
    assert np.all(np.isclose(transform.transform(x[3:5]), need_data[3:5]))
