# /src/utils_step_1.py

import pandas as pd


def load_data(PATH, target_col):
    data = pd.read_excel(PATH, index_col=0)
    cols_drop = ['Default (y)', 'Pred_default (y_hat)', 'PD', 'Group']
    X = data.drop(cols_drop, axis=1)
    y = data[[target_col]]

    return X, y
