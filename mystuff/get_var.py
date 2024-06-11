import re
import sys

import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from matplotlib import pyplot as plt

directory = 'mystuff/data/fixed_deepset'
AUC_range = (0, 550)
AUC_index = np.arange(AUC_range[0], AUC_range[1] + 1, 50)


def detrended_variance(series, window_size=2):
    moving_avg = series.rolling(window=window_size, min_periods=1).mean()
    detrended_data = series - moving_avg
    return np.var(detrended_data.diff().dropna())


def main():
    hat_vars = []
    loss_vars = []
    correlation = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as in_file:
                data, loss, hat = pickle.load(in_file)
                correlation.append(np.corrcoef(loss, hat)[0, 1])
                index = [float(tensor[-11]) for tensor in data]
                loss_var = pd.Series([detrended_variance(pd.Series(loss, index=index).loc[AUC_index[i]:AUC_index[i + 1]]) for i in range(len(AUC_index) - 1)], index=[f"INTRA-VAR(detrended):{AUC_index[i]}-{AUC_index[i + 1]}" for i in range(len(AUC_index) - 1)])
                hat_var = pd.Series([detrended_variance(pd.Series(hat, index=index).loc[AUC_index[i]:AUC_index[i + 1]]) for i in range(len(AUC_index) - 1)], index=[f"INTRA-VAR(detrended):{AUC_index[i]}-{AUC_index[i + 1]}" for i in range(len(AUC_index) - 1)])
                loss_var._set_value(f"INTRA-VAR(detrended):{AUC_index[0]}-{AUC_index[-1]}", detrended_variance(pd.Series(loss, index=index).loc[AUC_index[0]:AUC_index[-1]]))
                hat_var._set_value(f"INTRA-VAR(detrended):{AUC_index[0]}-{AUC_index[-1]}", detrended_variance(pd.Series(hat, index=index).loc[AUC_index[0]:AUC_index[-1]]))
                loss_vars.append(loss_var)
                hat_vars.append(hat_var)
    loss_vars = pd.concat(loss_vars, axis=1).mean(axis=1)
    hat_vars = pd.concat(hat_vars, axis=1).mean(axis=1)
    correlation = np.mean(correlation)
    print(loss_vars)
    print(hat_vars)


if __name__ == '__main__':
    main()
