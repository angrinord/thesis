import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from tensorboard.backend.event_processing import event_accumulator

# DIRECTORY = 'xgboost'
# DIRECTORY = 'mystuff/collected_runs/fixed_deepset/20'
DIRECTORY = 'runs'
name = "stats.csv"
regimes_colors = {
    'entropy': 'darkgreen',
    'random': 'red',
    'margin': 'yellow',
    'confidence': 'cyan',
    'uniform': 'magenta',
    'BADGE': 'blue',
    'surrogate': 'orange',
    'surrogate_loss': 'saddlebrown',
    'surrogate_hat': 'midnightblue',
    'surrogate_heuristics': 'slategray',
    'surrogate_heuristics_hat': 'black',
    'entropy_balanced': 'darkgreen',
    'random_balanced': 'red',
    'margin_balanced': 'yellow',
    'confidence_balanced': 'cyan',
    'uniform_balanced': 'magenta',
    'surrogate_heuristics_loss': 'pink'
}


regimes = list(regimes_colors.keys())
AUC_range = (0, 500)
AUC_index = np.arange(AUC_range[0], AUC_range[1] + 1, 50)


def detrended_variance(series, window_size=2):
    moving_avg = series.rolling(window=window_size, min_periods=1).mean()
    detrended_data = series - moving_avg
    return np.var(detrended_data.diff().dropna())


def main():
    regime_mean = {}
    regime_mean_auc = {}
    regime_mean_inter = {}
    regime_mean_intra = {}
    for regime in regimes:
        pattern = re.compile(r'^({})\d+$'.format(re.escape(regime)))
        regime_dirs = [os.path.join(DIRECTORY, filename) for filename in os.listdir(DIRECTORY) if pattern.match(filename)]
        regime_acc = []
        regime_auc = []
        for regime_dir in regime_dirs:
            event_file_path = os.path.join(regime_dir, os.listdir(regime_dir)[0])
            event_file = event_accumulator.EventAccumulator(event_file_path, size_guidance={'tensors': 0})
            event_file.Reload()
            tags = event_file.Tags()['scalars']

            # Initialize an empty DataFrame
            data = pd.DataFrame(columns=['step'] + tags)

            # Extract data for each tag and populate the DataFrame
            for tag in tags:
                df = pd.DataFrame(event_file.Scalars(tag))
                df = df.rename(columns={'value': tag}).drop(columns=['wall_time'])
                data = data.merge(df, on='step', how='outer')
            data.set_index('step', inplace=True)
            new_index = np.arange(data.index.min(), data.index.max() + 1, 1)
            ACC = data['accuracy_y']
            interpolated_acc = ACC.reindex(new_index).interpolate()
            AUC = pd.Series([simps(interpolated_acc[AUC_index[i]:AUC_index[i+1]], x=new_index[AUC_index[i]:AUC_index[i+1]]) for i in range(len(AUC_index)-1)], index=[f"AUC:{AUC_index[i]}-{AUC_index[i+1]}" for i in range(len(AUC_index)-1)]) / 50
            AUC._set_value(f"AUC:{AUC_index[0]}-{AUC_index[-1]}", simps(interpolated_acc[AUC_index[0]:-1], x=new_index[AUC_index[0]:-1])/AUC_range[-1])
            regime_acc.append(interpolated_acc)
            regime_auc.append(AUC)
        if regime_dirs:
            variance = pd.concat(regime_acc, axis=1).agg(np.var, 1)
            variances = [variance[AUC_index[i]:AUC_index[i+1]].agg(np.mean, 0) for i in range(len(AUC_index)-1)]

            regime_mean_intra[regime] = pd.Series([detrended_variance(ACC.loc[AUC_index[i]:AUC_index[i+1]], 2) for i in range(len(AUC_index)-1)], index=[f"INTRA-VAR(detrended):{AUC_index[i]}-{AUC_index[i+1]}" for i in range(len(AUC_index)-1)])
            regime_mean_inter[regime] = pd.Series(variances, index=[f"INTER-VAR:{AUC_index[i]}-{AUC_index[i+1]}" for i in range(len(AUC_index)-1)])
            regime_mean[regime] = pd.concat(regime_acc, axis=1).agg(np.mean, 1)
            regime_mean_auc[regime] = pd.concat(regime_auc, axis=1).agg(np.mean, 1)
    regime_mean_auc = pd.DataFrame(regime_mean_auc)
    regime_mean_inter = pd.DataFrame(regime_mean_inter)
    regime_mean_intra = pd.DataFrame(regime_mean_intra)
    regime_stats = pd.concat([regime_mean_auc, regime_mean_inter, regime_mean_intra]).T
    regime_stats.to_csv(f"{DIRECTORY}/{name}", sep='\t')


if __name__ == '__main__':
    main()
