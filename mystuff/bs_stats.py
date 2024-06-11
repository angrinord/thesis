import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from tensorboard.backend.event_processing import event_accumulator

DIRECTORY = 'mystuff/collected_runs/fixed_deepset'
name = "bs_slopes.csv"
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
    'surrogate_heuristics_loss': 'pink'
}


regimes = list(regimes_colors.keys())
AUC_range = (0, 550)
AUC_index = np.arange(AUC_range[0], AUC_range[1] + 1, 50)
batch_sizes = [5, 10, 15, 20]


def main():
    all_means = {}
    for size in batch_sizes:
        directory = DIRECTORY + "/" + str(size)
        regime_mean = {}
        regime_mean_auc = {}
        for regime in regimes:
            pattern = re.compile(r'^({})_\d+$'.format(re.escape(regime)))
            regime_dirs = [os.path.join(directory, filename) for filename in os.listdir(directory) if pattern.match(filename)]
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
                AUC._set_value(f"AUC:{AUC_index[0]}-{AUC_index[2]}", simps(interpolated_acc[AUC_index[0]:AUC_index[2]], x=new_index[AUC_index[0]:AUC_index[2]])/AUC_index[2])
                regime_acc.append(interpolated_acc)
                regime_auc.append(AUC)
            if regime_dirs:
                regime_mean[regime] = pd.concat(regime_acc, axis=1).agg(np.mean, 1)
                regime_mean_auc[regime] = pd.concat(regime_auc, axis=1).agg(np.mean, 1)
        all_means[size] = pd.DataFrame(regime_mean_auc)
    grouped = pd.concat(all_means.values())
    means_y = grouped.groupby(grouped.index).mean()
    means_x = np.mean(list(all_means.keys()))
    numerator = 0
    denominator = 0
    for size, table in all_means.items():
        table.to_csv(f"{DIRECTORY}/{size}.csv", sep='\t', float_format='%.2f')
        diff = size - means_x
        numerator += (diff) * (table-means_y)
        denominator += np.square(diff)
    m = numerator/denominator
    m.to_csv(f"{DIRECTORY}/{name}", sep='\t', float_format="%.2g")


if __name__ == '__main__':
    main()
