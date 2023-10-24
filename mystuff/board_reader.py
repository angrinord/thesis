import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

directory = 'runs'
regimes_colors = {
    'random_balanced': 'pink',
    'entropy_balanced': 'springgreen',
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
    'surrogate_heuristics_hat': 'black'
}

regimes = list(regimes_colors.keys())
# regimes = regimes[:4]  # Compare balanced and unbalanced classes
# regimes = regimes[:2] + regimes[-5:]  # Compare all surrogates
# regimes = regimes[:2] + [regimes[-6]] + [regimes[-1]]  # Compare best surrogate to BADGE and heuristics


def main():
    fig, axes = plt.subplots(nrows=3, ncols=1)
    i = 0
    for regime in regimes:
        regime_dirs = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.startswith(regime)]
        regime_acc = []
        regime_auc = []
        regime_loss = []
        regime_mean = {}
        regime_mean_auc = {}
        regime_mean_loss = {}
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
            regime_auc.append(data['auc_y'].reindex(new_index).interpolate())
            regime_acc.append(data['accuracy_y'].reindex(new_index).interpolate())
            regime_loss.append(data['loss_change_y'].reindex(new_index).interpolate())
        regime_mean[regime] = pd.concat(regime_acc, axis=1).agg(np.mean, 1)
        regime_mean_auc[regime] = pd.concat(regime_auc, axis=1).agg(np.mean, 1)
        regime_mean_loss[regime] = pd.concat(regime_loss, axis=1).agg(np.mean, 1)
        axes[0].plot(regime_mean[regime].index, regime_mean[regime].values, label=regime, color=regimes_colors[regime])
        axes[1].plot(regime_mean_auc[regime].index, regime_mean_auc[regime].values, label=regime, color=regimes_colors[regime])
        axes[2].plot(regime_mean_loss[regime].index, regime_mean_loss[regime].values, label=regime, color=regimes_colors[regime])
        i += 1
    fig.supxlabel('Labeled Pool Size')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('AUC')
    axes[2].set_ylabel('Loss')
    axes[0].set_title('Primary Model Accuracy')
    axes[1].set_title('Primary Model AUC')
    axes[2].set_title('Primary Model Loss')
    axes[0].legend(title='Acquisition Functions')
    plt.margins(0)
    plt.show()


if __name__ == '__main__':
    main()
