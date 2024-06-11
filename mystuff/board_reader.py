import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from tensorboard.backend.event_processing import event_accumulator

# DIRECTORY = 'runs'
# DIRECTORY = 'xgboost'
DIRECTORY = 'mystuff/collected_runs/fixed_deepset'
bs_hypothesis = False
primary = 5
regimes_colors = {
    'random': 'red',
    'entropy': 'darkgreen',
    'margin': 'yellow',
    'confidence': 'darkturquoise',
    'uniform': 'magenta',
    'BADGE': 'blue',
    'surrogate': 'orange',
    'surrogate_loss': 'saddlebrown',
    'surrogate_hat': 'midnightblue',
    'surrogate_heuristics': 'slategray',
    'surrogate_heuristics_hat': 'black',
    'surrogate_heuristics_loss': 'pink',
    'random_balanced': 'pink',
    'entropy_balanced': 'lightgreen',
    'margin_balanced': 'yellow',
    'confidence_balanced': 'steelblue',
    'uniform_balanced': 'magenta'
}

size_styles = {
    5: 'solid',
    10: 'dashed',
    15: 'dotted',
    20: 'dashdot'
}

regimes = list(regimes_colors.keys())


def collect_acc(regime_dirs):
    regime_acc = []
    for regime_dir in regime_dirs:
        event_file_path = os.path.join(regime_dir, os.listdir(regime_dir)[0])
        event_file = event_accumulator.EventAccumulator(event_file_path, size_guidance={'tensors': 0})
        event_file.Reload()
        tags = event_file.Tags()['scalars']

        data = pd.DataFrame(columns=['step'] + tags)

        for tag in tags:
            df = pd.DataFrame(event_file.Scalars(tag))
            df = df.rename(columns={'value': tag}).drop(columns=['wall_time'])
            data = data.merge(df, on='step', how='outer')
        data.set_index('step', inplace=True)
        new_index = np.arange(data.index.min(), data.index.max() + 1, 1)
        regime_acc.append(data['accuracy_y'].reindex(new_index).interpolate())
    return pd.concat(regime_acc, axis=1).agg(np.mean, 1)


def toy_graphs(run_dir, save=False, font_size=30, batch_size=5, auc_ranges=[(0, 550), (0, 100)], acc_ranges=[(0.8, 1.0), (0.0, 1.0)]):
    assert len(auc_ranges) == len(acc_ranges)
    plt.rc('legend', fontsize=font_size)
    plt.rc('legend', title_fontsize=font_size)
    directory = run_dir + f"/{batch_size}"
    graphs = {
        "heur": regimes[:6],
        "surr": regimes[6:12],
        "best": [regimes[0]] + [regimes[2]] + [regimes[5]] + [regimes[7]] + [regimes[11]]
             }
    for graph in graphs:
        subregimes = graphs[graph]
        the_means = {}
        for regime in subregimes:
            pattern = re.compile(r'^({})_\d+$'.format(re.escape(regime)))  # Toy
            regime_dirs = [os.path.join(directory, filename) for filename in os.listdir(directory) if pattern.match(filename)]
            if regime_dirs:
                the_means[regime] = collect_acc(regime_dirs)

        for i in range(0, len(auc_ranges)):
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_dpi(100)
            fig.set_size_inches(19.20, 10.80, forward=True)
            fig.supxlabel(f'Labeled Pool Size(bs={batch_size})')
            fig.supylabel('Accuracy')
            AUC_range = auc_ranges[i]
            ACC_range = acc_ranges[i]
            name = f"{graph}_{AUC_range[0]}:{AUC_range[1]}"
            for regime in subregimes:
                axes.plot(the_means[regime].index, the_means[regime].values, label=regime, color=regimes_colors[regime], linestyle='solid', linewidth=4)
            axes.set_title('Mean Accuracy against Test(10 runs)')
            axes.set_ylim(ACC_range)
            axes.set_xlim(AUC_range)
            [axes.hlines(y=i / 100, xmin=AUC_range[0], xmax=AUC_range[1], colors='black', alpha=0.3, linewidth=0.2, linestyle='dotted') for i in range(0, 100, 10)]
            [axes.vlines(x=i, ymin=ACC_range[0], ymax=ACC_range[1], colors='black', alpha=0.3, linewidth=0.2, linestyle='dotted') for i in range(AUC_range[0], AUC_range[1], 50)]
            axes.margins(0)
            plt.legend(loc='lower right', title='Querying Strategy')
            plt.tight_layout()
            if save:
                plt.savefig(f"{run_dir}/graphs/{name}.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()


def bs_graphs(run_dir, save=False, font_size=30, auc_ranges=[(0, 550), (0, 100)], acc_ranges=[(0.8, 1.0), (0.0, 1.0)]):
    global size_styles
    assert len(auc_ranges) == len(acc_ranges)
    plt.rc('legend', fontsize=font_size)
    plt.rc('legend', title_fontsize=font_size)
    subregimes = regimes[:6] + [regimes[7]] + [regimes[11]]
    for regime in subregimes:
        the_means = {}
        for batch_size in size_styles:
            directory = run_dir + f"/{batch_size}"
            pattern = re.compile(r'^({})_\d+$'.format(re.escape(regime)))  # Toy
            regime_dirs = [os.path.join(directory, filename) for filename in os.listdir(directory) if pattern.match(filename)]
            if regime_dirs:
                the_means[batch_size] = collect_acc(regime_dirs)
        for i in range(0, len(auc_ranges)):
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_dpi(100)
            fig.set_size_inches(19.20, 10.80, forward=True)
            fig.supxlabel(f'Labeled Pool Size(qs={regime})')
            fig.supylabel('Accuracy')
            AUC_range = auc_ranges[i]
            ACC_range = acc_ranges[i]
            name = f"{regime}_{AUC_range[0]}:{AUC_range[1]}"
            for batch_size in size_styles:
                axes.plot(the_means[batch_size].index, the_means[batch_size].values, label=batch_size, color=regimes_colors[regime], linestyle=size_styles[batch_size], linewidth=4)
            axes.set_title('Mean Accuracy against Test(10 runs)')
            axes.set_ylim(ACC_range)
            axes.set_xlim(AUC_range)
            [axes.hlines(y=i / 100, xmin=AUC_range[0], xmax=AUC_range[1], colors='black', alpha=0.3, linewidth=0.2, linestyle='dotted') for i in range(0, 100, 10)]
            [axes.vlines(x=i, ymin=ACC_range[0], ymax=ACC_range[1], colors='black', alpha=0.3, linewidth=0.2, linestyle='dotted') for i in range(AUC_range[0], AUC_range[1], 50)]
            axes.margins(0)
            plt.legend(loc='lower right', title="batch size")
            plt.tight_layout()
            if save:
                plt.savefig(f"{run_dir}/graphs/{name}.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()


def main_graphs(run_dir, save=False, font_size=30, batch_size=5, auc_ranges=[(0, 500), (0, 100)], acc_ranges=[(0.7, 0.9), (0.0, 0.9)]):
    assert len(auc_ranges) == len(acc_ranges)
    plt.rc('legend', fontsize=font_size)
    plt.rc('legend', title_fontsize=font_size)
    graphs = {
        "heur": regimes[:6],
        "surr": regimes[6:12],
        "balanced": [regimes[0]] + [regimes[3]] + [regimes[12]] + [regimes[15]],
        "best": [regimes[12]] + [regimes[15]] + [regimes[6]] + [regimes[9]]
             }
    for graph in graphs:
        subregimes = graphs[graph]
        the_means = {}
        for regime in subregimes:
            pattern = re.compile(r'^({})\d+$'.format(re.escape(regime)))
            regime_dirs = [os.path.join(run_dir, filename) for filename in os.listdir(run_dir) if pattern.match(filename)]
            if regime_dirs:
                the_means[regime] = collect_acc(regime_dirs)

        for i in range(0, len(auc_ranges)):
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_dpi(100)
            fig.set_size_inches(19.20, 10.80, forward=True)
            fig.supxlabel(f'Labeled Pool Size(bs={batch_size})')
            fig.supylabel('Accuracy')
            AUC_range = auc_ranges[i]
            ACC_range = acc_ranges[i]
            name = f"{graph}_{AUC_range[0]}:{AUC_range[1]}"
            for regime in subregimes:
                axes.plot(the_means[regime].index, the_means[regime].values, label=regime, color=regimes_colors[regime], linestyle='solid', linewidth=4)
            axes.set_title('Mean Accuracy against Test(20 runs)')
            axes.set_ylim(ACC_range)
            axes.set_xlim(AUC_range)
            [axes.hlines(y=i / 100, xmin=AUC_range[0], xmax=AUC_range[1], colors='black', alpha=0.3, linewidth=0.2, linestyle='dotted') for i in range(0, 100, 10)]
            [axes.vlines(x=i, ymin=ACC_range[0], ymax=ACC_range[1], colors='black', alpha=0.3, linewidth=0.2, linestyle='dotted') for i in range(AUC_range[0], AUC_range[1], 50)]
            axes.margins(0)
            plt.legend(loc='lower right', title='Querying Strategy')
            plt.tight_layout()
            if save:
                plt.savefig(f"{run_dir}/graphs/{name}.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()


def main():
    main_graphs('runs', True)
    # bs_graphs(DIRECTORY, True)
    # toy_graphs(DIRECTORY, True)


if __name__ == '__main__':
    main()
