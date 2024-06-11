from critdd import Diagram, Diagrams
import pandas as pd
import os
import re
import numpy as np
import pandas as pd
from scipy.integrate import simps
from tensorboard.backend.event_processing import event_accumulator

# DIRECTORY = 'xgboost'
# DIRECTORY = 'mystuff/collected_runs/fixed_deepset/20'
from main import draw_cd_diagram

DIRECTORY = 'runs'
name = "critdd.tex"

regimes_colors = {
    # 'entropy': 'darkgreen',
    # 'random': 'red',
    # 'margin': 'yellow',
    # 'confidence': 'cyan',
    # 'uniform': 'magenta',
    # 'BADGE': 'blue',
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
alpha = 0.05


def main():
    performances_dict = {}
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
            AUC = pd.Series([simps(interpolated_acc[AUC_index[i]:AUC_index[i + 1]], x=new_index[AUC_index[i]:AUC_index[i + 1]]) for i in range(len(AUC_index) - 1)], index=[f"AUC:{AUC_index[i]}-{AUC_index[i + 1]}" for i in range(len(AUC_index) - 1)]) / 50
            AUC._set_value(f"AUC:{AUC_index[0]}-{AUC_index[-1]}", simps(interpolated_acc[AUC_index[0]:-1], x=new_index[AUC_index[0]:-1]) / AUC_range[-1])
            regime_acc.append(interpolated_acc)
            regime_auc.append(AUC)
        if regime_dirs:
            df = pd.DataFrame(regime_auc)
            for column in df:
                if column in performances_dict.keys():
                    performances_dict[column][regime] = df[column]
                else:
                    performances_dict[column] = pd.DataFrame(df[column].rename(regime))

    total = performances_dict.pop(f"AUC:{AUC_range[0]}-{AUC_range[1]}")

    # df_reset = total.reset_index()
    # melted_df = pd.melt(df_reset, id_vars=['index'], var_name='classifier_name', value_name='accuracy')
    # melted_df.columns = ['dataset_name', 'classifier_name', 'accuracy']
    # draw_cd_diagram(df_perf=melted_df)

    diagram = Diagram(
        total.to_numpy(),
        treatment_names=total.columns,
        maximize_outcome=True
    )
    # export the diagram to a file
    diagram.to_file(
        f"{DIRECTORY}/graphs/critdd/AUC:{AUC_range[0]}-{AUC_range[1]}(a={alpha}).tex",
        alpha=alpha,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": f"AUC:{AUC_range[0]}-{AUC_range[1]}(α={alpha})"},
    )
    diagrams = Diagrams(np.stack(performances_dict.values()),
                    treatment_names=total.columns,
                    diagram_names=performances_dict.keys(),
                    maximize_outcome=True)
    diagrams.to_file(
        f"{DIRECTORY}/graphs/critdd/AUCs(a={alpha}).tex",
        alpha=alpha,
        adjustment="holm",
        reverse_x=True,
        axis_options={"title": f"AUCs(a={alpha})"},
    )


if __name__ == '__main__':
    main()
