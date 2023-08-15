import os
import pickle
import re

from scipy import stats

evaluation_file = "evaluations.pkl"


def main():
    regimes = {}
    if os.path.isfile(evaluation_file):
        with open(evaluation_file, 'rb') as input:
            test = pickle.load(input)
            for key, value in test.items():
                regime = re.split(r'^([^_0-9]+)', key)[1]
                if regime in regimes:
                    regimes[regime].append(value)
                else:
                    regimes[regime] = [value]
        statistics = {}
        print("AUC Statistics:")
        print("{:<10} {:<10} {:<10}".format("Regime", "Mean", "Variance"))
        print("-" * 30)
        for regime in regimes:
            statistics[regime] = stats.describe(regimes[regime])
            mean = f"{statistics[regime].mean:.2f}"
            variance = f"{statistics[regime].variance:.2f}"
            print(f"{regime:<10} {mean:>10} {variance:>10}")


if __name__ == '__main__':
    main()
