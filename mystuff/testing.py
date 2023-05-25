import os
import pickle

import torch

encodings_dir = 'data'
generator_checkpoint_dir = 'results/generator/model'
predictor_checkpoint_dir = 'results/predictor/model'
datapath = encodings_dir
directory = encodings_dir
dim_input = 512


def main():
    data = {}
    for file in os.listdir(predictor_checkpoint_dir):
        if file.endswith('.pt'):
            with open(predictor_checkpoint_dir + '/' + file, 'rb') as f:
                data = torch.load(f)

    with open('../../mystuff/intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('../../mystuff/inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)

    for file in os.listdir(directory):
        if file.endswith('mnistbylabel.pt'):
            with open(directory + '/' + file, 'rb') as f:
                data = torch.load(f)
    test_batch = []
    for clss, values in data.items():
        test_batch.append(values[0][0:2])
    test_batch = torch.stack(test_batch)
    out = intra_setpool(test_batch)
    out = inter_setpool(out)
    print('test')


if __name__ == '__main__':
    main()