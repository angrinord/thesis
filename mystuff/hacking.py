import os
import pickle
import random
from logging import warning
from typing import Iterator
import torchvision.datasets as dset

import torch
# import xgboost as xgb
import numpy as np
# import keras.backend as K
import tensorflow as tf
# from keras import Sequential, backend
# from keras.layers import Input, Dense, Lambda, TimeDistributed
# from keras.models import Model, clone_model
# from keras.optimizers import Adam
# from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping
# from keras.utils import io_utils, Sequence
from scipy import stats
from scipy.integrate import simps
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

from MetaD2A_nas_bench_201 import process_dataset

# batch_count = 2000
# batch_size = 10
# set_enc_range = (2, 32)
# primary_epochs = 10
# set_encoder_file = "deepset_with mean.pkl"
# surrogate_data_file = "surrogate_data_fixed.pkl"
# evaluation_file = "evaluations.pkl"
# predictor_checkpoint_dir = 'results/predictor/model'
# sigma = 1e-10
# encodings_dir = 'data'
directory = encodings_dir


NUM_SAMPLES = 20
NUM_CLASSES = 10
OUTER_BATCH_SIZE = 5
NUM_RANDOM_BATCHES = 17

def reconstruction_rapid_nas(intra_setpool, inter_setpool):

    for file in os.listdir(directory):
        if file.endswith('mnistbylabel.pt'):
            with open(directory + '/' + file, 'rb') as f:
                data = torch.load(f)
    X = []
    for _ in range(OUTER_BATCH_SIZE):
        batch = []
        classes = list(range(NUM_CLASSES))
        for cls in classes:
          cx = data[cls][0]
          ridx = torch.randperm(len(cx))
          batch.append(cx[ridx[:NUM_SAMPLES]])
        batch = torch.stack(batch, dim=0)
        X.append(batch)
    X = torch.stack(X, dim=0)

    proto_batch = []
    for x in X:
        cls_protos = intra_setpool(x).squeeze(1)
        proto_batch.append(inter_setpool(cls_protos.unsqueeze(0)).squeeze())
    proto_batch = torch.stack(proto_batch, dim=0)
    print("success")

def sort_by_class(x, y)->dict[torch.Tensor]:
    # return a dict of tensors
    pass

def foo_real(intra_setpool, inter_setpool):
    labeled_x = torch.rand(size=(100, 512))
    labeled_y = torch.randint(size=(100, 1), low=0, high=10)
    unlabeled_x = torch.rand(size=(1000, 512))
    def primary_model(x)->torch.Tensor:
        # forward pass of your primary model
        pass

    # train the primary model on your labeled data
    pseudo_labels = primary_model(unlabeled_x)
    data_by_class = sort_by_class(unlabeled_x, pseudo_labels)
    random_batches = []
    for _ in range(NUM_RANDOM_BATCHES):
        batch = []
        classes = list(range(NUM_CLASSES))
        for cls in classes:
            cx = data_by_class[cls]
            ids = torch.randperm(unlabeled_x.size(0))[:NUM_SAMPLES]
            batch.append(cx[ids[:NUM_SAMPLES]])
        batch = torch.stack(batch, dim=0)
        random_batches.append(batch)
    random_batches = torch.stack(random_batches, dim=0) # 17 x 10 x {1,2,20} x 512
    # pass through the set pools
    proto_batch = []
    for x in random_batches:
        cls_protos = intra_setpool(x).squeeze(1)
        proto_batch.append(inter_setpool(cls_protos.unsqueeze(0)).squeeze())
    proto_batch_unlabeled = torch.stack(proto_batch, dim=0) # 17 x 56
    # pass through a batch selectors
    selected_batch = None # 1 x 56

    # encode the labeled set with rapid nas
    proto_batch_labeled = None # 1 x 56

    other_stuff = None
    surrogate_data = torch.cat([selected_batch, proto_batch_labeled, other_stuff])
    # pass it through the surrogate model
    pass


def main():
    with open('my_stuff/intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('my_stuff/inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)

    # reconstruction_rapid_nas(intra_setpool, inter_setpool)
    foo_real(intra_setpool, inter_setpool)



if __name__ == '__main__':
    main()
