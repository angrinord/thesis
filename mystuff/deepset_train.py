import sys

sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
import os
import torch
import pickle
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import Sequential, backend
from keras.layers import Input, Dense, TimeDistributed, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import Sequence
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

batch_count = 2000
set_enc_range = (2, 16)
set_enc_labelling = 'sum'
set_encoder_file = f"deepset_test_{set_enc_labelling}_{set_enc_range[0]}_{set_enc_range[-1]}.pkl"
VAL_SEED = 0  # Seed for getting the same validation data every time
test_ratio = 0.1  # How much of dataset should be set aside for validation
PATIENCE = 20
embedding_dim = 16


def get_deepset_model(data_dim):
    adam = Adam(lr=1e-3, epsilon=1e-3)

    # Encoder
    # TimeDistributed should leave the latent features uncorrelated across instances.
    input_img = Input(shape=(None, data_dim,))
    x = TimeDistributed(Dense(128, activation='tanh'))(input_img)
    x = TimeDistributed(Dense(64, activation='tanh'))(x)
    x = TimeDistributed(Dense(32, activation='tanh'))(x)
    x = TimeDistributed(Dense(embedding_dim, activation='tanh'))(x)

    # Aggregator
    x = backend.sum(x, axis=1)
    x = Dense(embedding_dim)(x)
    x = Dense(1)(x)  # Throw this away

    model = Model(input_img, x)
    model.compile(optimizer=adam, loss="mae")

    return model


def generate_batches(data, labels, count, size_range, metafeatures=None, aggregator=None):
    batches = []
    batches_y = []
    if metafeatures is None:
        for _ in range(count):
            size = np.random.random_integers(size_range[0], size_range[-1])
            indices = np.random.choice(len(data), size=size, replace=True)  # Could change up size to be random to induce resilience to different batch sizes
            batch_data = data[indices]
            batch_y = labels[indices]
            batches.append(batch_data)
            batches_y.append(batch_y)
        return np.array(batches), np.array(batches_y)
    else:
        batches_mf = []
        batches_indices = []
        for _ in range(count):
            size = np.random.random_integers(size_range[0], size_range[-1])
            indices = np.random.choice(len(data), size=size, replace=True)
            batch_data = data[indices]
            batch_y = labels[indices]
            batch_mf = aggregator(metafeatures[indices].reshape(1, size, metafeatures.shape[-1]))
            batches.append(batch_data)
            batches_y.append(batch_y)
            batches_mf.append(batch_mf)
            batches_indices.append(indices)
        return np.array(batches), np.array(batches_y), np.array(batches_mf).reshape(count, metafeatures.shape[-1]), np.array(batches_indices)


def get_encoder_labels(X_train, y_train):
    assert set_enc_labelling == 'mean' or set_enc_labelling == 'sum'
    if set_enc_labelling == 'mean':
        do_avg = np.vectorize(np.mean)
        return do_avg(y_train)
    do_sum = np.vectorize(np.sum)
    return do_sum(y_train)


def main():
    if not os.path.isfile(set_encoder_file):
        data = load_digits()
        torch.manual_seed(VAL_SEED)
        tf.random.set_seed(VAL_SEED)
        np.random.seed(seed=VAL_SEED)

        data.data = data.data / 16.0

        X, val_X, y, val_y = train_test_split(data.data, data.target, test_size=test_ratio)
        X, y = generate_batches(X, y, batch_count, set_enc_range)
        val_X, val_y = generate_batches(val_X, val_y, batch_count, set_enc_range)
        encoder_y = get_encoder_labels(X, y)
        encoder_val_y = get_encoder_labels(val_X, val_y)

        K.clear_session()
        set_encoder = get_deepset_model(data.data.shape[-1])

        class set_enc_sequence(Sequence):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx].reshape(1, self.X[idx].shape[0], self.X[idx].shape[-1]), np.array([self.y[idx]])

        # train
        set_generator = set_enc_sequence(X, encoder_y)
        val_generator = set_enc_sequence(val_X, encoder_val_y)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=1e-6)
        set_encoder.fit(set_generator, epochs=300, shuffle=True, callbacks=[reduce_lr, EarlyStopping(patience=PATIENCE*2)], validation_data=val_generator)
        deep_we = set_encoder.get_weights()
        # save weights
        with open(set_encoder_file, 'wb') as output:
            pickle.dump(deep_we, output)
    return 0


if __name__ == '__main__':
    main()
