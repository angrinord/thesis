import os
import pickle
import random
from logging import warning
from typing import Iterator
import torchvision.datasets as dset

import torch
import xgboost as xgb
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import Sequential, backend
from keras.layers import Input, Dense, Lambda, TimeDistributed
from keras.models import Model, clone_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping
from keras.utils import io_utils, Sequence
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

from MetaD2A.MetaD2A_nas_bench_201 import process_dataset

batch_count = 2000
batch_size = 10
set_enc_range = (2, 32)
primary_epochs = 10
set_encoder_file = "deepset_with mean.pkl"
surrogate_data_file = "surrogate_data_fixed.pkl"
evaluation_file = "evaluations.pkl"
predictor_checkpoint_dir = 'results/predictor/model'
sigma = 1e-10
encodings_dir = 'data'
directory = encodings_dir


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


def batch_pool(pool_X, pool_y, pool_mfs, size, aggregator):
    assert len(pool_X) == len(pool_X) == len(pool_mfs)
    length = len(pool_X)
    p = np.random.permutation(length)
    pool_X = np.reshape(pool_X[p], (length // size, size, pool_X.shape[-1]))
    pool_y = np.reshape(pool_y[p], (length // size, size))
    pool_mfs = aggregator(np.reshape(pool_mfs[p], (length // size, size, pool_mfs.shape[-1])))
    return pool_X, pool_y, pool_mfs


def get_encoder_labels(X_train, y_train):
    # mean of the digits
    do_avg = np.vectorize(np.mean)
    return do_avg(y_train)
    # do_sum = np.vectorize(np.sum)
    # return do_sum(y_train)


def entropy_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    return np.argmax(np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=1), axis=1))


def margin_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    sorted_predictions = np.sort(predictions, axis=2)
    return np.argmin(np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1))


def confidence_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    return np.argmin(np.mean(np.max(predictions, axis=1), axis=1))


def random_selector(batches, model):
    return random.randrange(len(batches))


def uniform_selector(batches, model):
    selectors = {0: entropy_selector, 1: margin_selector, 2: confidence_selector, 3: random_selector}
    return selectors[np.random.choice(4)](batches, model)


class BatchGenerator:
    def __init__(self, data, labels, budget, count, size, metafeatures=None, aggregator=None):
        self.data = data
        self.labels = labels
        self.count = count
        self.size = size
        self.metafeatures = metafeatures
        self.aggregator = aggregator
        self.finished = False
        self.n = 0
        self.budget = budget
        self.used = 0
        self.X, self.y, self.mfs, self.indices = generate_batches(self.data, self.labels, self.count, (self.size, self.size), self.metafeatures, self.aggregator)
        self.selected_X = []
        self.selected_y = []
        self.selected_mfs = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        next_X = self.X[self.n]
        next_y = self.y[self.n]
        return next_X, next_y

    def add(self, data, labels, metafeatures):
        self.selected_X.append(data)
        self.selected_y.append(labels)
        self.selected_mfs.append(metafeatures)
        self.budget += len(data)
        self.used += len(data)

    def regenerate(self):
        if self.finished:
            return
        io_utils.print_msg(
            "Remaining Budget: "
            f"{self.budget - self.used:.2f}")

        indices = self.indices[self.n]
        self.used += len(np.unique(indices))

        self.selected_X.append(self.data[indices])
        self.selected_y.append(self.labels[indices])
        self.selected_mfs.append(self.metafeatures[indices])

        self.data = np.delete(self.data, indices, axis=0)
        self.labels = np.delete(self.labels, indices, axis=0)
        self.metafeatures = np.delete(self.metafeatures, indices, axis=0)

        self.n = 0
        if self.budget == self.used:
            self.finished = True
        elif self.budget - self.size < self.used:
            self.X, self.y, self.mfs, self.indices = generate_batches(self.data, self.labels, self.count, (self.budget - self.used, self.budget - self.used), self.metafeatures, self.aggregator)
        else:
            self.X, self.y, self.mfs, self.indices = generate_batches(self.data, self.labels, self.count, (self.size, self.size), self.metafeatures, self.aggregator)


def pretrain(data, labels, metafeatures, set_encoder, encoder, aggregator, budget=1000, pool_size=100):
    regimes = [random_selector, entropy_selector, margin_selector, confidence_selector, uniform_selector]
    regime_iterations = 10
    surrogate_X = []
    surrogate_y = []
    surrogate_y_hat = []

    X, val_X, y, val_y, mfs, _ = train_test_split(data, labels, metafeatures, test_size=0.015)

    # Train model using given selection heuristic multiple times
    for i in range(regime_iterations):
        np.random.seed(i)
        tf.random.set_seed(i)
        # Initial Training Pool
        X, y, mfs = shuffle(X, y, mfs, random_state=i)
        pool_X, unlabeled_X = X[:pool_size], X[pool_size:]
        pool_y, unlabeled_y = y[:pool_size], y[pool_size:]
        pool_mfs, unlabeled_mfs = mfs[:pool_size], mfs[pool_size:]

        # Simple primary model
        primary_in = Input(shape=set_encoder.input_shape[-1])
        primary_hidden = Dense(32, activation='relu')(primary_in)
        primary_hidden = Dense(32, activation='relu')(primary_hidden)
        primary_out = Dense(10, activation='softmax')(primary_hidden)
        primary_model = Model(primary_in, primary_out)
        primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        one_step_model = clone_model(primary_model)
        one_step_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        primary_model.fit(pool_X, pool_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)])
        pool_weights = primary_model.get_weights()

        # Iterate over different batch selection heuristics
        for regime in regimes:
            print(regime.__name__)
            primary_model.set_weights(pool_weights)  # Every model starts trained on same initial training pool
            data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, 1000, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
            data_generator.add(pool_X, pool_y, pool_mfs)
            initial_loss = None
            writer = SummaryWriter("runs/" f"{regime.__name__}" f"{i + 1}")
            labeled_X = pool_X
            labeled_y = pool_y
            try:  # Iterators are supposed to throw StopIteration exception when they reach the end
                while True:  # This goes until budget is exhausted
                    data_generator.n = regime(data_generator.X, primary_model)  # Sets the new batch
                    x, label = next(data_generator)  # get the batch
                    batch_metafeatures = data_generator.mfs[data_generator.n]  # Metafeature vector for the current batch
                    labeled_X = np.vstack((labeled_X, x))
                    labeled_y = np.concatenate((labeled_y, label))

                    one_step_model.set_weights(primary_model.get_weights())
                    one_step_model.train_on_batch(x, label)  # single gradient update on batch
                    val_loss_hat, accuracy_hat = one_step_model.evaluate(val_X, val_y, verbose=0)

                    # K.clear_session()
                    primary_model.fit(labeled_X, labeled_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)], verbose=0)
                    val_loss, accuracy = primary_model.evaluate(val_X, val_y)

                    data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances

                    # Generate various features for surrogate model inputs
                    pool_metafeatures = np.concatenate(data_generator.selected_mfs)
                    pool_metafeatures = pool_metafeatures.reshape(1, pool_metafeatures.shape[0], pool_metafeatures.shape[-1])
                    pool_metafeatures = np.array(aggregator(pool_metafeatures)).flatten()
                    predictions = primary_model.predict(x, verbose=0)
                    entropy = np.array([np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=1))])
                    sorted_predictions = np.sort(predictions, axis=1)
                    margin = np.array([np.mean(sorted_predictions[:, -1] - sorted_predictions[:, -2])])
                    confidence = np.array([np.mean(np.max(predictions, axis=1))])
                    used = np.array([data_generator.used])
                    histogram = np.mean([np.histogram(predictions[:, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[1])], axis=0)
                    surrogate_in = np.concatenate((batch_metafeatures, pool_metafeatures, entropy, margin, confidence, used, histogram))

                    if initial_loss is None:
                        initial_loss = val_loss
                    else:
                        surrogate_X.append(surrogate_in)
                        surrogate_y.append(initial_loss - val_loss)
                        surrogate_y_hat.append(initial_loss - val_loss_hat)
                        initial_loss = val_loss
                    writer.add_scalar('loss_change', val_loss, data_generator.used)
                    writer.add_scalar('accuracy', accuracy, data_generator.used)
                    writer.add_scalar('loss_hat_change', val_loss_hat, data_generator.used)
                    writer.add_scalar('hat_accuracy', accuracy_hat, data_generator.used)
            except StopIteration:
                with open(surrogate_data_file, 'wb') as f:
                    pickle.dump((surrogate_X, surrogate_y, surrogate_y_hat), f)


def set_generator_n(data_generator, primary_model, surrogate_model, scaler):
    # TODO
    # Generate various features for surrogate model inputs
    pool_metafeatures = np.concatenate(data_generator.selected_mfs)
    pool_metafeatures = pool_metafeatures.reshape(1, pool_metafeatures.shape[0], pool_metafeatures.shape[-1])
    pool_metafeatures = np.tile(np.array(data_generator.aggregator(pool_metafeatures)).flatten(), (data_generator.indices.shape[0], 1))
    predictions = primary_model.predict(data_generator.X.reshape(-1, data_generator.X.shape[-1]), verbose=0).reshape(data_generator.X.shape[0], data_generator.X.shape[1], -1)
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (data_generator.indices.shape[0], 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])
    surrogate_in = xgb.DMatrix(scaler.transform(np.concatenate((data_generator.mfs, pool_metafeatures, entropy, margin, confidence, used, histogram), axis=1)))
    return np.argmax(surrogate_model.predict(surrogate_in)[:, 0])


def train_surrogate(data, labels, metafeatures, set_encoder, encoder, aggregator, surrogate_data, targets, targets_hat, budget=1000, pool_size=100):
    param = {
        'objective': 'reg:squarederror',  # Learning task and loss function
        'eval_metric': 'rmse',  # Metric for evaluation
        'n_estimators': 100,  # Number of boosting rounds (trees)
        'learning_rate': 0.1,  # Step size at each iteration
        'max_depth': 3,  # Maximum depth of a tree
        'min_child_weight': 1,  # Minimum sum of instance weight in a child node
        'subsample': 1.0,  # Fraction of samples used for training each tree
        'colsample_bytree': 1.0,  # Fraction of features used for training each tree
        'gamma': 0,  # Minimum loss reduction required for further partition
        'seed': 0  # Random seed for reproducibility
    }
    surrogate_X, surrogate_val_X, surrogate_y, surrogate_val_y = train_test_split(surrogate_data, np.column_stack((targets, targets_hat)), test_size=0.015)
    scaler = StandardScaler()
    surrogate_data = scaler.fit_transform(surrogate_X)
    surrogate_val_X = scaler.transform(surrogate_val_X)
    dtrain = xgb.DMatrix(surrogate_data, label=surrogate_y)
    evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
    num_rounds = 300
    surrogate_model = xgb.train(param, dtrain, num_rounds, evallist, early_stopping_rounds=20)

    regimes = [random_selector, entropy_selector, margin_selector, confidence_selector, uniform_selector]
    regime_iterations = 10

    X, val_X, y, val_y, mfs, _ = train_test_split(data, labels, metafeatures, test_size=0.015)

    ###########################

    evaluations = {}

    # Train model using given selection heuristic multiple times
    for i in range(regime_iterations):
        np.random.seed(i)
        tf.random.set_seed(i)
        # Initial Training Pool
        X, y, mfs = shuffle(X, y, mfs, random_state=i)
        pool_X, unlabeled_X = X[:pool_size], X[pool_size:]
        pool_y, unlabeled_y = y[:pool_size], y[pool_size:]
        pool_mfs, unlabeled_mfs = mfs[:pool_size], mfs[pool_size:]

        # Simple primary model
        primary_in = Input(shape=set_encoder.input_shape[-1])
        primary_hidden = Dense(32, activation='relu')(primary_in)
        primary_hidden = Dense(32, activation='relu')(primary_hidden)
        primary_out = Dense(10, activation='softmax')(primary_hidden)
        primary_model = Model(primary_in, primary_out)
        primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        one_step_model = clone_model(primary_model)
        one_step_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        primary_model.fit(pool_X, pool_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)])
        pool_weights = primary_model.get_weights()

        # Iterate over different batch selection heuristics
        for regime in regimes:
            primary_model.set_weights(pool_weights)  # Every model starts trained on same initial training pool
            data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, 1000, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
            data_generator.add(pool_X, pool_y, pool_mfs)
            writer = SummaryWriter("runs/new/" f"{regime.__name__}" f"{i + 1}")
            labeled_X = pool_X
            labeled_y = pool_y
            values = []
            indices = []
            aucs = []
            try:  # Iterators are supposed to throw StopIteration exception when they reach the end
                while True:  # This goes until budget is exhausted
                    data_generator.n = regime(data_generator.X, primary_model)  # Sets the new batch
                    x, label = next(data_generator)  # get the batch
                    batch_metafeatures = data_generator.mfs[data_generator.n]  # Metafeature vector for the current batch
                    labeled_X = np.vstack((labeled_X, x))
                    labeled_y = np.concatenate((labeled_y, label))

                    primary_model.fit(labeled_X, labeled_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)], verbose=0)
                    val_loss, accuracy = primary_model.evaluate(val_X, val_y)

                    data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances
                    values.append(accuracy)
                    indices.append(data_generator.used)
                    auc = simps(values, x=indices)
                    aucs.append(auc)
                    writer.add_scalar('auc', auc, data_generator.used)
                    writer.add_scalar('loss_change', val_loss, data_generator.used)
                    writer.add_scalar('accuracy', accuracy, data_generator.used)
            except StopIteration:
                with open(evaluation_file, 'wb') as f:
                    evaluations[f"{regime.__name__}" f"{i + 1}"] = auc
                    pickle.dump(evaluations, f)
        # Surrogate part
        primary_model.set_weights(pool_weights)  # Every model starts trained on same initial training pool
        data_generator = BatchGenerator(X, y, budget, 1000, batch_size, mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
        data_generator.add(pool_X, pool_y, pool_mfs)

        writer = SummaryWriter("runs/new/surrogate" f"{i + 1}")
        labeled_X = pool_X
        labeled_y = pool_y
        values = []
        indices = []
        aucs = []
        try:  # Iterators are supposed to throw StopIteration exception when they reach the end
            while True:  # This goes until budget is exhausted
                data_generator.n = set_generator_n(data_generator, primary_model, surrogate_model, scaler)  # Sets the new batch
                x, label = next(data_generator)  # get the batch
                labeled_X = np.vstack((labeled_X, x))
                labeled_y = np.concatenate((labeled_y, label))

                primary_model.fit(labeled_X, labeled_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)], verbose=0)
                val_loss, accuracy = primary_model.evaluate(val_X, val_y)

                data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances
                values.append(accuracy)
                indices.append(data_generator.used)
                auc = simps(values, x=indices)
                aucs.append(auc)
                writer.add_scalar('auc', auc, data_generator.used)
                writer.add_scalar('loss_change', val_loss, data_generator.used)
                writer.add_scalar('accuracy', accuracy, data_generator.used)
        except StopIteration:
            with open(evaluation_file, 'wb') as f:
                evaluations["surrogate" f"{i + 1}"] = auc
                pickle.dump(evaluations, f)


class MyTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = SummaryWriter()
        self.step = 1

    def on_test_end(self, logs=None):
        super().on_test_end(logs)
        if logs is not None and 'loss' in logs:
            self.writer.add_scalar('loss', logs['loss'], self.step)
            self.step += 1


def main():
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
        # test = values[0][0:2]
        # test = intra_setpool(test)
        test_batch.append(values[0][0:2])
    test_batch = torch.stack(test_batch)
    out = intra_setpool(test_batch)
    out = inter_setpool(out)
    print('test')

    # surrogate_X, surrogate_y, surrogate_y_hat = None, None, None
    # if not os.path.isfile(surrogate_data_file):
    #     pretrain(data.data, data.target, instance_mfs, set_encoder, encoder, aggregator)
    #     return 0
    # with open(surrogate_data_file, 'rb') as input:
    #     surrogate_X, surrogate_y, surrogate_y_hat = pickle.load(input)
    # train_surrogate(data.data, data.target, instance_mfs, set_encoder, encoder, aggregator, surrogate_X, surrogate_y, surrogate_y_hat)


if __name__ == '__main__':
    main()
