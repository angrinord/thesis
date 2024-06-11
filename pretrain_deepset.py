import sys


sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
from mystuff import deepset_train
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse
import os
import pickle
import random
import numpy as np
import tensorflow as tf
import torch
from keras import Sequential, Input, Model, backend
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import clone_model
from keras.utils import io_utils
from torch.utils.tensorboard import SummaryWriter

# Constants and Variables Booker
sigma = 1e-10  # This is so entropy doesn't break
batch_count = 2000  # Number of random batches BatchGenerator should create
PATIENCE = 20  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
VAL_SEED = 0  # Seed for getting the same validation data every time
EPOCHS = 10000  # Set very high so that early stopping always happens
test_ratio = 0.1  # How much of dataset should be set aside for validation
DEFAULT_POOL_SIZE = 0
SECONDARY_POOL_SIZE = 50
DEFAULT_BUDGET = 550 - DEFAULT_POOL_SIZE
directory = 'mystuff/data/fixed_deepset'  # Directory that stores preprocessed MNIST and pretrained surrogate data
set_encoder_file = deepset_train.set_encoder_file
embedding_dim = deepset_train.embedding_dim


def split_model(model, split=5):  # NOTE: Passing X is only necessary for testing.

    # Encoder is half of set encoder that is done instance-wise
    encoder_layers = model.layers[:split]
    encoder_model = Sequential(encoder_layers)
    encoder_model.build(input_shape=model.input_shape)
    encoder_model.set_weights(model.get_weights()[:split+3])

    # Aggregator is half of set encoder that is done batch-wise
    # TODO generalize to arbitrary architecture.  Right now this breaks if arch of set encoder changes.
    agg_in = Input(shape=encoder_model.output_shape[1:])
    agg_x = backend.sum(agg_in, axis=1)
    agg_x = Dense(embedding_dim)(agg_x)
    aggregator_model = Model(agg_in, agg_x)
    aggregator_model.set_weights(model.get_weights()[split+3:-2])

    return encoder_model, aggregator_model


def entropy_selector(batches, model):
    predictions = tf.reshape(model(tf.reshape(tf.stack(batches), (-1, model.input.shape[-1]))), (len(batches), batches[0].shape[0], -1))
    return int(tf.argmax(tf.reduce_mean(tf.reduce_sum(-predictions * tf.math.log(tf.maximum(predictions, sigma)), axis=2), axis=1), axis=0))


def margin_selector(batches, model):
    predictions = tf.reshape(model(tf.reshape(tf.stack(batches), (-1, model.input.shape[-1]))), (len(batches), batches[0].shape[0], -1))
    sorted_predictions = tf.sort(predictions, axis=2)
    return int(tf.argmin(tf.reduce_mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1), axis=0))


def confidence_selector(batches, model):
    predictions = tf.reshape(model(tf.reshape(tf.stack(batches), (-1, model.input.shape[-1]))), (len(batches), batches[0].shape[0], -1))
    return int(tf.argmin(tf.reduce_mean(tf.reduce_max(predictions, axis=2), axis=1), axis=0))


def random_selector(batches, model):
    return random.randrange(len(batches))


def uniform_selector(batches, model):
    selectors = {0: entropy_selector, 1: margin_selector, 2: confidence_selector, 3: random_selector}
    return selectors[np.random.choice(4)](batches, model)


regimes = {"random": random_selector, "entropy": entropy_selector, "margin": margin_selector, "confidence": confidence_selector, "uniform": uniform_selector}


# This class handles the generation of random batches and the storage of data relevant to current and previously generated batches.
class BatchGenerator:
    def __init__(self, data, labels, budget, count, size, metafeatures, aggregator=None):
        self.data = data  # The data from which to generate random batches
        self.labels = labels  # The labels of the data from which random batches are generated
        self.count = count  # The number of random batches to generate when generate_batches is called
        self.size = size  # The size of the batches to be generated.  Should be a multiple of the cardinality of the labels (10 in the case of MNIST)
        self.metafeatures = metafeatures
        self.aggregator = aggregator  # The set encoder used to generate metafeature representations of batches
        self.finished = False  # Flag for when the budget is exhausted or nearly exhausted
        self.n = 0  # The index of the batch to return (is set externally)
        self.budget = budget  # The total number of samples that can be labeled
        self.used = 0  # The number of samples that have been labeled so far
        self.X, self.y, self.mfs, self.indices = self.generate_batches()  # The initial batches and relevant data
        self.selected_X = []
        self.selected_y = []
        self.selected_mfs = []
        self.classes = list(range(self.labels.shape[-1]))  # The class labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        next_X = self.X[self.n]
        next_y = self.y[self.n]
        return next_X, next_y

    # Called when a batch (or the initial training pool) has is to be added to the selected data
    def add(self, data, labels, metafeatures, budgeted=False):
        self.selected_X.append(data)
        self.selected_y.append(labels)
        self.selected_mfs.append(metafeatures)
        if not budgeted:
            self.used += len(data)
            self.budget += len(data)

    # Called when new batches are to be generated.  The selected batch is added to selected, removed from the unlabeled pool, and various other things.
    def regenerate(self):
        if self.finished:
            return
        io_utils.print_msg(
            "Remaining Budget: "
            f"{self.budget - self.used:.2f}")
        indices = self.indices[self.n]
        self.used += len(np.unique(indices))

        self.add(self.data[indices], self.labels[indices], self.metafeatures[indices], budgeted=True)

        self.data = np.delete(self.data, indices, axis=0)
        self.labels = np.delete(self.labels, indices, axis=0)
        self.metafeatures = np.delete(self.metafeatures, indices, axis=0)

        self.n = 0
        if self.budget == self.used:
            self.finished = True
        elif self.budget - self.size < self.used:
            self.X, self.y, self.mfs, self.indices = self.generate_batches()
        else:
            self.X, self.y, self.mfs, self.indices = self.generate_batches()

    # Generates 'count' balanced batches of size 'size' from 'data' as well as generating their metafeatures using 'aggregator'
    def generate_batches(self):
        batches = []
        batches_y = []
        batches_indices = []  # Stores indices of batch elements
        batches_mf = []
        size = self.size
        if self.budget - self.size < self.used:
            size = self.budget - self.used
        for _ in range(self.count):
            indices = np.random.choice(len(self.data), size=size, replace=True)
            batch_data = tf.gather(self.data, indices)
            batch_y = tf.gather(self.labels, indices)
            batch_mf = self.aggregator(self.metafeatures[indices].reshape(1, size, self.metafeatures.shape[-1]))
            batches.append(batch_data)
            batches_y.append(batch_y)
            batches_mf.append(tf.squeeze(batch_mf, axis=0))
            batches_indices.append(tf.convert_to_tensor(indices, dtype=tf.int64))
        return batches, batches_y, batches_mf, tf.reshape(tf.stack(batches_indices), (self.count, size))


# This function is for the generation of training data for the surrogate model.  The data is composed of... TODO: More comments
def pretrain(data, labels, metafeatures, aggregator, idx, regime_name, batch_size, budget=DEFAULT_BUDGET, pool_size=DEFAULT_POOL_SIZE):
    X, test_X, y, test_y, mfs, _ = train_test_split(data, labels, metafeatures, test_size=test_ratio, random_state=VAL_SEED)

    torch.manual_seed(idx)
    np.random.seed(idx)
    tf.random.set_seed(idx)
    X, val_X, y, val_y, mfs, _ = train_test_split(X, y, mfs, test_size=test_ratio, random_state=idx)

    # Initial Training Pool
    X, y, mfs = shuffle(X, y, mfs, random_state=idx)
    pool_X, unlabeled_X = X[:pool_size], X[pool_size:]
    pool_y, unlabeled_y = y[:pool_size], y[pool_size:]
    pool_mfs, unlabeled_mfs = mfs[:pool_size], mfs[pool_size:]

    file_path = f"{directory}/initial_models{pool_size}/initial{idx}.pkl"
    lock_path = f"{directory}/initial_models{pool_size}/lock{idx}.lock"
    while os.path.exists(lock_path):
        time.sleep(1)
    primary_in = Input(shape=data.shape[-1])
    primary_hidden = Dense(32, activation='relu')(primary_in)
    primary_hidden = Dense(32, activation='relu')(primary_hidden)
    primary_out = Dense(10, activation='softmax')(primary_hidden)
    primary_model = Model(primary_in, primary_out)
    primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    one_step_model = clone_model(primary_model)
    one_step_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if not os.path.isfile(file_path):
        with open(lock_path, 'w') as lock_file:
            if pool_size > 0:
                primary_model.fit(pool_X, pool_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)])
            with open(file_path, 'wb') as output:
                pickle.dump(primary_model.get_weights(), output)
        os.remove(lock_path)
    else:
        with open(file_path, 'rb') as model_file:
            pool_weights = pickle.load(model_file)
            primary_model.set_weights(pool_weights)

    print(f"{regime_name}{idx}")
    surrogate_X = []
    surrogate_y = []
    surrogate_y_hat = []
    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    initial_loss = None
    # writer = SummaryWriter(f"mystuff/collected_runs/fixed_deepset/{batch_size}/{regime_name}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    # test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=0)
    # writer.add_scalar('loss_change', test_loss, data_generator.used)
    # writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:
        while True:
            data_generator.n = regimes[regime_name](data_generator.X, primary_model)  # Sets the new batch
            x, label = next(data_generator)  # get the batch
            batch_metafeatures = data_generator.mfs[data_generator.n]  # Metafeature vector for the current batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            one_step_model.set_weights(primary_model.get_weights())
            one_step_model.train_on_batch(x, label)  # single gradient update on batch
            val_loss_hat, accuracy_hat = one_step_model.evaluate(val_X, val_y, verbose=0)
            # test_loss_hat, test_accuracy_hat = one_step_model.evaluate(test_X, test_y, verbose=0)

            predictions = primary_model.predict(x, verbose=0)  # TODO: Rerun this bitch... :`(
            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            val_loss, accuracy = primary_model.evaluate(val_X, val_y)
            # test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=0)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances

            # TODO this should have been moved before primary_model.fit too...
            pool_metafeatures = np.concatenate(data_generator.selected_mfs)
            pool_metafeatures = pool_metafeatures.reshape(1, pool_metafeatures.shape[0], pool_metafeatures.shape[-1])
            pool_metafeatures = np.array(aggregator(pool_metafeatures)).flatten()

            entropy = np.array([np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=1))])
            sorted_predictions = np.sort(predictions, axis=1)
            margin = np.array([np.mean(sorted_predictions[:, -1] - sorted_predictions[:, -2])])
            confidence = np.array([np.mean(np.max(predictions, axis=1))])
            used = np.array([data_generator.used])
            histogram = np.mean([np.histogram(predictions[:, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[1])], axis=0)
            surrogate_in = tf.concat((batch_metafeatures, pool_metafeatures, entropy, margin, confidence, used, histogram), axis=0)

            if initial_loss is None:
                initial_loss = val_loss
            else:
                surrogate_X.append(surrogate_in)
                surrogate_y.append(initial_loss - val_loss)
                surrogate_y_hat.append(initial_loss - val_loss_hat)
                initial_loss = val_loss
            # writer.add_scalar('loss_change', test_loss, data_generator.used)
            # writer.add_scalar('accuracy', test_accuracy, data_generator.used)
            # writer.add_scalar('loss_hat_change', test_loss_hat, data_generator.used)
            # writer.add_scalar('hat_accuracy', test_accuracy_hat, data_generator.used)
    except StopIteration:
        with open(f'{directory}/{regime_name}_{batch_size}_{idx}.pkl', 'wb') as f:
            pickle.dump((surrogate_X, surrogate_y, surrogate_y_hat), f)
        with open(f'{directory}/clipped/{regime_name}_{batch_size}_{idx}.pkl', 'wb') as f:
            pickle.dump((surrogate_X[SECONDARY_POOL_SIZE//batch_size:], surrogate_y[SECONDARY_POOL_SIZE//batch_size:], surrogate_y_hat[SECONDARY_POOL_SIZE//batch_size:]), f)


def main():
    parser = argparse.ArgumentParser("pretrain_deepset")
    parser.add_argument("-i", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--regime")
    args = parser.parse_args()
    i = args.i
    regime = args.regime
    batch_size = args.batch_size

    data = load_digits()
    data.data = data.data / 16.0

    with open(set_encoder_file, 'rb') as input:
        deep_we = pickle.load(input)
        set_encoder = deepset_train.get_deepset_model(data.data.shape[-1])
        set_encoder.set_weights(deep_we)
    encoder, aggregator = split_model(set_encoder)
    instance_mfs = np.squeeze(encoder.predict(np.expand_dims(data.data, 1)), 1)
    pretrain(data.data, data.target, instance_mfs, aggregator, i, regime, batch_size)


if __name__ == '__main__':
    main()
