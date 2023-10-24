import sys
sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
from mystuff import deepset_train
import argparse
import torch
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from keras import Sequential, backend
from keras.layers import Input, Dense
from keras.models import Model, clone_model
from keras.callbacks import EarlyStopping
from keras.utils import io_utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

directory = 'mystuff/data/deepset_toy'  # pretrained surrogate data
batch_count = 2000
batch_size = 15
set_encoder_file = deepset_train.set_encoder_file
sigma = 1e-10
VAL_SEED = 0  # Seed for getting the same validation data every time
test_ratio = 0.1  # How much of dataset should be set aside for validation
PATIENCE = 20  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
EPOCHS = 10000  # Set very high so that early stopping always happens
DEFAULT_BUDGET = 500
DEFAULT_POOL_SIZE = 50
embedding_dim = deepset_train.embedding_dim


def split_model(model, X=None, split=8):  # NOTE: Passing X is only necessary for testing.

    # Encoder is half of set encoder that is done instance-wise
    encoder_layers = model.layers[:split]
    encoder_model = Sequential(encoder_layers)
    encoder_model.build(input_shape=model.input_shape)
    encoder_model.set_weights(model.get_weights()[:split])

    # Aggregator is half of set encoder that is done batch-wise
    # TODO generalize to arbitrary architecture.  Right now this breaks if arch of set encoder changes.
    agg_in = Input(shape=encoder_model.output_shape[1:])
    agg_x = backend.mean(agg_in, axis=1)
    agg_x = Dense(embedding_dim)(agg_x)
    aggregator_model = Model(agg_in, agg_x)
    aggregator_model.set_weights(model.get_weights()[split:])

    # # testing stuff
    # preds = encoder_model(X)
    # preds = aggregator_model(preds)

    return encoder_model, aggregator_model


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


def entropy_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    return np.argmax(np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=1), axis=1))


def margin_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    sorted_predictions = np.sort(predictions, axis=2)
    return np.argmin(np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1))


def confidence_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    return np.argmin(np.mean(np.max(predictions, axis=2), axis=1))


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


regimes = {"confidence": confidence_selector, "random": random_selector, "entropy": entropy_selector, "margin": margin_selector, "uniform": uniform_selector}


def pretrain(data, labels, metafeatures, set_encoder, encoder, aggregator, idx, budget=DEFAULT_BUDGET, pool_size=DEFAULT_POOL_SIZE):
    surrogate_X = []
    surrogate_y = []
    surrogate_y_hat = []
    torch.manual_seed(VAL_SEED)
    tf.random.set_seed(VAL_SEED)
    np.random.seed(seed=VAL_SEED)
    X, test_X, y, test_y, mfs, _ = train_test_split(data, labels, metafeatures, test_size=test_ratio)

    torch.manual_seed(VAL_SEED)
    np.random.seed(idx)
    tf.random.set_seed(idx)
    X, val_X, y, val_y, mfs, _ = train_test_split(X, y, mfs, test_size=test_ratio)

    # Initial Training Pool
    X, y, mfs = shuffle(X, y, mfs, random_state=idx)
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
    primary_model.fit(pool_X, pool_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)])
    pool_weights = primary_model.get_weights()

    for regime_name, selector in regimes.items():
        print(regime_name)
        primary_model.set_weights(pool_weights)  # Every model starts trained on same initial training pool
        data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
        data_generator.add(pool_X, pool_y, pool_mfs)
        initial_loss = None
        writer = SummaryWriter("runs/deepset_toy/" f"{regime_name}" f"{idx}")
        labeled_X = pool_X
        labeled_y = pool_y
        test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=0)
        writer.add_scalar('loss_change', test_loss, data_generator.used)
        writer.add_scalar('accuracy', test_accuracy, data_generator.used)
        try:  # Iterators are supposed to throw StopIteration exception when they reach the end
            while True:  # This goes until budget is exhausted
                data_generator.n = selector(data_generator.X, primary_model)  # Sets the new batch
                x, label = next(data_generator)  # get the batch
                batch_metafeatures = data_generator.mfs[data_generator.n]  # Metafeature vector for the current batch
                labeled_X = np.vstack((labeled_X, x))
                labeled_y = np.concatenate((labeled_y, label))

                one_step_model.set_weights(primary_model.get_weights())
                one_step_model.train_on_batch(x, label)  # single gradient update on batch
                val_loss_hat, accuracy_hat = one_step_model.evaluate(val_X, val_y, verbose=0)
                test_loss_hat, test_accuracy_hat = one_step_model.evaluate(test_X, test_y, verbose=0)

                # K.clear_session()
                primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
                val_loss, accuracy = primary_model.evaluate(val_X, val_y)
                test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=0)

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
                writer.add_scalar('loss_change', test_loss, data_generator.used)
                writer.add_scalar('accuracy', test_accuracy, data_generator.used)
                writer.add_scalar('loss_hat_change', test_loss_hat, data_generator.used)
                writer.add_scalar('hat_accuracy', test_accuracy_hat, data_generator.used)
        except StopIteration:
            with open(f'{directory}/{regime_name}{idx}.pkl', 'wb') as f:
                pickle.dump((surrogate_X, surrogate_y, surrogate_y_hat), f)


def main():
    parser = argparse.ArgumentParser("cluster_deepset")
    parser.add_argument("-i", type=int)
    args = parser.parse_args()
    i = args.i

    data = load_digits()
    torch.manual_seed(VAL_SEED)
    tf.random.set_seed(VAL_SEED)
    np.random.seed(seed=VAL_SEED)

    data.data = data.data / 16.0

    # Set Encoder Stuff
    with open(set_encoder_file, 'rb') as input:
        deep_we = pickle.load(input)
        set_encoder = deepset_train.get_deepset_model(data.data.shape[-1])
        set_encoder.set_weights(deep_we)
    encoder, aggregator = split_model(set_encoder)
    instance_mfs = np.squeeze(encoder.predict(np.expand_dims(data.data, 1)), 1)

    pretrain(data.data, data.target, instance_mfs, set_encoder, encoder, aggregator, i)


if __name__ == '__main__':
    main()
