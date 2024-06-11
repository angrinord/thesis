import sys

from scipy import stats
from scipy.integrate import simps
from sklearn.metrics import pairwise_distances

sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
from mystuff import deepset_train, deepset_surrogate_train
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
from keras.utils import io_utils
from torch.utils.tensorboard import SummaryWriter
from keras.losses import SparseCategoricalCrossentropy

# Constants and Variables Booker
sigma = 1e-10  # This is so entropy doesn't break
batch_count = 1000  # Number of random batches BatchGenerator should create
PATIENCE = 20  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
VAL_SEED = 0  # Seed for getting the same validation data every time
EPOCHS = 10000  # Set very high so that early stopping always happens
test_ratio = 0.1  # How much of dataset should be set aside for validation
DEFAULT_POOL_SIZE = 0
DEFAULT_BUDGET = 550-DEFAULT_POOL_SIZE
directory = f'mystuff/data/fixed_deepset'  # Directory that stores preprocessed MNIST and pretrained surrogate data
run_directory = f'mystuff/collected_runs/fixed_deepset'
set_encoder_file = deepset_train.set_encoder_file
embedding_dim = deepset_train.embedding_dim


def kmeanspp(gradients, K, idx):
    ind = int(tf.argmax(tf.linalg.norm(gradients, axis=1)))
    mu = [gradients[ind]]
    indsAll = [ind]
    centInds = [0.] * len(gradients)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(gradients, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(gradients, [mu[-1]]).ravel().astype(float)
            for i in range(len(gradients)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist), seed=idx)
        ind = customDist.rvs(size=1)[0]
        mu.append(gradients[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


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


def get_instance(record_bytes):
    feature_description = {
        "data": tf.io.FixedLenFeature([], dtype=tf.string),
        "loss": tf.io.FixedLenFeature([], dtype=tf.string),
        "hat": tf.io.FixedLenFeature([], dtype=tf.string)
    }
    example = tf.io.parse_single_example(record_bytes, feature_description)

    data = tf.io.parse_tensor(example['data'], tf.float32)
    loss = tf.io.parse_tensor(example['loss'], tf.float32)
    hat = tf.io.parse_tensor(example['hat'], tf.float32)

    return data, loss, hat


def set_generator_n(data_generator, primary_model, surrogate_model, mean, std):
    pool_metafeatures = np.concatenate(data_generator.selected_mfs)
    pool_metafeatures = pool_metafeatures.reshape(1, pool_metafeatures.shape[0], pool_metafeatures.shape[-1])
    pool_metafeatures = np.tile(np.array(data_generator.aggregator(pool_metafeatures)).flatten(), (len(data_generator.indices), 1))
    predictions = tf.reshape(primary_model(tf.reshape(tf.stack(data_generator.X), (-1, primary_model.input.shape[-1]))), (len(data_generator.X), data_generator.X[0].shape[0], -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (len(data_generator.indices), 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])
    surrogate_in = np.concatenate((data_generator.mfs, pool_metafeatures, entropy, margin, confidence, used, histogram), axis=1)
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in)[:, 0])


def set_generator_n1d(data_generator, primary_model, surrogate_model, mean, std):
    pool_metafeatures = np.concatenate(data_generator.selected_mfs)
    pool_metafeatures = pool_metafeatures.reshape(1, pool_metafeatures.shape[0], pool_metafeatures.shape[-1])
    pool_metafeatures = np.tile(np.array(data_generator.aggregator(pool_metafeatures)).flatten(), (len(data_generator.indices), 1))
    predictions = tf.reshape(primary_model(tf.reshape(tf.stack(data_generator.X), (-1, primary_model.input.shape[-1]))), (len(data_generator.X), data_generator.X[0].shape[0], -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (len(data_generator.indices), 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])
    surrogate_in = np.concatenate((data_generator.mfs, pool_metafeatures, entropy, margin, confidence, used, histogram), axis=1)
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in))


def set_generator_n_raw(data_generator, primary_model, surrogate_model, mean, std):
    predictions = tf.reshape(primary_model(tf.reshape(tf.stack(data_generator.X), (-1, primary_model.input.shape[-1]))), (len(data_generator.X), data_generator.X[0].shape[0], -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (len(data_generator.indices), 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])
    surrogate_in = np.concatenate((entropy, margin, confidence, used, histogram), axis=1)
    mean = mean[embedding_dim * 2:]
    std = std[embedding_dim * 2:]
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in)[:, 0])


def set_generator_n1d_raw(data_generator, primary_model, surrogate_model, mean, std):
    predictions = tf.reshape(primary_model(tf.reshape(tf.stack(data_generator.X), (-1, primary_model.input.shape[-1]))), (len(data_generator.X), data_generator.X[0].shape[0], -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (len(data_generator.indices), 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])
    surrogate_in = np.concatenate((entropy, margin, confidence, used, histogram), axis=1)
    mean = mean[embedding_dim * 2:]
    std = std[embedding_dim * 2:]
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in))


def badge_selector(data_generator, primary_model, idx):
    gradients = []
    loss_function = SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    for datum in tf.convert_to_tensor(data_generator.data):
        with tf.GradientTape() as tape:
            tape.watch(datum)
            model_out = primary_model(tf.expand_dims(datum, 0))
            predictions = tf.argmax(model_out, axis=1)
            loss = loss_function(predictions, model_out)
            last_layer_params = primary_model.layers[-1].trainable_variables
            gradients.append(tape.gradient(loss, last_layer_params))
    gradients = [tf.norm(tf.concat([gradient[0], tf.expand_dims(gradient[1], axis=0)], axis=0), axis=1) for gradient in gradients]
    closest = kmeanspp(gradients, data_generator.X[0].shape[0], idx)
    data_generator.X[0] = tf.gather(data_generator.data, closest)
    data_generator.y[0] = tf.gather(data_generator.labels, closest)
    data_generator.mfs[0] = data_generator.aggregator(data_generator.metafeatures[closest].reshape(1, data_generator.X[0].shape[0], data_generator.metafeatures.shape[-1]))
    data_generator.indices[0] = tf.convert_to_tensor(closest, dtype=tf.int64)
    data_generator.n = 0
    return data_generator


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
        return batches, batches_y, batches_mf, batches_indices


def badge(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, val_X, val_y):
    regime = "BADGE"
    print(f"{regime} {idx}")
    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator = badge_selector(data_generator, primary_model, idx)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def surrogate_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, surrogate, val_X, val_y):
    regime = "surrogate"
    print(f"{regime} {idx}")
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=1)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = set_generator_n(data_generator, primary_model, surrogate, mean, std)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def loss_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, surrogate_loss, val_X, val_y):
    regime = "surrogate_loss"
    print(f"{regime} {idx}")
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=1)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = set_generator_n1d(data_generator, primary_model, surrogate_loss, mean, std)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def hat_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, surrogate_hat, val_X, val_y):
    regime = "surrogate_hat"
    print(f"{regime} {idx}")
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=1)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = set_generator_n1d(data_generator, primary_model, surrogate_hat, mean, std)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def heuristics_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, surrogate_heuristics, val_X, val_y):
    regime = "surrogate_heuristics"
    print(f"{regime} {idx}")
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=1)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = set_generator_n_raw(data_generator, primary_model, surrogate_heuristics, mean, std)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def heuristics_hat(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, surrogate_heuristics_hat, val_X, val_y):
    regime = "surrogate_heuristics_hat"
    print(f"{regime} {idx}")
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=1)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = set_generator_n1d_raw(data_generator, primary_model, surrogate_heuristics_hat, mean, std)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def heuristics_loss(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, surrogate_heuristics_hat, val_X, val_y):
    regime = "surrogate_heuristics_loss"
    print(f"{regime} {idx}")
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=1)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = set_generator_n1d_raw(data_generator, primary_model, surrogate_heuristics_hat, mean, std)
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instance

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


def raw_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, regime, val_X, val_y):
    print(f"{regime} {idx}")
    writer = SummaryWriter(f"{run_directory}/{batch_size}/{regime}_{idx}")
    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, unlabeled_mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y, pool_mfs)
    labeled_X = pool_X
    labeled_y = pool_y

    test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)
    values = [test_accuracy]
    indices = [data_generator.used]
    auc = simps(values, x=indices)
    aucs = [auc]

    writer.add_scalar('auc', auc, data_generator.used)
    writer.add_scalar('loss_change', test_loss, data_generator.used)
    writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            data_generator.n = regimes[regime](tf.convert_to_tensor(data_generator.X), primary_model)  # Sets the new batch
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            test_loss, test_accuracy = primary_model.evaluate(test_X, test_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances

            values.append(test_accuracy)
            indices.append(data_generator.used)
            auc = simps(values, x=indices)
            aucs.append(auc)
            writer.add_scalar('auc', auc, data_generator.used)
            writer.add_scalar('loss_change', test_loss, data_generator.used)
            writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    except StopIteration:
        pass


# This function is for the generation of training data for the surrogate model.  The data is composed of... TODO: More comments
def evaluate(data, labels, metafeatures, aggregator, idx, regime, batch_size, budget=DEFAULT_BUDGET, pool_size=DEFAULT_POOL_SIZE):
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

    if regime == "badge":
        badge(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, val_X, val_y)
    elif regime == "surrogate":
        with open(deepset_surrogate_train.surrogate_file, 'rb') as file:
            model = pickle.load(file)
            surrogate_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, model, val_X, val_y)
    elif regime == "surrogate_loss":
        with open(deepset_surrogate_train.surrogate_file_loss, 'rb') as file:
            model = pickle.load(file)
            loss_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, model, val_X, val_y)
    elif regime == "surrogate_hat":
        with open(deepset_surrogate_train.surrogate_file_hat, 'rb') as file:
            model = pickle.load(file)
            hat_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, model, val_X, val_y)
    elif regime == "surrogate_heuristics":
        with open(deepset_surrogate_train.surrogate_file_heuristics, 'rb') as file:
            model = pickle.load(file)
            heuristics_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, model, val_X, val_y)
    elif regime == "surrogate_heuristics_hat":
        with open(deepset_surrogate_train.surrogate_file_heuristics_hat, 'rb') as file:
            model = pickle.load(file)
            heuristics_hat(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, model, val_X, val_y)
    elif regime == "surrogate_heuristics_loss":
        with open(deepset_surrogate_train.surrogate_file_heuristics_loss, 'rb') as file:
            model = pickle.load(file)
            heuristics_loss(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, model, val_X, val_y)
    else:
        raw_part(unlabeled_X, unlabeled_y, budget, batch_size, unlabeled_mfs, aggregator, pool_X, pool_y, pool_mfs, idx, primary_model, test_X, test_y, regime, val_X, val_y)


def main():
    parser = argparse.ArgumentParser("deepset_evaluate")
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
    evaluate(data.data, data.target, instance_mfs, aggregator, i, regime, batch_size)


if __name__ == '__main__':
    main()
