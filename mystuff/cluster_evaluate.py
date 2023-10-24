import sys
import time

sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
import fcntl
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from keras.losses import SparseCategoricalCrossentropy
from mystuff import deepset_surrogate_train
import argparse
from scipy.integrate import simps
import torch
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import io_utils
from torch.utils.tensorboard import SummaryWriter

directory = 'mystuff/data'  # pretrained surrogate data
batch_count = 1000
batch_size = 10
sigma = 1e-10
VAL_SEED = 0  # Seed for getting the same validation data every time
test_ratio = 0.1  # How much of dataset should be set aside for validation
PATIENCE = 20  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
EPOCHS = 10000  # Set very high so that early stopping always happens
DEFAULT_BUDGET = 500
DEFAULT_POOL_SIZE = 50


def badge_selector(data_generator, primary_model):
    gradients = []
    loss_function = SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    for datum in data_generator.data:
        with tf.GradientTape() as tape:
            tape.watch(datum)
            model_out = primary_model(tf.expand_dims(datum, 0))
            predictions = tf.argmax(model_out, axis=1)
            loss = loss_function(predictions, model_out)
            last_layer_params = primary_model.layers[-1].trainable_variables
            gradients.append(tape.gradient(loss, last_layer_params))
    gradients = [tf.norm(tf.concat([gradient[0], tf.expand_dims(gradient[1], axis=0)], axis=0), axis=1) for gradient in gradients]
    kmeans = KMeans(n_clusters=data_generator.size, init='k-means++', random_state=VAL_SEED)
    mat = distance_matrix(kmeans.fit(gradients).cluster_centers_, np.asmatrix(gradients))
    closest = [i for i in np.argmin(mat, axis=1)]
    data_generator.X[0] = tf.gather(data_generator.data, closest)
    data_generator.y[0] = tf.gather(data_generator.labels, closest)
    data_generator.indices[0] = tf.convert_to_tensor(closest, dtype=tf.int64)
    data_generator.n = 0
    return data_generator


def entropy_selector(batches, model):
    predictions = tf.reshape(model(tf.reshape(batches, (-1, batches.shape[-1]))), (batches.shape[0], batches.shape[1], -1))
    return int(tf.argmax(tf.reduce_mean(tf.reduce_sum(-predictions * tf.math.log(tf.maximum(predictions, sigma)), axis=2), axis=1), axis=0))


def margin_selector(batches, model):
    predictions = tf.reshape(model(tf.reshape(batches, (-1, batches.shape[-1]))), (batches.shape[0], batches.shape[1], -1))
    sorted_predictions = tf.sort(predictions, axis=2)
    return int(tf.argmin(tf.reduce_mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1), axis=0))


def confidence_selector(batches, model):
    predictions = tf.reshape(model(tf.reshape(batches, (-1, batches.shape[-1]))), (batches.shape[0], batches.shape[1], -1))
    return int(tf.argmin(tf.reduce_mean(tf.reduce_max(predictions, axis=2), axis=1), axis=0))


def random_selector(batches, model):
    return random.randrange(len(batches))


def uniform_selector(batches, model):
    selectors = {0: entropy_selector, 1: margin_selector, 2: confidence_selector, 3: random_selector}
    return selectors[np.random.choice(4)](batches, model)


class BatchGenerator:
    def __init__(self, data, labels, budget, count, size, aggregator=None, balanced=True):
        self.data = data  # The data from which to generate random batches
        self.labels = labels  # The labels of the data from which random batches are generated
        self.count = count  # The number of random batches to generate when generate_batches is called
        self.size = size  # The size of the batches to be generated.  Should be a multiple of the cardinality of the labels (10 in the case of MNIST)
        self.aggregator = aggregator  # The set encoder used to generate metafeature representations of batches
        self.finished = False  # Flag for when the budget is exhausted or nearly exhausted
        self.n = 0  # The index of the batch to return (is set externally)
        self.budget = budget  # The total number of samples that can be labeled
        self.used = 0  # The number of samples that have been labeled so far
        self.balanced = balanced
        self.X, self.y, self.mfs, self.indices = self.generate_batches(balanced)  # The initial batches and relevant data
        self.selected = {}  # For storing the data that has been chosen for labeling
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
    def add(self, data, labels, budgeted=False):
        label_indices = tf.argmax(labels, axis=1)
        for i in self.classes:
            if i in self.selected.keys():
                self.selected[i] = tf.concat([self.selected[i], tf.squeeze(tf.gather(data, tf.where(tf.equal(label_indices, i))), 1)], axis=0)
            else:
                self.selected[i] = tf.squeeze(tf.gather(data, tf.where(tf.equal(label_indices, i))), 1)
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

        selected_X = tf.gather(self.data, indices)
        selected_y = tf.gather(self.labels, indices)
        self.add(selected_X, selected_y, budgeted=True)

        self.data = tf.boolean_mask(self.data, ~tf.reduce_any(tf.equal(tf.range(tf.shape(self.data)[0], dtype=tf.int64), indices[:, tf.newaxis]), axis=0))
        self.labels = tf.boolean_mask(self.labels, ~tf.reduce_any(tf.equal(tf.range(tf.shape(self.labels)[0], dtype=tf.int64), indices[:, tf.newaxis]), axis=0))

        self.n = 0
        if self.budget == self.used:
            self.finished = True
        elif self.budget - self.size < self.used:
            self.X, self.y, self.mfs, self.indices = self.generate_batches(self.balanced)
        else:
            self.X, self.y, self.mfs, self.indices = self.generate_batches(self.balanced)

    # Necessary because selected is a dictionary, but we want a padded tensor to represent our labeled data
    def get_selected(self):
        length = max(self.selected.values(), key=tf.size).shape[0]
        selected = []
        for i, tensor in self.selected.items():
            paddings = tf.constant([[0, length - tensor.shape[0]], [0, 0]])
            selected.append(tf.pad(tensor, paddings, constant_values=0))
        return tf.stack(selected)

    def generate_batches(self, balanced):
        if balanced:
            return self.generate_balanced()
        batches = []
        batches_y = []
        batches_indices = []  # Stores indices of batch elements
        for _ in range(self.count):
            indices = np.random.choice(len(self.data), size=self.size, replace=True)
            batch_data = tf.gather(self.data, indices)
            batch_y = tf.gather(self.labels, indices)
            batches.append(tf.squeeze(batch_data))
            batches_y.append(tf.squeeze(batch_y))
            batches_indices.append(tf.convert_to_tensor(indices, dtype=tf.int64))

        return batches, batches_y, tf.zeros(self.count), tf.reshape(tf.stack(batches_indices), (self.count, self.size))

    # Generates 'count' balanced batches of size 'size' from 'data' as well as generating their metafeatures using 'aggregator'
    def generate_balanced(self):
        batches = []  # Stores data of batches
        batches_y = []  # Stores labels of batches
        batches_mf = []  # Stores metafeature representation of batches
        batches_indices = []  # Stores indices of batch elements

        # Each iteration creates 1 batch, up to 'count' batches.
        for _ in range(self.count):
            # Gets the indices of each class instance as a list of tensors
            num_classes = self.labels.shape[-1]
            class_indices = [tf.where(tf.equal(self.labels[:, i], 1)) for i in range(num_classes)]
            random_balanced_indices = []

            # Draw 'size//num_classes' number of samples randomly for each class
            for i in range(num_classes):
                random_balanced_indices.append(tf.random.shuffle(class_indices[i])[:self.size // num_classes])

            random_balanced_indices = tf.transpose(tf.concat(random_balanced_indices, axis=1), [1, 0])
            batch_data = tf.gather(self.data, random_balanced_indices)
            batch_y = tf.gather(self.labels, random_balanced_indices)
            batches.append(tf.squeeze(batch_data))
            batches_y.append(tf.squeeze(batch_y))
            batches_indices.append(tf.squeeze(random_balanced_indices))
            if self.aggregator is not None:
                batch_mf = self.aggregator[1](self.aggregator[0](torch.tensor(tf.stack(batch_data).numpy())).squeeze(1).unsqueeze(0)).squeeze()
                batch_mf = batch_mf.detach()
                batches_mf.append(tf.squeeze(batch_mf))
            else:
                batches_mf.append(tf.zeros(0))
        return batches, batches_y, tf.stack(batches_mf), batches_indices


regimes = {"entropy": entropy_selector, "margin": margin_selector, "confidence": confidence_selector, "uniform": uniform_selector, "random": random_selector}


def set_generator_n(data_generator, primary_model, surrogate_model, mean, std):
    pool_metafeatures = np.tile(data_generator.aggregator[1](data_generator.aggregator[0](torch.tensor(tf.stack(data_generator.get_selected()).numpy())).squeeze(1).unsqueeze(0)).squeeze().detach(), (data_generator.indices.shape[0], 1))
    predictions = tf.reshape(primary_model(tf.reshape(data_generator.X, (-1, data_generator.X[0].shape[-1])), training=False), (len(data_generator.X), data_generator.size, -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (data_generator.indices.shape[0], 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])

    surrogate_in = np.concatenate((data_generator.mfs, pool_metafeatures, entropy, margin, confidence, used, histogram), axis=1)
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in)[:, 0])


def set_generator_n_raw(data_generator, primary_model, surrogate_model, mean, std):
    predictions = tf.reshape(primary_model(tf.reshape(data_generator.X, (-1, data_generator.X[0].shape[-1])), training=False), (len(data_generator.X), data_generator.size, -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (data_generator.indices.shape[0], 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])

    surrogate_in = np.concatenate((entropy, margin, confidence, used, histogram), axis=1)
    mean = mean[112:]
    std = std[112:]
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in)[:, 0])


def set_generator_n1d_raw(data_generator, primary_model, surrogate_model, mean, std):
    predictions = tf.reshape(primary_model(tf.reshape(data_generator.X, (-1, data_generator.X[0].shape[-1])), training=False), (len(data_generator.X), data_generator.size, -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (data_generator.indices.shape[0], 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])

    surrogate_in = np.concatenate((entropy, margin, confidence, used, histogram), axis=1)
    mean = mean[112:]
    std = std[112:]
    surrogate_in = (surrogate_in - mean) / std
    return np.argmax(surrogate_model.predict(surrogate_in))


def set_generator_n1d(data_generator, primary_model, surrogate_model, mean, std):
    pool_metafeatures = np.tile(data_generator.aggregator[1](data_generator.aggregator[0](torch.tensor(tf.stack(data_generator.get_selected()).numpy())).squeeze(1).unsqueeze(0)).squeeze().detach(), (data_generator.indices.shape[0], 1))
    predictions = tf.reshape(primary_model(tf.reshape(data_generator.X, (-1, data_generator.X[0].shape[-1])), training=False), (len(data_generator.X), data_generator.size, -1))
    entropy = np.mean(-np.sum(predictions * np.log2(np.maximum(predictions, sigma)), axis=2), axis=1).reshape(-1, 1)
    sorted_predictions = np.sort(predictions, axis=2)
    margin = np.mean(sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2], axis=1).reshape(-1, 1)
    confidence = np.mean(np.max(predictions, axis=2), axis=1).reshape(-1, 1)
    used = np.tile([data_generator.used], (data_generator.indices.shape[0], 1))
    histogram = np.array([np.mean([np.histogram(predictions[j, :, i], bins=10, range=(0, 1), density=True)[0] for i in range(predictions.shape[-1])], axis=0) for j in range(predictions.shape[0])])

    surrogate_in = np.concatenate((data_generator.mfs, pool_metafeatures, entropy, margin, confidence, used, histogram), axis=1)
    surrogate_in = (surrogate_in - mean) / std
    # surrogate_in = xgb.DMatrix(surrogate_in)
    return np.argmax(surrogate_model.predict(surrogate_in))


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


def badge(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y):
    print("BADGE")
    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    writer = SummaryWriter("runs/BADGE" f"{idx}")
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
            data_generator = badge_selector(data_generator, primary_model)
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


def surrogate_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, surrogate):
    print("surrogate")

    with open('intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, (intra_setpool, inter_setpool))  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    writer = SummaryWriter("runs/surrogate" f"{idx}")
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


def loss_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, surrogate_loss):
    print("surrogate(loss only)")

    with open('intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)
    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, (intra_setpool, inter_setpool))  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    writer = SummaryWriter("runs/surrogate_loss" f"{idx}")
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


def hat_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, surrogate_hat):
    print("surrogate(hat only)")
    with open('intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)

    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, (intra_setpool, inter_setpool))  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    writer = SummaryWriter("runs/surrogate_hat" f"{idx}")
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


def heuristics_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, surrogate_heuristics):
    print("surrogate(heuristics only)")
    with open('intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)

    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, (intra_setpool, inter_setpool))  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    writer = SummaryWriter("runs/surrogate_heuristics" f"{idx}")
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


def heuristics_hat(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, surrogate_heuristics):
    print("surrogate(heuristics with hat)")
    with open('intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)

    surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
    cardinality = sum(1 for _ in surrogate_data)
    mean = surrogate_data.map(lambda instances, loss, hat: instances).reduce(tf.constant(0.0), lambda instances, acc: instances + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(surrogate_data.map(lambda instances, loss, hat: instances).map(lambda instances: tf.math.squared_difference(instances, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)

    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, (intra_setpool, inter_setpool))  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    writer = SummaryWriter("runs/surrogate_heuristics_hat" f"{idx}")
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
            data_generator.n = set_generator_n1d_raw(data_generator, primary_model, surrogate_heuristics, mean, std)
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


def raw_part(regime, primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, balanced=False):
    print(regime)
    writer = SummaryWriter("runs/" f"{regime}" f"{idx}")
    if "_balanced" in regime:
        balanced = True
        regime, _ = regime.split("_")
    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, batch_count, batch_size, balanced=balanced)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
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


def evaluate(data, idx, budget=DEFAULT_BUDGET, pool_size=DEFAULT_POOL_SIZE, regime="random"):
    torch.manual_seed(VAL_SEED)
    tf.random.set_seed(VAL_SEED)

    data_tensor = tf.concat([tf.convert_to_tensor(tensor[0].numpy()) for tensor in data.values()], axis=0)
    label_tensor = tf.keras.utils.to_categorical(tf.concat([tf.fill((tensor[0].shape[0],), label) for label, tensor in data.items()], axis=0))

    num_test_samples = int(test_ratio * len(data_tensor))
    indices = tf.random.shuffle(tf.range(data_tensor.shape[0]), seed=VAL_SEED)
    train_indices = indices[num_test_samples:]
    test_indices = indices[:num_test_samples]
    train_X = tf.gather(data_tensor, train_indices)
    train_y = tf.gather(label_tensor, train_indices)
    test_X = tf.gather(data_tensor, test_indices)
    test_y = tf.gather(label_tensor, test_indices)

    indices = tf.random.shuffle(tf.range(train_X.shape[0]))
    train_indices = indices[num_test_samples:]
    val_indices = indices[:num_test_samples]
    X = tf.gather(train_X, train_indices)
    y = tf.gather(train_y, train_indices)
    val_X = tf.gather(train_X, val_indices)
    val_y = tf.gather(train_y, val_indices)

    torch.manual_seed(idx)
    tf.random.set_seed(idx)

    shuffled_indices = tf.random.shuffle(tf.range(X.shape[0]), seed=idx)
    X = tf.gather(X, shuffled_indices)
    y = tf.gather(y, shuffled_indices)
    pool_X, unlabeled_X = X[:pool_size], X[pool_size:]
    pool_y, unlabeled_y = y[:pool_size], y[pool_size:]

    file_path = f"{directory}/initial_models/poolsize{pool_size}_{idx}.pkl"
    lock_path = f"{directory}/initial_models/lock{pool_size}_{idx}.lock"
    while os.path.exists(lock_path):
        time.sleep(1)
    primary_model = Sequential([Dense(10, input_shape=(data_tensor.shape[-1],), activation='softmax')])
    primary_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if not os.path.isfile(file_path):
        with open(lock_path, 'w') as lock_file:
            primary_model.fit(pool_X, pool_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)])
            with open(file_path, 'wb') as output:
                pickle.dump(primary_model.get_weights(), output)
        os.remove(lock_path)
    else:
        with open(file_path, 'rb') as model_file:
            pool_weights = pickle.load(model_file)
            primary_model.set_weights(pool_weights)

    if regime == "badge":
        badge(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y)
    elif regime == "surrogate":
        with open(deepset_surrogate_train.surrogate_file, 'rb') as file:
            model = pickle.load(file)
            surrogate_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, model)
    elif regime == "surrogate_loss":
        with open(deepset_surrogate_train.surrogate_file_loss, 'rb') as file:
            model = pickle.load(file)
            loss_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, model)
    elif regime == "surrogate_hat":
        with open(deepset_surrogate_train.surrogate_file_hat, 'rb') as file:
            model = pickle.load(file)
            hat_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, model)
    elif regime == "surrogate_heuristics":
        with open(deepset_surrogate_train.surrogate_file_heuristics, 'rb') as file:
            model = pickle.load(file)
            heuristics_part(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, model)
    elif regime == "surrogate_heuristics_hat":
        with open(deepset_surrogate_train.surrogate_file_heuristics_hat, 'rb') as file:
            model = pickle.load(file)
            heuristics_hat(primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y, model)
    else:
        raw_part(regime, primary_model, unlabeled_X, unlabeled_y, budget, pool_X, pool_y, idx, test_X, test_y, val_X, val_y)


def main():
    parser = argparse.ArgumentParser("cluster_evaluate")
    parser.add_argument("-i", type=int)
    parser.add_argument("--regime", type=str)
    args = parser.parse_args()
    i = args.i
    regime = args.regime
    for file in os.listdir(directory):
        if file.endswith('mnistbylabel.pt'):
            with open(directory + '/' + file, 'rb') as f:
                data = torch.load(f)
    evaluate(data, i, regime=regime)


if __name__ == '__main__':
    main()
