import sys
import time

sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
import argparse
import os
import pickle
import random
import numpy as np
import tensorflow as tf
import torch
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import clone_model
from keras.utils import io_utils
from torch.utils.tensorboard import SummaryWriter

# Constants and Variables Booker
batch_size = 10  # How big should the generated batches be.  Best be a multiple of data cardinality (10 in MNIST's case)
sigma = 1e-10  # This is so entropy doesn't break
directory = 'mystuff/data/fixed_main'  # Directory that stores preprocessed MNIST and pretrained surrogate data
NUM_RANDOM_BATCHES = 1000  # Number of random batches BatchGenerator should create
PATIENCE = 10  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
VAL_SEED = 0  # Seed for getting the same validation data every time
EPOCHS = 10000  # Set very high so that early stopping always happens
test_ratio = 0.1  # How much of dataset should be set aside for validation
DEFAULT_BUDGET = 500
DEFAULT_POOL_SIZE = 0


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


regimes = {"random": random_selector, "entropy": entropy_selector, "margin": margin_selector, "confidence": confidence_selector, "uniform": uniform_selector}


# This class handles the generation of random batches and the storage of data relevant to current and previously generated batches.
class BatchGenerator:
    def __init__(self, data, labels, budget, count, size, aggregator=None):
        self.data = data  # The data from which to generate random batches
        self.labels = labels  # The labels of the data from which random batches are generated
        self.count = count  # The number of random batches to generate when generate_batches is called
        self.size = size  # The size of the batches to be generated.  Should be a multiple of the cardinality of the labels (10 in the case of MNIST)
        self.aggregator = aggregator  # The set encoder used to generate metafeature representations of batches
        self.finished = False  # Flag for when the budget is exhausted or nearly exhausted
        self.n = 0  # The index of the batch to return (is set externally)
        self.budget = budget  # The total number of samples that can be labeled
        self.used = 0  # The number of samples that have been labeled so far
        self.X, self.y, self.mfs, self.indices = self.generate_batches()  # The initial batches and relevant data
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
        if self.budget - self.size < self.used:
            self.finished = True
        else:
            self.X, self.y, self.mfs, self.indices = self.generate_batches()

    # Necessary because selected is a dictionary, but we want a padded tensor to represent our labeled data
    def get_selected(self):
        length = max(self.selected.values(), key=tf.size).shape[0]
        selected = []
        for i, tensor in self.selected.items():
            paddings = tf.constant([[0, length - tensor.shape[0]], [0, 0]])
            selected.append(tf.pad(tensor, paddings, constant_values=0))
        return tf.stack(selected)

    # Generates 'count' balanced batches of size 'size' from 'data' as well as generating their metafeatures using 'aggregator'
    def generate_batches(self):
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
            batches.append(batch_data)
            batches_y.append(batch_y)
            batches_indices.append(random_balanced_indices)
            batch_mf = self.aggregator[1](self.aggregator[0](torch.tensor(tf.stack(batch_data).numpy())).squeeze(1).unsqueeze(0)).squeeze()
            batch_mf = batch_mf.detach()
            batches_mf.append(batch_mf)
        return tf.reshape(tf.stack(batches), (self.count, self.size, self.data.shape[-1])), tf.reshape(tf.stack(batches_y), (self.count, self.size, self.labels.shape[-1])), tf.stack(batches_mf), tf.reshape(tf.stack(batches_indices), (self.count, self.size))


# This function is for the generation of training data for the surrogate model.  The data is composed of... TODO: More comments
def pretrain(intra_setpool, inter_setpool, data, idx, regime_name, budget=DEFAULT_BUDGET, pool_size=DEFAULT_POOL_SIZE):
    torch.manual_seed(VAL_SEED)
    tf.random.set_seed(VAL_SEED)

    data_tensor = tf.concat([tf.convert_to_tensor(tensor[0].numpy()) for tensor in data.values()], axis=0)
    label_tensor = tf.keras.utils.to_categorical(tf.concat([tf.fill((tensor[0].shape[0],), label) for label, tensor in data.items()], axis=0))

    num_test_samples = int(test_ratio * len(data_tensor))
    indices = tf.random.shuffle(tf.range(data_tensor.shape[0]))
    train_indices = indices[num_test_samples:]
    test_indices = indices[:num_test_samples]
    train_X = tf.gather(data_tensor, train_indices)
    train_y = tf.gather(label_tensor, train_indices)
    # test_X = tf.gather(data_tensor, test_indices)
    # test_y = tf.gather(label_tensor, test_indices)

    indices = tf.random.shuffle(tf.range(train_X.shape[0]))
    train_indices = indices[num_test_samples:]
    val_indices = indices[:num_test_samples]
    X = tf.gather(train_X, train_indices)
    y = tf.gather(train_y, train_indices)
    val_X = tf.gather(train_X, val_indices)
    val_y = tf.gather(train_y, val_indices)

    torch.manual_seed(idx)
    tf.random.set_seed(idx)

    shuffled_indices = tf.random.shuffle(tf.range(X.shape[0]))
    X = tf.gather(X, shuffled_indices)
    y = tf.gather(y, shuffled_indices)
    pool_X, unlabeled_X = X[:pool_size], X[pool_size:]
    pool_y, unlabeled_y = y[:pool_size], y[pool_size:]

    file_path = f"{directory}/initial_models/initial{idx}.pkl"
    lock_path = f"{directory}/initial_models/lock{idx}.lock"
    while os.path.exists(lock_path):
        time.sleep(1)

    primary_model = Sequential([Dense(10, input_shape=(data_tensor.shape[-1],), activation='softmax')])
    primary_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    one_step_model = clone_model(primary_model)
    one_step_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    data_generator = BatchGenerator(unlabeled_X, unlabeled_y, budget, NUM_RANDOM_BATCHES, batch_size, (intra_setpool, inter_setpool))  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.add(pool_X, pool_y)
    initial_loss = None
    # writer = SummaryWriter("runs/" f"{regime_name}" f"{idx}")
    labeled_X = pool_X
    labeled_y = pool_y
    # test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=0)
    # writer.add_scalar('loss_change', test_loss, data_generator.used)
    # writer.add_scalar('accuracy', test_accuracy, data_generator.used)
    try:
        while True:
            pool_metafeatures = inter_setpool(intra_setpool(torch.tensor(tf.stack(data_generator.get_selected()).numpy())).squeeze(1).unsqueeze(0)).squeeze().detach()
            data_generator.n = regimes[regime_name](data_generator.X, primary_model)  # Sets the new batch
            x, label = next(data_generator)  # get the batch
            batch_metafeatures = data_generator.mfs[data_generator.n]  # Metafeature vector for the current batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            one_step_model.set_weights(primary_model.get_weights())
            one_step_model.train_on_batch(x, label)  # single gradient update on batch
            val_loss_hat, accuracy_hat = one_step_model.evaluate(val_X, val_y, verbose=0)
            # test_loss_hat, test_accuracy_hat = one_step_model.evaluate(test_X, test_y, verbose=0)

            predictions = primary_model.predict(x, verbose=0)
            primary_model.fit(labeled_X, labeled_y, epochs=EPOCHS, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=PATIENCE)], verbose=0)
            val_loss, accuracy = primary_model.evaluate(val_X, val_y)
            # test_loss, test_accuracy = primary_model.evaluate(test_X, test_y, verbose=0)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances

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
        with open(f'{directory}/{regime_name}{idx}.pkl', 'wb') as f:
            pickle.dump((surrogate_X, surrogate_y, surrogate_y_hat), f)


def main():
    parser = argparse.ArgumentParser("cluster_pretrain")
    parser.add_argument("-i", type=int)
    parser.add_argument("--regime")
    args = parser.parse_args()
    i = args.i
    regime = args.regime
    with open('intra_setpool.pkl', 'rb') as input:
        intra_setpool = pickle.load(input)
    with open('inter_setpool.pkl', 'rb') as input:
        inter_setpool = pickle.load(input)

    for file in os.listdir(directory):
        if file.endswith('mnistbylabel.pt'):
            with open(directory + '/' + file, 'rb') as f:
                data = torch.load(f)
    pretrain(intra_setpool, inter_setpool, data, i, regime)


if __name__ == '__main__':
    main()
