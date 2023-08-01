import os
import pickle
import random
from logging import warning
from typing import Iterator

import numpy as np
import keras.backend as K
import tensorflow
from keras import Sequential, backend
from keras.layers import Input, Dense, Lambda, TimeDistributed
from keras.models import Model, clone_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping
from keras.utils import io_utils, Sequence
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

batch_count = 2000
batch_size = 10
set_enc_range = (2, 64)
super_batch_count = 150
primary_epochs = 10
set_encoder_file = "deepset1.pkl"
surrogate_data_file = "surrogate_data1.pkl"


def get_deepset_model(data_dim, images=None):  # NOTE: Passing images is only necessary for testing.
    adam = Adam(lr=1e-3, epsilon=1e-3)

    # Encoder
    # TimeDistributed should leave the latent features uncorrelated across instances.
    input_img = Input(shape=(None, data_dim, ))
    x = TimeDistributed(Dense(300, activation='tanh'))(input_img)
    x = TimeDistributed(Dense(100, activation='tanh'))(x)
    x = TimeDistributed(Dense(30, activation='tanh'))(x)

    # Aggregator
    x = backend.mean(x, axis=1)
    x = Dense(30)(x)

    model = Model(input_img, x)
    model.compile(optimizer=adam, loss='mae')
    # test_out = test.predict(images)

    return model


def split_model(model, X=None, split=4):  # NOTE: Passing X is only necessary for testing.
    split_index = 2*(split-1)

    # Encoder is half of set encoder that is done instance-wise
    encoder_layers = model.layers[:split]
    encoder_model = Sequential(encoder_layers)
    encoder_model.build(input_shape=model.input_shape)
    encoder_model.set_weights(model.get_weights()[:split_index])

    # Aggregator is half of set encoder that is done batch-wise
    # TODO generalize to arbitrary architecture.  Right now this breaks if arch of set encoder changes.
    agg_in = Input(shape=encoder_model.output_shape[1:])
    agg_x = backend.mean(agg_in, axis=1)
    agg_x = Dense(30)(agg_x)
    aggregator_model = Model(agg_in, agg_x)
    aggregator_model.set_weights(model.get_weights()[split_index:])

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


def batch_pool(pool_X, pool_y, pool_mfs, size, aggregator):
    assert len(pool_X) == len(pool_X) == len(pool_mfs)
    length = len(pool_X)
    p = np.random.permutation(length)
    pool_X = np.reshape(pool_X[p], (length//size, size, pool_X.shape[-1]))
    pool_y = np.reshape(pool_y[p], (length//size, size))
    pool_mfs = aggregator(np.reshape(pool_mfs[p], (length//size, size, pool_mfs.shape[-1])))
    return pool_X, pool_y, pool_mfs


def train_random(batches, X_test, y_test):
    primary_model = Sequential([
        Dense(64, activation='relu', input_shape=(64,)),
        Dense(10, activation='softmax')])
    primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    training_pool = list(batches.pop())

    log_dir = "logs/"
    tensorboard_callback = MyTensorBoard(log_dir=log_dir, histogram_freq=1)
    for i, batch in enumerate(batches):
        primary_model.fit(training_pool[0], training_pool[1], epochs=primary_epochs, batch_size=32, verbose=0)
        loss, accuracy = primary_model.evaluate(X_test, y_test, callbacks=[tensorboard_callback])
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        training_pool[0] = np.concatenate((training_pool[0], batch[0]))
        training_pool[1] = np.concatenate((training_pool[1], batch[1]))


def get_encoder_labels(X_train, y_train):
    # mean of the digits
    do_avg = np.vectorize(np.average)
    return do_avg(y_train)
    # return np.mean(y_train, axis=1)  # TODO less dumb labels


def train_non_random(X, y, metafeatures, set_encoder, encoder, aggregator, budget=100, pool_size=100):
    # Get initial training pool
    pool_X, X = X[:pool_size], X[pool_size:]
    pool_y, y = y[:pool_size], y[pool_size:]
    pool_mfs, metafeatures = metafeatures[:pool_size], metafeatures[pool_size:]

    # Reshape training pool into batches
    batched_X, batched_y, batched_mfs = batch_pool(pool_X, pool_y, pool_mfs, batch_size, aggregator)

    # Simple primary model
    primary_in = Input(shape=set_encoder.input_shape[-1])
    primary_1 = Dense(64, activation='relu')(primary_in)
    primary_out = Dense(10, activation='softmax')(primary_1)
    primary_model = Model(primary_in, primary_out)
    primary_model.compile(optimizer='adam', loss='mse')

    # Simple surrogate model
    # TODO Surrogate should have more comprehensive inputs
    surrogate_in = Input(shape=set_encoder.output_shape[-1])
    surrogate_1 = Dense(64, activation='relu')(surrogate_in)
    surrogate_out = Dense(1)(surrogate_1)
    surrogate_model = Model(surrogate_in, surrogate_out)
    surrogate_model.compile(optimizer='adam', loss='mse')

    # Online training on the initial training pool so that surrogate can do a little learning (maybe)
    train_X = [batched_X[0]]
    train_y = [batched_y[0]]
    train_mfs = []
    surrogate_y = []
    primary_model.fit(np.concatenate(train_X), np.concatenate(train_y), epochs=primary_epochs, batch_size=1, verbose=0)
    old_loss = primary_model.evaluate(pool_X, pool_y)
    for i, x in enumerate(batched_X[1:], 1):
        # Add next batch to training data
        train_X.append(x)
        train_y.append(batched_y[i])
        train_mfs.append(batched_mfs[i])

        # Fit and evaluate primary model on new training data
        primary_model.fit(np.concatenate(train_X), np.concatenate(train_y), epochs=primary_epochs, verbose=0)  # Should epoch be 1?
        new_loss = primary_model.evaluate(pool_X, pool_y)

        # Fit and evaluate surrogate on impact new batch had on loss
        surrogate_y.append(np.array(old_loss-new_loss).reshape((-1, 1)))
        surrogate_model.fit(np.stack(train_mfs), np.concatenate(surrogate_y), epochs=primary_epochs, verbose=0)  # Should epoch be 1?

        old_loss = new_loss

    for i in range(budget):
        batches_X, batches_y, batches_mfs, batches_indices = generate_batches(X, y, 1000, (batch_size, batch_size), metafeatures, aggregator)
        predictions = surrogate_model.predict(batches_mfs)
        next_index = np.argmax(predictions)
        predicted_difference = predictions[next_index]

        train_X.append(batches_X[next_index])
        train_y.append(batches_y[next_index])
        train_mfs.append(batches_mfs[next_index])

        primary_model.fit(np.concatenate(train_X), np.concatenate(train_y), epochs=primary_epochs, verbose=0)  # Should epoch be 1?
        actual_loss = primary_model.evaluate(np.concatenate(train_X), np.concatenate(train_y))

        surrogate_y.append(np.array(old_loss - actual_loss).reshape((-1, 1)))
        surrogate_model.fit(np.stack(train_mfs), np.concatenate(surrogate_y), epochs=primary_epochs, verbose=0)

        old_loss = actual_loss
        # TODO where to begin...
    print('test')


def entropy_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    return np.argmax(np.mean(-np.sum(predictions * np.log2(predictions), axis=1), axis=1))


def margin_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    sorted_predictions = np.sort(predictions, axis=1)
    return np.argmin(np.mean(sorted_predictions[:, -1] - sorted_predictions[:, -2], axis=1))


def confidence_selector(batches, model):
    predictions = model.predict(batches.reshape(-1, batches.shape[-1])).reshape(batches.shape[0], batches.shape[1], -1)
    return np.argmin(np.mean(1 - np.max(predictions, axis=1), axis=1))


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
        if self.budget - self.size < self.used:
            self.finished = True
        else:
            self.X, self.y, self.mfs, self.indices = generate_batches(self.data, self.labels, self.count, (self.size, self.size), self.metafeatures, self.aggregator)


def pretrain(data, labels, metafeatures, set_encoder, encoder, aggregator, budget=1000, pool_size=100):
    regimes = [random_selector, entropy_selector, margin_selector, confidence_selector, uniform_selector]
    regime_iterations = 10
    surrogate_X = []
    surrogate_y = []
    surrogate_y_hat = []

    X, val_X, y, val_y, mfs, _ = train_test_split(data, labels, metafeatures, test_size=0.0123456789)

    # Initial Training Pool
    pool_X, X = X[:pool_size], X[pool_size:]
    pool_y, y = y[:pool_size], y[pool_size:]
    pool_mfs, mfs = mfs[:pool_size], mfs[pool_size:]

    # Simple primary model
    primary_in = Input(shape=set_encoder.input_shape[-1])
    primary_hidden = Dense(32, activation='relu')(primary_in)
    primary_hidden = Dense(32, activation='relu')(primary_hidden)
    primary_out = Dense(10, activation='softmax')(primary_hidden)
    primary_model = Model(primary_in, primary_out)
    primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    primary_model.fit(pool_X, pool_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)])
    pool_weights = primary_model.get_weights()

    one_step_model = clone_model(primary_model)
    one_step_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Iterate over different batch selection heuristics
    for regime in regimes:
        # Train model using given selection heuristic multiple times
        for i in range(regime_iterations):
            primary_model.set_weights(pool_weights)  # Every model starts trained on same initial training pool
            data_generator = BatchGenerator(X, y, budget, 1000, batch_size, mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
            data_generator.selected_mfs.append(pool_mfs)
            data_generator.selected_X.append(pool_X)
            data_generator.selected_y.append(pool_y)
            initial_loss = None
            writer = SummaryWriter("runs/" f"{regime.__name__}" f"{i+1}")
            batch = 0
            labeled_X = pool_X
            labeled_y = pool_y
            try:  # Iterators are supposed to throw StopIteration exception when they reach the end
                while True:  # This goes until budget is exhausted
                    batch += 1
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
                    entropy = np.array([np.mean(-np.sum(predictions * np.log2(predictions), axis=1))])
                    sorted_predictions = np.sort(predictions, axis=1)
                    margin = np.array([np.mean(sorted_predictions[:, -1] - sorted_predictions[:, -2])])
                    confidence = np.array([np.mean(1 - np.max(predictions, axis=1))])
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
                    writer.add_scalar('loss_change', val_loss, batch)
                    writer.add_scalar('accuracy', accuracy, batch)
                    writer.add_scalar('loss_hat_change', val_loss_hat, batch)
                    writer.add_scalar('hat_accuracy', accuracy_hat, batch)
            except StopIteration:
                with open(surrogate_data_file, 'wb') as f:
                    pickle.dump((surrogate_X, surrogate_y, surrogate_y_hat), f)


def set_generator_n(data_generator, primary_model, surrogate_model):
    return 0


def train_surrogate(data, labels, metafeatures, set_encoder, encoder, aggregator, surrogate_data, targets, targets_hat, budget=1000, pool_size=100):
    # Simple surrogate model
    surrogate_in = Input(shape=surrogate_data[0].shape[-1])
    surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
    surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
    surrogate_out = Dense(2)(surrogate_hidden)
    surrogate_model = Model(surrogate_in, surrogate_out)
    surrogate_model.compile(optimizer='adam', loss='mae')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=0, patience=20, min_lr=0.000001)
    surrogate_model.fit(np.array(surrogate_data), np.column_stack((targets, targets_hat)), epochs=1, validation_split=0.15, callbacks=[reduce_lr])

    regimes = [random_selector, entropy_selector, margin_selector, confidence_selector, uniform_selector]
    regime_iterations = 10

    X, val_X, y, val_y, mfs, _ = train_test_split(data, labels, metafeatures, test_size=0.0123456789)

    # Initial Training Pool
    pool_X, X = X[:pool_size], X[pool_size:]
    pool_y, y = y[:pool_size], y[pool_size:]
    pool_mfs, mfs = mfs[:pool_size], mfs[pool_size:]

    # Simple primary model
    primary_in = Input(shape=set_encoder.input_shape[-1])
    primary_hidden = Dense(32, activation='relu')(primary_in)
    primary_hidden = Dense(32, activation='relu')(primary_hidden)
    primary_out = Dense(10, activation='softmax')(primary_hidden)
    primary_model = Model(primary_in, primary_out)
    primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    primary_model.fit(pool_X, pool_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)])
    pool_weights = primary_model.get_weights()

    primary_model.set_weights(pool_weights)  # Every model starts trained on same initial training pool
    data_generator = BatchGenerator(X, y, budget, 1000, batch_size, mfs, aggregator)  # Generate 1000 random batches from remaining unlabelled pool
    data_generator.selected_mfs.append(pool_mfs)
    data_generator.selected_X.append(pool_X)
    data_generator.selected_y.append(pool_y)

    writer = SummaryWriter("runs/surrogate")
    batch = 0
    labeled_X = pool_X
    labeled_y = pool_y
    try:  # Iterators are supposed to throw StopIteration exception when they reach the end
        while True:  # This goes until budget is exhausted
            batch += 1
            data_generator.n = set_generator_n(data_generator, primary_model, surrogate_model)  # Sets the new batch
            x, label = next(data_generator)  # get the batch
            labeled_X = np.vstack((labeled_X, x))
            labeled_y = np.concatenate((labeled_y, label))

            primary_model.fit(labeled_X, labeled_y, validation_data=(val_X, val_y), callbacks=[EarlyStopping(patience=20)], verbose=0)
            val_loss, accuracy = primary_model.evaluate(val_X, val_y)

            data_generator.regenerate()  # Regenerate 1000 batches, excluding already used instances

            writer.add_scalar('loss_change', val_loss, batch)
            writer.add_scalar('accuracy', accuracy, batch)
    except StopIteration:
        pass


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
    data = load_digits()

    # Set Encoder Stuff
    if os.path.isfile(set_encoder_file):
        with open(set_encoder_file, 'rb') as input:
            deep_we = pickle.load(input)
            set_encoder = get_deepset_model(data.data.shape[-1])
            set_encoder.set_weights(deep_we)
    else:
        X, val_X, y, val_y = train_test_split(data.data, data.target, test_size=0.05)
        X, y = generate_batches(X, y, batch_count, set_enc_range)
        val_X, val_y = generate_batches(val_X, val_y, 50, set_enc_range)
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
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=20, min_lr=0.000001)
        set_encoder.fit(set_generator, epochs=300, shuffle=True, callbacks=[reduce_lr], validation_data=val_generator)
        deep_we = set_encoder.get_weights()
        # save weights
        with open(set_encoder_file, 'wb') as output:
            pickle.dump(deep_we, output)

    encoder, aggregator = split_model(set_encoder)
    instance_mfs = np.squeeze(encoder.predict(np.expand_dims(data.data, 1)), 1)

    surrogate_X, surrogate_y, surrogate_y_hat = None, None, None
    if not os.path.isfile(surrogate_data_file):
        pretrain(data.data, data.target, instance_mfs, set_encoder, encoder, aggregator)
    with open(surrogate_data_file, 'rb') as input:
        surrogate_X, surrogate_y, surrogate_y_hat = pickle.load(input)
    train_surrogate(data.data, data.target, instance_mfs, set_encoder, encoder, aggregator, surrogate_X, surrogate_y, surrogate_y_hat)
    # train_non_random(data.data, data.target, instance_mfs, set_encoder, encoder, aggregator)

    # batches = generate_batches(X_train, y_train, batch_count, batch_size)
    # train_random(batches, X_test, y_test)


if __name__ == '__main__':
    main()
