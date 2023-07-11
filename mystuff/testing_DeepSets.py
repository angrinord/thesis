import os
import pickle

import numpy as np
import keras.backend as K
from keras import Sequential, backend
from keras.layers import Input, Dense, Lambda, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

batch_count = 2000
batch_size = 10
super_batch_count = 150
sub_batch_size = 10
primary_epochs = 10


def get_deepset_model(images):
    adam = Adam(lr=1e-3, epsilon=1e-3)

    # Encoder
    input_img = Input(shape=(images.shape[1], images.shape[-1],))  # TODO allow for masking so sub-batch can vary in size
    x = TimeDistributed(Dense(300, activation='tanh'))(input_img)
    # x = Embedding(images.shape[0], images.shape[1], mask_zero=True, trainable=False)(input_img)  # Purpose???
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)

    # Aggregator
    x = backend.mean(x, axis=1)
    x = Dense(30)(x)

    model = Model(input_img, x)
    model.compile(optimizer=adam, loss='mae')
    # test_out = test.predict(images)

    return model


def split_model(model, X, split=4):
    encoder_layers = model.layers[:split]  # Example: First 5 layers
    encoder_model = Sequential(encoder_layers)
    encoder_model.build(input_shape=model.input_shape)
    encoder_model.set_weights(model.get_weights()[:2*(split-1)])

    aggregator_layers = model.layers[split:]
    aggregator_model = Sequential(aggregator_layers)

    # TODO once batching fixed, simplify this
    aggregator_model = Sequential()
    aggregator_model.add(aggregator_layers[0])
    aggregator_model.add(aggregator_layers[1])
    aggregator_layers[2].set_weights(model.get_weights()[-2:])
    aggregator_model.add(aggregator_layers[2])
    # aggregator_model.build(input_shape=encoder_model.output_shape)
    # # aggregator_model.compile()
    # aggregator_model.set_weights(model.get_weights()[2*(split-2):])

    # testing stuff
    preds = encoder_model(X)
    preds = aggregator_model(preds)

    return encoder_model, aggregator_model


def generate_batches(data, labels, count, size, metafeatures=None):
    batches = []
    batches_y = []
    for _ in range(count):
        indices = np.random.choice(len(data), size=size, replace=True)
        batch_data = data[indices]
        batch_y = labels[indices]
        # batch_metafeatures = metafeature_batch_computation(metafeatures[indices])
        # batches.append((batch_data, batch_y, batch_metafeatures))
        batches.append(batch_data)
        batches_y.append(batch_y)
    return np.array(batches), np.array(batches_y)


def metafeature_instance_computation(X, y):
    # TODO any computation that can be done instance-wise
    return [0]


def metafeature_batch_computation(instance_metafeatures):
    # TODO any computation that cannot be done instance-wise
    return 0


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


def train_non_random(batches, X_test, y_test, set_encoder):
    primary_model = Sequential([
        Dense(64, activation='relu', input_shape=(64,)),
        Dense(10, activation='softmax')])
    primary_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    surrogate_model = Sequential([
        Dense(64, activation='relu', input_shape=(1, 30)),
        Dense(1)
    ])
    surrogate_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO Create Loss metric(relative_improvement) and surrogate targets (set_encoding+remaining_budget+labeled_pool_class_counts+etc)

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
    # sum of the digits
    return np.mean(y_train, axis=1)  # TODO less dumb labels


def main():
    data = load_digits()

    X, y = generate_batches(data.data, data.target, batch_count, batch_size)
    encoder_y = get_encoder_labels(X, y)
    X_train, X_test, encoder_train, encoder_test = train_test_split(X, encoder_y, test_size=0.2, random_state=42)

    # Set Encoder Stuff
    if os.path.isfile("deepset.pkl"):
        with open('deepset.pkl', 'rb') as input:
            deep_we = pickle.load(input)
            set_encoder = get_deepset_model(X_train)
            set_encoder.set_weights(deep_we)
    else:
        K.clear_session()
        set_encoder = get_deepset_model(X_train)

        # train
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=20, min_lr=0.000001)
        set_encoder.fit(X_train, encoder_train, epochs=500, shuffle=True, validation_split=0.0123456789, callbacks=[reduce_lr])
        deep_we = set_encoder.get_weights()
        # save weights
        with open('deepset.pkl', 'wb') as output:
            pickle.dump(deep_we, output)
    # encoder, aggregator = split_model(set_encoder, X_test)
    # batches = generate_batches(X_train, y_train, batch_count, batch_size)

    # train_non_random(batches, X_test, y_test, set_encoder)
    train_random(batches, X_test, y_test)


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


if __name__ == '__main__':
    main()
