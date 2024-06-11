import sys
import numpy as np
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense
sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
import pickle
import os
import torch
import tensorflow as tf
import xgboost as xgb

# Constants and Variables Booker
directory = 'mystuff/data/fixed_main'  # Directory that stores preprocessed MNIST and pretrained surrogate data
use_xgb = False
if use_xgb:
    file_name = "xgboost_surrogate"
else:
    file_name = "main_surrogate"
surrogate_file = f"{file_name}.pkl"
surrogate_file_loss = f"{file_name}_loss.pkl"
surrogate_file_hat = f"{file_name}_hat.pkl"
surrogate_file_heuristics = f"{file_name}_heuristics.pkl"
surrogate_file_heuristics_hat = f"{file_name}_heuristics_hat.pkl"
surrogate_file_heuristics_loss = f"{file_name}_heuristics_loss.pkl"
surrogate_file_entropy = f"{file_name}_entropy.pkl"
PATIENCE = 20  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
VAL_SEED = 0  # Seed for getting the same validation data every time
test_ratio = 0.1  # How much of dataset should be set aside for validation
EPOCHS = 500

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


def train_surrogate(surrogate_data, variant=surrogate_file):
    torch.manual_seed(VAL_SEED)
    tf.random.set_seed(VAL_SEED)
    cardinality = sum(1 for _ in surrogate_data)
    split = int(cardinality*test_ratio)

    surrogate_data = surrogate_data.shuffle(buffer_size=cardinality, seed=VAL_SEED)
    train_data = surrogate_data

    mean = train_data.map(lambda data, loss, hat: data).reduce(tf.constant(0.0), lambda data, acc: data + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(train_data.map(lambda data, loss, hat: data).map(lambda data: tf.math.squared_difference(data, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)
    def standardization(feature): return (feature - mean) / std
    train_data = train_data.map(lambda data, loss, hat: (standardization(data), loss, hat))

    val_data = train_data.take(split)
    train_data = train_data.skip(split)
    surrogate_X, surrogate_y = [], []
    surrogate_val_X, surrogate_val_y = [], []

    if variant == surrogate_file_loss:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data.numpy())
            surrogate_y.append(loss.numpy())

        surrogate_val_X, surrogate_val_y = [], []
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data.numpy())
            surrogate_val_y.append(loss.numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=len(mean))
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(1)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file_loss, 'wb') as f:
            pickle.dump(surrogate_model, f)

    elif variant == surrogate_file_hat:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data.numpy())
            surrogate_y.append(hat.numpy())
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data.numpy())
            surrogate_val_y.append(hat.numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=len(mean))
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(1)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file_hat, 'wb') as f:
            pickle.dump(surrogate_model, f)

    elif variant == surrogate_file_heuristics:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data[32:].numpy())
            surrogate_y.append(tf.stack([loss, hat], axis=0).numpy())
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data[32:].numpy())
            surrogate_val_y.append(tf.stack([loss, hat], axis=0).numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=len(mean[32:]))
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(2)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file_heuristics, 'wb') as f:
            pickle.dump(surrogate_model, f)

    elif variant == surrogate_file_entropy:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data[32].numpy())
            surrogate_y.append(tf.stack([loss, hat], axis=0).numpy())
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data[32].numpy())
            surrogate_val_y.append(tf.stack([loss, hat], axis=0).numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=1)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(2)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file_heuristics, 'wb') as f:
            pickle.dump(surrogate_model, f)

    elif variant == surrogate_file_heuristics_hat:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data[32:].numpy())
            surrogate_y.append(hat.numpy())
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data[32:].numpy())
            surrogate_val_y.append(hat.numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=len(mean[32:]))
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(1)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file_heuristics_hat, 'wb') as f:
            pickle.dump(surrogate_model, f)

    elif variant == surrogate_file_heuristics_loss:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data[32:].numpy())
            surrogate_y.append(loss.numpy())
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data[32:].numpy())
            surrogate_val_y.append(loss.numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=len(mean[32:]))
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(1)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file_heuristics_loss, 'wb') as f:
            pickle.dump(surrogate_model, f)

    else:
        for data, loss, hat in iter(train_data):
            surrogate_X.append(data.numpy())
            surrogate_y.append(tf.stack([loss, hat], axis=0).numpy())
        for data, loss, hat in iter(val_data):
            surrogate_val_X.append(data.numpy())
            surrogate_val_y.append(tf.stack([loss, hat], axis=0).numpy())
        if use_xgb:
            dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
            evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
            surrogate_model = xgb.train(param, dtrain, EPOCHS, evallist, early_stopping_rounds=PATIENCE)
        else:
            surrogate_in = Input(shape=len(mean))
            surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
            surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
            surrogate_out = Dense(2)(surrogate_hidden)
            surrogate_model = Model(surrogate_in, surrogate_out)
            surrogate_model.compile(optimizer='adam', loss='mse')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=PATIENCE, min_lr=0.000001)
            surrogate_model.fit(np.array(surrogate_X), np.array(surrogate_y), epochs=EPOCHS, validation_data=(np.array(surrogate_val_X), np.array(surrogate_val_y)), callbacks=[reduce_lr])
        with open(surrogate_file, 'wb') as f:
            pickle.dump(surrogate_model, f)


def main():
    if not os.path.isfile(f'{directory}/{surrogate_file}'):
        surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
        print('training surrogate')
        train_surrogate(surrogate_data)
        print('training surrogate loss only')
        train_surrogate(surrogate_data, surrogate_file_loss)
        print('training surrogate hat only')
        train_surrogate(surrogate_data, surrogate_file_hat)
        print('training surrogate heuristics only')
        train_surrogate(surrogate_data, surrogate_file_heuristics)
        print('training surrogate heuristics hat')
        train_surrogate(surrogate_data, surrogate_file_heuristics_hat)
        # print('training surrogate entropy only')
        # train_surrogate(surrogate_data, surrogate_file_entropy)
        print('training surrogate heuristics loss')
        train_surrogate(surrogate_data, surrogate_file_heuristics_loss)
        return 0
    print(f'file \'{surrogate_file}\' already exists.  Skipping job...')


if __name__ == '__main__':
    main()
