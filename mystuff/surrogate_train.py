import sys
sys.path.append('/home/angrimson/thesis')
import experiment_util  # IMPORTANT!  DON'T REMOVE.
import pickle
import os
import torch
import xgboost as xgb
import tensorflow as tf


# Constants and Variables Booker
directory = 'mystuff/data'  # Directory that stores preprocessed MNIST and pretrained surrogate data
surrogate_file = "surrogate_test.pkl"
PATIENCE = 5  # Patience for early stopping callbacks.  Could technically be different between different models, but who cares?
VAL_SEED = 0  # Seed for getting the same validation data every time
test_ratio = 0.1  # How much of dataset should be set aside for validation
num_rounds = 100000
param = {
        'objective': 'reg:squarederror',  # Learning task and loss function
        'eval_metric': 'rmse',  # Metric for evaluation
        'n_estimators': 1000,  # Number of boosting rounds (trees)
        'learning_rate': 0.01,  # Step size at each iteration
        'max_depth': 6,  # Maximum depth of a tree
        'min_child_weight': 1,  # Minimum sum of instance weight in a child node
        'subsample': 1.0,  # Fraction of samples used for training each tree
        'colsample_bytree': 1.0,  # Fraction of features used for training each tree
        'gamma': 0,  # Minimum loss reduction required for further partition
        'seed': VAL_SEED  # Random seed for reproducibility
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


def train_surrogate(surrogate_data):
    torch.manual_seed(VAL_SEED)
    tf.random.set_seed(VAL_SEED)
    cardinality = sum(1 for _ in surrogate_data)
    split = int(cardinality*test_ratio)

    surrogate_data = surrogate_data.shuffle(buffer_size=cardinality, seed=0)
    train_data = surrogate_data

    mean = train_data.map(lambda data, loss, hat: data).reduce(tf.constant(0.0), lambda data, acc: data + acc) / tf.cast(cardinality, tf.float32)
    std = tf.sqrt(train_data.map(lambda data, loss, hat: data).map(lambda data: tf.math.squared_difference(data, mean)).reduce(tf.constant(0.0), lambda squared_diff, acc: squared_diff + acc) / tf.cast(cardinality, tf.float32))
    def standardization(feature): return (feature - mean) / std
    train_data = train_data.map(lambda data, loss, hat: (standardization(data), loss, hat))

    val_data = train_data.take(split)
    train_data = train_data.skip(split)

    surrogate_X, surrogate_y = [], []
    for data, loss, hat in iter(train_data):
        surrogate_X.append(data)
        surrogate_y.append(tf.concat([loss, hat], axis=0))

    surrogate_val_X, surrogate_val_y = [], []
    for data, loss, hat in iter(val_data):
        surrogate_val_X.append(data)
        surrogate_val_y.append(tf.concat([loss, hat], axis=0))

    dtrain = xgb.DMatrix(surrogate_X, label=surrogate_y)
    evallist = [(xgb.DMatrix(surrogate_val_X, label=surrogate_val_y), "val")]
    surrogate_model = xgb.train(param, dtrain, num_rounds, evallist, early_stopping_rounds=PATIENCE)
    with open(surrogate_file, 'wb') as f:
        pickle.dump(surrogate_model, f)


def main():
    if not os.path.isfile(f'{directory}/{surrogate_file}'):
        surrogate_data = tf.data.TFRecordDataset([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.tfrecords')]).map(get_instance)
        train_surrogate(surrogate_data)
        return 0
    print(f'file \'{surrogate_file}\' already exists.  Skipping job...')


if __name__ == '__main__':
    main()
