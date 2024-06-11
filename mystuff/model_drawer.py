from IPython.core.display import SVG
from keras.utils import model_to_dot, plot_model
from sklearn.datasets import load_digits
from keras import backend
from keras.layers import Input, Dense, TimeDistributed
from keras.models import Model
import pydot
import graphviz


def get_deepset_model(data_dim):
    surrogate_in = Input(data_dim)
    surrogate_hidden = Dense(64, activation='relu')(surrogate_in)
    surrogate_hidden = Dense(64, activation='relu')(surrogate_hidden)
    surrogate_out = Dense(2)(surrogate_hidden)
    surrogate_model = Model(surrogate_in, surrogate_out)
    surrogate_model.compile(optimizer='adam', loss='mse')
    return surrogate_model


def main():
    set_encoder = get_deepset_model(46)
    plot_model(set_encoder, show_shapes=True, show_layer_names=False, show_layer_activations=True, to_file="toy_surrogate.svg")


if __name__ == '__main__':
    main()
