import numpy as np

from keras.models import Sequential

from keras.models import Model
from keras.layers import (
    Conv1D,
    UpSampling1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Input,
    Dense,
    Flatten,
    Reshape,
    Dropout,
    Masking
    )
from keras.optimizers import Adam
#from keras.objectives import categorical_crossentropy


if __name__ == "__main__":
    # create toy data
    sig_len_max = 110
    sig_len = np.random.randint(low=90, high=sig_len_max)
    sig_dim = 2
    sig = np.random.random(size=(sig_len, sig_dim))


    # around 600K parameters
    n_filters = 32
    sz_kernel = 3

    n_stride = 2

    n_layers = 3
    # assuming zeropadded input!
    # TODO: make input length variable
    encoder_input = Input(shape=(sig_len_max, sig_dim))
    encoder = encoder_input
    for i in range(n_layers):
        n_filter_up = 8 * (2**i)
        encoder = Conv1D(n_filter_up, sz_kernel, activation='relu', padding='same')(encoder)
        encoder = MaxPooling1D(n_stride, padding='same')(encoder)
    encoder = GlobalAveragePooling1D()(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(2, activation="relu")(encoder)

    decoder = Dense(n_filter_up)(encoder)
    decoder = Reshape((1, -1))(encoder)
    for i in range(n_layers):
        n_filter_down = n_filter_up // (2**i)
        decoder = Conv1D(n_filter_down, sz_kernel, activation='relu', padding="same")(decoder)
        decoder = UpSampling1D(n_stride)(decoder)
    decoder = Conv1D(1, sz_kernel, activation='sigmoid', padding='same')(decoder)

    autoencoder = Model(encoder_input, decoder)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')