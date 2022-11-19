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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from keras.objectives import categorical_crossentropy


if __name__ == "__main__":
    
    sig_len_max = 112
    sig_dim = 2

    # create toy data
    n_signals = 10
    
    data = []
    for i in range(n_signals):
        sig_len = np.random.randint(low=90, high=sig_len_max)
        data.append(np.random.random(size=(sig_len, sig_dim)))
    # zeropadding
    data_pad = []
    for el in data:
        sig_pad = np.concatenate((el, np.zeros(shape=(sig_len_max-el.shape[0], el.shape[1]))))
        data_pad.append(sig_pad)
    data_pad = np.stack(data_pad, axis=0)



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
    sig_len_before_global_pooling = encoder.shape[1]
    encoder = GlobalAveragePooling1D()(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(2, activation="relu")(encoder)
    model_encoder = Model(encoder_input, encoder)

    #decoder_input = Input(shape=encoder.shape) # FIXME: wrap to function to call twice
    decoder = Dense(n_filter_up)(encoder)
    decoder = Reshape((1, -1))(decoder)
    decoder = UpSampling1D(sig_len_before_global_pooling)(decoder)
    for i in range(n_layers):
        n_filter_down = n_filter_up // (2**i)
        decoder = Conv1D(n_filter_down, sz_kernel, activation='relu', padding="same")(decoder)
        decoder = UpSampling1D(n_stride)(decoder)
    decoder = Conv1D(sig_dim, sz_kernel, activation='sigmoid', padding='same')(decoder)
    #model_decoder = Model(decoder_input, decoder)

    autoencoder = Model(encoder_input, decoder)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')


    history = autoencoder.fit(x=data_pad,
                              y=data_pad,
                              batch_size=5, 
                              epochs=1000,
                              callbacks=[ReduceLROnPlateau(monitor="loss"), 
                                         EarlyStopping(monitor="loss", 
                                                       patience=10,
                                                       restore_best_weights=True
                                                       )
                                         ]
                              )
