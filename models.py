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

class Autoencoder:
    __n_filter_init = 8
    _encoder_input = None

    def __init__(self, 
                 sig_len: int, 
                 sig_dim: int = 1,
                 n_layers: int = 3,
                 kernel_sz: int = 3,
                 stride_sz: int = 2,
                 compressed_dim: int = 2
                 ) -> None:
        assert sig_len > 0, f"Signal length should be positive but was {sig_len}."
        self.sig_len = sig_len

        assert sig_dim > 0, f"Signal dimension should be positive but was {sig_dim}."
        self.sig_dim = sig_dim

        assert n_layers > 0, f"Number of convolution layers should be positive but was {n_layers}."
        self.n_layers = n_layers  

        assert kernel_sz > 0, f"Kernel size of the convolition should be positive but was {kernel_sz}."
        self.kernel_sz = kernel_sz

        assert stride_sz > 0, f"Stride should be positive but was {stride_sz}."
        self.stride_sz = stride_sz

        assert compressed_dim > 0, f"Number of compressed dimension should be positive but was {compressed_dim}."
        self.compressed_dim = compressed_dim


    def get_encoder(self, build_model: bool = True):
        encoder_input = Input(shape=(self.sig_len, self.sig_dim))
        self._encoder_input = encoder_input

        encoder = encoder_input
        for i in range(self.n_layers):
            n_filter_up = self.__n_filter_init * (2**i)
            encoder = Conv1D(n_filter_up, self.kernel_sz, activation='relu', padding='same')(encoder)
            encoder = MaxPooling1D(self.stride_sz, padding='same')(encoder)
        self.__sig_len_before_global_pooling = encoder.shape[1]
        encoder = GlobalAveragePooling1D()(encoder)
        encoder = Flatten()(encoder)
        encoder = Dense(self.compressed_dim, activation="relu")(encoder)

        if build_model:
            encoder = Model(encoder_input, encoder)
            # TODO load weights
        
        return encoder
    
    def get_decoder(self, encoder = None):
        if encoder is None:
            decoder_input = Input(shape=(self.compressed_dim))
        else:
            decoder_input = encoder

        decoder = Reshape((1, -1))(decoder_input)
        decoder = UpSampling1D(self.__sig_len_before_global_pooling)(decoder)
        n_filter_up_init = self.__n_filter_init * (2**self.n_layers)
        for i in range(self.n_layers):
            n_filter_down = n_filter_up_init // (2**i)
            decoder = Conv1D(n_filter_down, self.kernel_sz, activation='relu', padding="same")(decoder)
            decoder = UpSampling1D(self.stride_sz)(decoder)
        decoder = Conv1D(self.sig_dim, self.kernel_sz, activation='sigmoid', padding='same')(decoder)
        
        if encoder is None:
            decoder = Model(decoder_input, decoder)
            # TODO load weights
        return decoder

    def get_autoencoder(self):
        encoder = self.get_encoder(build_model=False)
        decoder = self.get_decoder(encoder)
        autoencoder = Model(self._encoder_input, decoder)
        return autoencoder



if __name__ == "__main__":
    
    sig_len_max = 112 # FIXME: doesn't work with all sizes
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

    # TODO: make input length variable

    

    auto = Autoencoder(sig_len=sig_len_max, sig_dim=sig_dim).get_autoencoder()

    auto.compile(optimizer=Adam(), loss='binary_crossentropy')


    # history = autoencoder.fit(x=data_pad,
    #                           y=data_pad,
    #                           batch_size=5, 
    #                           epochs=1000,
    #                           callbacks=[ReduceLROnPlateau(monitor="loss"), 
    #                                      EarlyStopping(monitor="loss", 
    #                                                    patience=10,
    #                                                    restore_best_weights=True
    #                                                    )
    #                                      ]
    #                           )