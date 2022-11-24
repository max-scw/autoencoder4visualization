import numpy as np
import pandas as pd
from time import time

from keras.models import Model
from keras.layers import (
    Conv1D,
    UpSampling1D,
    MaxPooling1D,
    GlobalMaxPool1D,
    Input,
    Dense,
    Flatten,
    Reshape,
    Dropout,
    BatchNormalization
)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from matplotlib import pyplot as plt


class Autoencoder:
    __early_stopping_after_n_epochs = 10
    _steps_per_epoch = 5

    __shape_before_global_pooling = None
    _encoder_input = None
    _autoencoder = None

    def __init__(self,
                 sig_len: int,
                 sig_dim: int = 1,
                 n_layers: int = 3,
                 kernel_sz: int = 3,
                 stride_sz: int = 2,
                 compressed_dim: int = 2,
                 n_width: int = 4
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

        assert compressed_dim > 1, f"Number of filters in first layer (i.e. the initial width) should be greater than one but was {n_width}."
        self.__n_filter_init = n_width

    def build_encoder(self, build_model: bool = True):
        encoder_input = Input(shape=(self.sig_len, self.sig_dim))
        self._encoder_input = encoder_input

        encoder = encoder_input
        encoder = Dropout(rate=0.1)(encoder)
        print(f"encoder.shape={encoder.shape} (Dropout)")
        for i in range(self.n_layers):
            n_filter_up = 2 ** (self.__n_filter_init + i)

            encoder = Conv1D(n_filter_up, self.kernel_sz, activation="relu", padding="same")(encoder)
            print(f"encoder.shape={encoder.shape} (Conv1D)")
            # encoder = BatchNormalization()(encoder)
            # if i < (self.n_layers - 1):
            encoder = MaxPooling1D(self.stride_sz, padding="same")(encoder)
            print(f"encoder.shape={encoder.shape} (MaxPooling1D)")
            if i == (self.n_layers // 2):
                encoder = Dropout(rate=0.1)(encoder)
                print(f"encoder.shape={encoder.shape} (Dropout)")
        self.__shape_before_global_pooling = encoder.shape[1:]

        encoder = GlobalMaxPool1D()(encoder)
        print(f"encoder.shape={encoder.shape} (GlobalMaxPool1D)")
        encoder = Flatten()(encoder)
        print(f"encoder.shape={encoder.shape} (Flatten)")
        encoder = Dense(self.compressed_dim, activation="relu")(encoder)
        print(f"encoder.shape={encoder.shape} (Dense)")

        if build_model:
            encoder = Model(encoder_input, encoder)
            # TODO load weights

        return encoder

    def get_decoder(self, encoder=None):
        if encoder is None:
            decoder_input = Input(shape=self.compressed_dim)
        else:
            decoder_input = encoder

        decoder = Dense(self.__shape_before_global_pooling[-1])(decoder_input)
        print(f"decoder.shape={decoder.shape} (Dense)")
        decoder = Reshape((self.sig_dim, -1))(decoder)
        print(f"decoder.shape={decoder.shape} (Reshape)")
        decoder = UpSampling1D(self.__shape_before_global_pooling[0])(decoder)
        print(f"decoder.shape={decoder.shape} (UpSampling1D)")
        for i in range(self.n_layers):
            n_filter_down = 2 ** (self.__n_filter_init + self.n_layers - i -1)
            decoder = Conv1D(n_filter_down, self.kernel_sz, activation="relu", padding="same")(decoder)
            print(f"decoder.shape={decoder.shape} (Conv1D)")
            decoder = UpSampling1D(self.stride_sz)(decoder)
            print(f"decoder.shape={decoder.shape} (UpSampling1D)")
        # cost function
        decoder = Conv1D(self.sig_dim, self.kernel_sz, activation="sigmoid", padding="same")(decoder)

        if encoder is None:
            decoder = Model(decoder_input, decoder)
            # TODO load weights
        return decoder

    def build_autoencoder(self):
        encoder = self.build_encoder(build_model=False)
        decoder = self.get_decoder(encoder)
        autoencoder = Model(self._encoder_input, decoder)
        return autoencoder

    @property
    def autoencoder(self):
        if self._autoencoder is None:
            self._autoencoder = self.build_autoencoder()
            print("build new autoencoder model")  # FIXME: for debugging
        return self._autoencoder

    # def _build_generator(self, data):
    #     self._steps_per_epoch

    def fit(self, data_tf, epochs: int = 100) -> pd.DataFrame:
        model = self.autoencoder
        model.compile(optimizer="adam",
                      loss='mse')

        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=self.__early_stopping_after_n_epochs,
                                   restore_best_weights=True,
                                   min_delta=1e-4
                                   ),
                     ReduceLROnPlateau(monitor="val_loss",
                                       factor=0.2,
                                       patience=5,
                                       min_lr=1e-6,
                                       min_delta=1e-6,
                                       cooldown=2,
                                       )
                     ]
        t1 = time()
        history = model.fit(x=data_tf,
                            y=data_tf,
                            epochs=epochs,
                            steps_per_epoch=self._steps_per_epoch,
                            callbacks=callbacks,
                            validation_split=0.2,
                            shuffle=False,  # FIXME: shuffle data
                            )
        dt = time() - t1
        print(f"Training took {dt: 0.4} seconds.")
        self._autoencoder = model
        return pd.DataFrame(history.history)


if __name__ == "__main__":
    sig_len_max = 128  # FIXME: doesn't work with all sizes
    sig_dimension = 1

    # create toy data
    n_signals = 10

    data = (np.random.random((n_signals, sig_len_max)) +
            np.asarray(list(np.arange(sig_len_max / 8)) * 8).flatten()) / (sig_len_max / 8)
    print(f"data.shape={data.shape}")
    data = np.asarray(list(data) * 1000)

    # TODO: make input length variable

    auto = Autoencoder(sig_len=sig_len_max, sig_dim=sig_dimension, n_layers=5)
    auto.fit(data)

    sig = data[0, :]
    sig_prd = auto.autoencoder.predict(np.expand_dims(sig, 0))

    plt.plot(sig, color="b")
    plt.plot(sig_prd.flatten(), color="r")
    plt.show()
