import numpy as np
import pandas as pd
import re
from time import time
from typing import Union, Tuple

from keras.models import Model
from keras.layers import (
    Conv1D,
    Conv1DTranspose,
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
import keras.backend as K

from matplotlib import pyplot as plt


def count_model_parameters(model: Model) -> Tuple[int, int]:
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return int(trainable_count), int(non_trainable_count)


class Autoencoder:
    __early_stopping_after_n_epochs = 10
    _steps_per_epoch = 5

    __shape_hidden_layer = None
    _encoder_input = None
    _autoencoder = None

    __model_types = {"CNN": r"CNN",
                     "fullyCNN": r"(fully?[\s\t\-]?CNN)|(CNN[\s\t\-]?full)",
                     "FCN": r"(FCN)|(Dense)",
                     "LSTM": r"LSTM"
                     }

    def __init__(self,
                 sig_len: int,
                 sig_dim: int = 1,
                 n_layers: int = 3,
                 kernel_sz: int = 3,
                 stride_sz: int = 2,
                 compressed_dim: int = 2,
                 n_width: int = 4,
                 model_type: str = "CNN"
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

        assert self.__match_model_type(model_type), f"Unknown model type '{model_type}'."
        self._model_type = self.__match_model_type(model_type)

    @property
    def num_params(self) -> Union[None, Tuple[int, int]]:
        if self._autoencoder:
            return count_model_parameters(self._autoencoder)
        else:
            return None

    def __match_model_type(self, model_type: str) -> Union[str, None]:
        for ky, val in self.__model_types.items():
            if re.match(val, model_type, re.IGNORECASE):
                return ky
        return None

    def build_encoder(self, build_model: bool = True):
        self._encoder_input = Input(shape=(self.sig_len, self.sig_dim))

        # input layer
        encoder = self._encoder_input
        if self._model_type == "FCN":
            encoder = Flatten()(encoder)
        encoder = Dropout(rate=0.1)(encoder)
        print(f"Layer -1: encoder.shape={encoder.shape} (input)")

        # hidden layers
        for i in range(self.n_layers):
            # number of neurons / filter in this layer
            if self._model_type == "FCN":
                n_neurons = 2 ** (self.n_layers + self.__n_filter_init - i - 1)
            else:
                n_neurons = 2 ** (self.__n_filter_init + i)

            if self._model_type == "CNN":
                shape_conv1d_in = encoder.shape
                encoder = Conv1D(n_neurons, self.kernel_sz, activation="relu", padding="same")(encoder)
                print(f"Layer {i}: encoder.shape={shape_conv1d_in} => {encoder.shape} (Conv1D)")

                shape_conv1d_in = encoder.shape
                encoder = MaxPooling1D(self.stride_sz, padding="same")(encoder)
                print(f"Layer {i}: encoder.shape={shape_conv1d_in} => {encoder.shape} (MaxPooling1D)")

            elif self._model_type == "fullyCNN":
                shape_conv1d_in = encoder.shape
                encoder = Conv1D(n_neurons, self.kernel_sz, activation="relu", padding="same")(encoder)
                print(f"Layer {i}: encoder.shape={shape_conv1d_in} => {encoder.shape} (Conv1D)")

                shape_conv1d_in = encoder.shape
                encoder = Conv1D(n_neurons, 1, strides=self.stride_sz, padding="same")(encoder)
                print(f"Layer {i}: encoder.shape={shape_conv1d_in} => {encoder.shape} (Conv1D stride)")

            elif self._model_type == "FCN":
                encoder = Dense(n_neurons, activation="relu")(encoder)
                print(f"Layer {i}: encoder.shape={encoder.shape} (Dense)")

            elif self._model_type == "LSTM":
                pass
            else:
                raise ValueError(f"Unknown model type '{self._model_type}'.")

            if i % 2:
                encoder = Dropout(rate=0.1)(encoder)  # FIXME only for CNNs?
        self.__shape_hidden_layer = encoder.shape[1:]
        # output layer
        encoder = Flatten()(encoder)
        print(f"Layer {i+1}: encoder.shape={encoder.shape} (Flatten)")
        encoder = Dense(self.compressed_dim, activation="relu")(encoder)
        print(f"Layer {i+1}: encoder.shape={encoder.shape} (Dense)")

        if build_model:
            encoder = Model(self._encoder_input, encoder)
            # TODO load weights

        return encoder

    def build_decoder(self, encoder=None):
        # input layer
        if encoder is None:
            decoder_input = Input(shape=self.compressed_dim)
        else:
            decoder_input = encoder

        decoder = Dense(np.prod(self.__shape_hidden_layer))(decoder_input)
        print(f"Layer -1: decoder.shape={decoder.shape} (Dense)")
        decoder = Reshape(self.__shape_hidden_layer)(decoder)
        print(f"Layer -1: decoder.shape={decoder.shape} (Reshape)")

        for i in range(self.n_layers):
            # number of neurons / filter in this layer
            if self._model_type == "FCN":
                n_neurons = 2 ** (self.__n_filter_init + i)
            else:
                n_neurons = 2 ** (self.n_layers + self.__n_filter_init - i - 2)
            activation = "relu"

            if i == self.n_layers - 1:
                if self._model_type == "FCN":
                    n_neurons = self._encoder_input.shape[1]
                else:
                    n_neurons = self.sig_dim

                activation = "sigmoid"

            if self._model_type == "CNN":
                shape_conv1d_in = decoder.shape
                decoder = UpSampling1D(self.stride_sz)(decoder)
                print(f"Layer {i}: decoder.shape={shape_conv1d_in} => {decoder.shape} (UpSampling1D)")

                shape_conv1d_in = decoder.shape
                decoder = Conv1D(n_neurons, self.kernel_sz, activation=activation, padding="same")(decoder)
                print(f"Layer {i}: decoder.shape={shape_conv1d_in} => {decoder.shape} (Conv1D)")

                # FIXME: Transposed convolution
                # decoder = Conv1DTranspose(n_neurons, self.kernel_sz, strides=self.stride_sz,
                #                           activation=activation, padding="same")(decoder)
                # print(f"Layer {i}: decoder.shape={shape_conv1d_in} => {decoder.shape} (Conv1DTranspose)")
            elif self._model_type == "fullyCNN":
                shape_conv1d_in = decoder.shape
                decoder = Conv1DTranspose(decoder.shape[-1], 1, strides=self.stride_sz,
                                          activation=activation, padding="same")(decoder)
                print(f"Layer {i}: decoder.shape={shape_conv1d_in} => {decoder.shape} (Conv1DTranspose)")

                shape_conv1d_in = decoder.shape
                decoder = Conv1D(n_neurons, self.kernel_sz, activation=activation, padding="same")(decoder)
                print(f"Layer {i}: decoder.shape={shape_conv1d_in} => {decoder.shape} (Conv1D)")
            elif self._model_type == "FCN":
                decoder = Dense(n_neurons, activation=activation)(decoder)
                print(f"Layer {i}: decoder.shape={decoder.shape} (Dense)")
            elif self._model_type == "LSTM":
                pass
            else:
                raise ValueError(f"Unknown model type '{self._model_type}'.")

        if encoder is None:
            decoder = Model(decoder_input, decoder)
            # TODO load weights
        return decoder

    def build_autoencoder(self):
        encoder = self.build_encoder(build_model=False)
        decoder = self.build_decoder(encoder)
        autoencoder = Model(self._encoder_input, decoder)
        return autoencoder

    @property
    def autoencoder(self):
        if self._autoencoder is None:
            self._autoencoder = self.build_autoencoder()
            n_trainable, n_non_trainable = self.num_params
            print(f"Build new autoencoder model with {n_trainable}/{n_non_trainable} trainable parameters.")
        return self._autoencoder

    def fit(self, data_tf, epochs: int = 100) -> pd.DataFrame:
        model = self.autoencoder
        model.compile(optimizer="adam",
                      loss='mse',
                      # metrics=['accuracy']
                      )

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
                            shuffle=True,
                            )
        dt = time() - t1
        print(f"Training took {dt: 0.4} seconds.")
        self._autoencoder = model
        return pd.DataFrame(history.history)


if __name__ == "__main__":
    sig_len_max = 128  # FIXME: doesn't work with all sizes
    sig_dimension = 1

    # create toy data
    n_signals = 1000

    data = (np.random.random((n_signals, sig_len_max)) +
            np.arange(sig_len_max) / (sig_len_max / 4) +
            np.asarray(list(np.arange(sig_len_max / 8)) * 8).flatten()) / (sig_len_max / 8) / 1.2
    print(f"data.shape={data.shape}")

    # TODO: make input length variable

    auto = Autoencoder(sig_len=sig_len_max, sig_dim=sig_dimension, n_layers=3, model_type="fullyCNN")

    # print(f"================> # model parameter autoencoder: {count_model_parameters(auto.autoencoder)}")
    print(f"================> # model parameter encoder: {count_model_parameters(auto.build_encoder())}")
    print(f"================> # model parameter decoder: {count_model_parameters(auto.build_decoder())}")

    auto.fit(data, epochs=500)

    sig = data[0, :]
    sig_prd = auto.autoencoder.predict(np.expand_dims(sig, 0))

    fig, axs = plt.subplots(1, 2)
    for i in range(2):
        if i == 0:
            axs[i].plot(sig, color="b")
            axs[i].plot(sig_prd.flatten(), color="r")
        elif i == 1:
            axs[i].plot(sig_prd.flatten() - sig, color="k")
    fig.show()