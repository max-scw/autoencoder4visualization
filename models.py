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
    Dropout
    )
from keras.optimizers import Adam
#from keras.objectives import categorical_crossentropy


if __name__ == "__main__":
    # create toy data
    sig_len = np.random.randint(low=90, high=110)
    sig_dim = 2
    sig = np.random.random(size=(sig_len, sig_dim))

#tf.keras.layers.Conv1D(
#    filters,
#    kernel_size,
#    strides=1,
#    padding="valid",
#    data_format="channels_last",
#    dilation_rate=1,
#    groups=1,
#    activation=None,
#    use_bias=True,
#    kernel_initializer="glorot_uniform",
#    bias_initializer="zeros",
#    kernel_regularizer=None,
#    bias_regularizer=None,
#    activity_regularizer=None,
#    kernel_constraint=None,
#    bias_constraint=None,
#    **kwargs
#)




    # around 600K parameters
    n_filters = 32
    sz_kernel = 3

    n_stride = 2

    sig_input = Input(shape=(sig_len, sig_dim)) # zeropadd input?

    x = Conv1D(16, sz_kernel, activation='relu', padding='same')(sig_input)
    x = MaxPooling1D(n_stride, padding='same')(x)
    x = Conv1D(8, sz_kernel, activation='relu', padding='same')(x)
    x = MaxPooling1D(n_stride, padding='same')(x)
    x = Conv1D(8, sz_kernel, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(n_stride, padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv1D(8, sz_kernel, activation='relu', padding='same')(encoded)
    x = UpSampling1D(n_stride)(x)
    x = Conv1D(8, sz_kernel, activation='relu', padding='same')(x)
    x = UpSampling1D(n_stride)(x)
    x = Conv1D(16, sz_kernel, activation='relu')(x)
    x = UpSampling1D(n_stride)(x)
    decoded = Conv1D(1, sz_kernel, activation='sigmoid', padding='same')(x)

    autoencoder = Model(sig_input, decoded)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')



