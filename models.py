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

    sig_input = Input(shape=(None, sig_dim)) # zeropadd input?

    enc_conv1 = Conv1D(16, sz_kernel, activation='relu', padding='same')(sig_input)
    enc_pool1 = MaxPooling1D(n_stride, padding='same')(enc_conv1)

    enc_conv2 = Conv1D(8, sz_kernel, activation='relu', padding='same')(enc_pool1)
    enc_pool2 = MaxPooling1D(n_stride, padding='same')(enc_conv2)

    enc_conv3 = Conv1D(8, sz_kernel, activation='relu', padding='same')(enc_pool2)
    #encoded = MaxPooling1D(n_stride, padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    enc_glob = GlobalAveragePooling1D()(enc_conv3)
    enc_flat = Flatten()(enc_glob)
    encoded = Dense(2, activation="relu")(enc_flat)
    # at this point the representation is (2,) i.e. 2D
    
    dec_glob = Dense(enc_glob.shape[-1] * sig_dim)(encoded)
    dec_flat = Reshape((enc_glob.shape[-1], sig_dim))(dec_glob)

    dec_conv3 = Conv1D(enc_conv3.shape[-1], sz_kernel, activation='relu', padding='same')(dec_flat)
    dec_pool3 = UpSampling1D(n_stride)(dec_conv3)

    dec_conv2 = Conv1D(enc_conv2.shape[-1], sz_kernel, activation='relu', padding='same')(dec_pool3)
    dec_pool2 = UpSampling1D(n_stride)(dec_conv2)

    dec_conv1 = Conv1D(enc_conv1.shape[-1], sz_kernel, activation='relu')(dec_pool2)
    dec_pool1 = UpSampling1D(n_stride)(dec_conv1)
    decoded = Conv1D(1, sz_kernel, activation='sigmoid', padding='same')(dec_pool1)

    autoencoder = Model(sig_input, decoded)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')



