import numpy as np
from keras.datasets import mnist

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers
from sklearn.cluster import KMeans
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib
from keras.utils import to_categorical

import ha_utils.graphic_an_ting as GnT
from data_trawlers.mnist_trawler import MnistTrawler

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    # returns z_mean + rand(z_sd)
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class CONV(object):

    def __init__(self):

        self.n_digits = 10
        # MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train[y_train < self.n_digits]
        y_train = y_train[y_train < self.n_digits]

        x_test = x_test[y_test < self.n_digits]
        y_test = y_test[y_test < self.n_digits]

        image_size = x_train.shape[1]
        self.original_dim = image_size * image_size
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255

        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

        # network parameters
        self.input_shape1 = (image_size * image_size,)
        self.input_shape2 = (image_size, image_size, 1)
        self.intermediate_dim = 512
        self.inputs1 = Input(shape=self.input_shape1, name='encoder_input')
        self.inputs2 = Input(shape=self.input_shape2, name='encoder_input')

    def build_models(self):

        def build_conv():

            x = Conv2D(16, (3, 3), activation='relu', padding='same')(self.inputs2)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            #x = Dropout(0.20)(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            #x = Dropout(0.25)(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Flatten()(x)

            x = Dense(128)(x)
            x = Dense(10, activation='softmax')(x)

            # instantiate encoder model
            conv = Model(self.inputs2, [x], name='conv')
            return conv

        self.conv = build_conv()

        self.conv.summary()

from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, conv):
        self.conv = conv

    def on_epoch_end(self, epoch, logs={}):

        self.conv.conv.save_weights('conv_mnist.h5')

def run():

    weights = False
    epochs = 50
    batch_size = 32

    data_trawler = MnistTrawler()
    dataset_train = data_trawler.get_dataset("label")
    dataset_test = data_trawler.get_dataset("test")

    conv = CONV()
    conv.build_models()
    #vae.vae.load_weights('vae_mlp_mnist.h5')
    conv.conv.compile(loss=categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    print("########")
    print(dataset_train.X_norm.shape)
    print(dataset_train.Y_hot.shape)

    if weights:
        conv.conv.load_weights('conv_mnist.h5')
    else:
        # train the autoencoder
        #conv.conv.fit(conv.x_train, conv.y_train,
        conv.conv.fit(dataset_train.X_norm, dataset_train.Y_hot,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(dataset_test.X_norm, dataset_test.Y_hot),
                    callbacks=[TestCallback(conv)],
                    verbose=2)


    score = conv.conv.evaluate(conv.x_test, conv.y_test, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":

    run()