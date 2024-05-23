from __future__ import print_function
import numpy as np
import sys
from functools import partial

seed = 1337
np.random.seed(seed) 

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, Callback
from keras.utils import to_categorical
from keras.activations import relu
from keras.callbacks import ModelCheckpoint
from quantize.quantized_layers import QuantizedConv2D, QuantizedDense
from quantize.quantized_ops import quantized_relu as quantized_relu_op
from quantize.quantized_ops import quantized_tanh as quantized_tanh_op
import math
from argparse import ArgumentParser
import tensorflow as tf

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


from keras.datasets import cifar10, fashion_mnist, mnist

# Load MNIST data
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Load CIFAR10 data
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

# Load Fashion MNIST data
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

# Preprocess CIFAR10 data
x_train_cifar10 = x_train_cifar10.astype('float32') / 255
x_test_cifar10 = x_test_cifar10.astype('float32') / 255
y_train_cifar10 = to_categorical(y_train_cifar10, 10)
y_test_cifar10 = to_categorical(y_test_cifar10, 10)

# Preprocess Fashion MNIST data
x_train_fashion = x_train_fashion.astype('float32') / 255
x_test_fashion = x_test_fashion.astype('float32') / 255
y_train_fashion = to_categorical(y_train_fashion, 10)
y_test_fashion = to_categorical(y_test_fashion, 10)

# Preprocess MNIST data
x_train_mnist = x_train_mnist.astype('float32') / 255
x_test_mnist = x_test_mnist.astype('float32') / 255
y_train_mnist = to_categorical(y_train_mnist, 10)
y_test_mnist = to_categorical(y_test_mnist, 10)




parser = ArgumentParser()
parser.add_argument("-nb", "--num_bits", dest="num_bits",
                    help="Number of bits to quantize", default = 4)
args = parser.parse_args()


def mnist_process(x):
	for j in range(len(x)):
		x[j] = x[j]*2-1
		if(len(x[j][0]) == 784):
			x[j] = np.reshape(x[j], [-1, 28, 28, 1])
	return x

def cifar10_process(x):
    for j in range(len(x)):
        x[j] = x[j]*2-1
    return x

def fashion_mnist_process(x):
    for j in range(len(x)):
        x[j] = x[j]*2-1
        if(len(x[j][0]) == 784):
            x[j] = np.reshape(x[j], [-1, 28, 28, 1])
    return x


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


# nn
batch_size = 128
epochs = 1000 
channels = 1
img_rows = 28 
img_cols = 28 
filters = 32 
kernel_size = (3, 3)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False
n_bits = int(args.num_bits)
# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)


def create_model(img_rows, img_cols):

    H = 1.
    kernel_lr_multiplier = 'Glorot'


    # BN
    epsilon = 1e-6
    momentum = 0.9

    def add_quant_conv_layer(model, conv_num_filters, conv_kernel_size, conv_strides, mpool_kernel_size, mpool_strides, n_bits):

        model.add(QuantizedConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), input_shape=( img_rows, img_cols, channels),
                            data_format='channels_last', strides=(conv_strides,conv_strides),
                            H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                            padding='valid', use_bias=use_bias, nb = n_bits))
        model.add(MaxPooling2D(pool_size=(mpool_kernel_size, mpool_kernel_size),strides = (mpool_strides,mpool_strides) ,padding='valid' , data_format='channels_last'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1))
        model.add(Activation(partial(quantized_relu_op, nb = n_bits)))
        return model 


    assert n_bits >= 2 , "Numer of bits should be at least 2 and atmost 32"
    assert n_bits <= 32 , "Numer of bits should be at least 2 and atmost 32"

    print("Quantize to bits: ", n_bits)
    print(type(n_bits))

    # -------------Model Architecture-----

    model = Sequential()

    conv_kernel_size = 5
    conv_num_filters = 32
    conv_strides = 2
    mpool_kernel_size = 2
    mpool_strides = 2

    add_quant_conv_layer(model = model, conv_num_filters = conv_num_filters, conv_kernel_size = conv_kernel_size, conv_strides = conv_strides, mpool_kernel_size = mpool_kernel_size, mpool_strides = mpool_strides, n_bits=n_bits)


    conv_kernel_size = 3
    conv_num_filters = 64
    conv_strides = 1
    mpool_kernel_size = 2
    mpool_strides = 2

    add_quant_conv_layer(model = model, conv_num_filters = conv_num_filters, conv_kernel_size = conv_kernel_size, conv_strides = conv_strides, mpool_kernel_size = mpool_kernel_size, mpool_strides = mpool_strides, n_bits=n_bits)

    model.add(Flatten())

    # dense1
    model.add(QuantizedDense(512, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, nb = n_bits))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum ))
    model.add(Activation(partial(quantized_relu_op, nb = n_bits)))
    # dense2
    model.add(QuantizedDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, nb = n_bits))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

    opt = Adam(lr=lr_start) 
    model.compile(loss='squared_hinge', optimizer=opt, metrics=[
            'acc',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr')  # AUC of the PR curve is equivalent to the F1 score
        ])
    model.summary()
    return model

# ---------------------------------



# ------------- MNIST Unpack and Augment Code------------

train_data = x_train_mnist
train_labels = y_train_mnist
test_data = x_test_mnist
test_labels = y_test_mnist

x = [train_data, train_labels, test_data, test_labels]
x_train, y_train, x_test, y_test = mnist_process(x)

print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)

# --------------------------------------------------------



# -------- Train Loop----------------------

model = create_model(28, 28)
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler, ModelCheckpoint('temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True), early_stopping])
score = model.evaluate(x_test, y_test, verbose=1)
print('MNIST Test losss:', score[0])
print('MNIST Test accuracy:', score[1])
print('MNIST Test precision:', score[2])
print('MNIST Test recall:', score[3])
print('MNIST Test F1:', score[4])
# ---------------------------------------


# ------------- CIFAR10 Unpack and Augment Code------------
train_data = x_train_cifar10
train_labels = y_train_cifar10
test_data = x_test_cifar10
test_labels = y_test_cifar10

x = [train_data, train_labels, test_data, test_labels]
x_train, y_train, x_test, y_test = cifar10_process(x)
# remove the the last 3
x_train = x_train[:,:,:,0]
x_test = x_test[:,:,:,0]
x_train = np.reshape(x_train, [-1, 32, 32, 1])
x_test = np.reshape(x_test, [-1, 32, 32, 1])


print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)

# --------------------------------------------------------

# -------- Train Loop----------------------

model = create_model(32, 32)
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler, ModelCheckpoint('temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True), early_stopping])
score = model.evaluate(x_test, y_test, verbose=1)
print('CIFAR10 Test losss:', score[0])
print('CIFAR10 Test accuracy:', score[1])
print('CIFAR10 Test precision:', score[2])
print('CIFAR10 Test recall:', score[3])
print('CIFAR10 Test F1:', score[4])
# ---------------------------------------

# ------------- Fashion MNIST Unpack and Augment Code------------
train_data = x_train_fashion
train_labels = y_train_fashion
test_data = x_test_fashion
test_labels = y_test_fashion

x = [train_data, train_labels, test_data, test_labels]
x_train, y_train, x_test, y_test = fashion_mnist_process(x)

print("X train: ", x_train.shape)
print("Y train: ", y_train.shape)

# --------------------------------------------------------

# -------- Train Loop----------------------

model = create_model(28, 28)
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=[lr_scheduler, ModelCheckpoint('temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True), early_stopping])
score = model.evaluate(x_test, y_test, verbose=1)
print('Fashion MNIST Test losss:', score[0])
print('Fashion MNIST Test accuracy:', score[1])
print('Fashion MNIST Test precision:', score[2])
print('Fashion MNIST Test recall:', score[3])
print('Fashion MNIST Test F1:', score[4])
# ---------------------------------------

