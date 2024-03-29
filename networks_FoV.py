import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np

def lrelu(x):
    # leaky rely
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

def apply_conv(x, filters=32, kernel_size=3, he_init=True, factor=2.0, cst_ini=False):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=factor)
    else:
        initializer = tf.contrib.layers.xavier_initializer()

    if cst_ini:
        initializer = tf.constant_initializer(value=1e-2)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

    return tf.layers.conv3d(x,
                            filters=filters, kernel_size=kernel_size,
                            padding='SAME',
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer)
def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.leaky_relu(x)

def meanpool(x):
    with tf.name_scope('meanpool'):
        return tf.layers.average_pooling3d(
            x,
            2,
            2,
            padding='same',
            data_format='channels_last',
        )

def resblock(x, filters, he_init=False, factor=2.0):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        #        update = apply_conv(activation(x), filters=filters, he_init=he_init, factor=factor)
        #        update = apply_conv(activation(update), filters=filters, he_init=he_init, factor=factor)
        update = apply_conv(activation(x), filters=filters)
        update = apply_conv(activation(update), filters=filters)

        #        skip = apply_conv(x, filters=filters, kernel_size=1, he_init=he_init, factor=factor, cst_ini=True)
        skip = apply_conv(x, filters=filters, kernel_size=1, he_init=he_init, factor=factor)

        return skip + update


class network(ABC):

    # The FoV k and the maximal size of a patch fitting on a GPU m
    # Override with network specifics in the subclass
    m = np.array((0,0,0))
    k = np.array((0,0,0))

    # Method defining the neural network architecture, returns computation result. Use reuse=tf.AUTO_REUSE.
    @abstractmethod
    def net(self, input):
        pass

    
class ResNetL2_local(network):
        # FoV 4*2+2*2+2+2+1 = 17

    m = np.array((64,64,64))
    k = np.array((17,17,17))

    def net(self, x_in):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with tf.name_scope('pre_process'):
                x_ini = apply_conv(x_in, filters=8, kernel_size=3)

            with tf.name_scope('x0'):
                x0 = resblock(x_ini, 16)

            with tf.name_scope('x1'):
                x1 = resblock(x0, 32)

            with tf.name_scope('x2'):
                x2 = resblock(meanpool(x1), filters=32)  # 2

            with tf.name_scope('x3'):
                x3 = resblock(meanpool(x2), filters=64)  # 4

            with tf.name_scope('Global_averaging'):
                x6 = tf.reduce_mean(x3, axis=(1,2,3))

            with tf.name_scope('flat'):
                flat = tf.layers.dense(x6, 128, activation=activation)
                flat = tf.layers.dense(flat, 1)

            with tf.name_scope('l2_norm'):
                base_case = tf.sqrt(tf.reduce_sum(x_in ** 2, axis=[1, 2, 3, 4]))

            with tf.name_scope('return'):
                return base_case + flat
    

class ResNetL2(network):

    # FoV 16*2+8*2+4*2+2*2+2+2+1 = 65

    m = np.array((64,64,64))
    k = np.array((65,65,65))

    def net(self, x_in):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with tf.name_scope('pre_process'):
                x_ini = apply_conv(x_in, filters=8, kernel_size=3)

            with tf.name_scope('x0'):
                x0 = resblock(x_ini, 16)

            with tf.name_scope('x1'):
                x1 = resblock(x0, 32)

            with tf.name_scope('x2'):
                x2 = resblock(meanpool(x1), filters=32)  # 2

            with tf.name_scope('x3'):
                x3 = resblock(meanpool(x2), filters=64)  # 4

            with tf.name_scope('x4'):
                x4 = resblock(meanpool(x3), filters=128)  # 8

            with tf.name_scope('x5'):
                x5 = resblock(meanpool(x4), filters=128)  # 16

            with tf.name_scope('Global_averaging'):
                x6 = tf.reduce_mean(x5, axis=(1,2,3))

            with tf.name_scope('flat'):
                flat = tf.layers.dense(x6, 128, activation=activation)
                flat = tf.layers.dense(flat, 1)

            with tf.name_scope('l2_norm'):
                base_case = tf.sqrt(tf.reduce_sum(x_in ** 2, axis=[1, 2, 3, 4]))

            with tf.name_scope('return'):
                return base_case + flat