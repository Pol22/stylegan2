import numpy as np
import tensorflow as tf


def get_weight(shape, gain=1.0, use_wscale=True, lrmul=1.0):
    fan_in = tf.reduce_prod(shape[:-1])
    he_std = gain / tf.sqrt(tf.cast(fan_in, tf.float32)) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.random.normal(shape, mean=0, stddev=init_std)
    return tf.Variable(init) * runtime_coef


def modConv2D(x, y, filters, kernel_size=3, demod=True):
    in_shape = tf.shape(x)
    w_shape = (kernel_size, kernel_size, in_shape[-1], filters)
    w = get_weight(w_shape)
    w = w[np.newaxis] # [BkkIO]

    # Modulation through input's channels
    y = tf.keras.layers.Flatten()(y)
    s = 1 + tf.keras.layers.Dense(in_shape[-1], activation=None)(y) # [BI]
    w *= s[:, np.newaxis, np.newaxis, :, np.newaxis] # [BkkIO]

    p = tf.keras.layers.Dense(in_shape[-1], activation=None)(y)
    w += p[:, np.newaxis, np.newaxis, :, np.newaxis]

    if demod: # through output's channels
        d = tf.math.rsqrt(tf.reduce_mean(tf.square(w), axis=(1,2,3)) + 1e-8) # [BO]
        w *= d[:, np.newaxis, np.newaxis, np.newaxis, :]

    # Fuse batch dimention into output's channels
    x = tf.reshape(
        tf.transpose(x, (1, 2, 3, 0)), # [HWIB]
        (1, in_shape[1], in_shape[2], -1)
    ) # [1 H W IB]
    # TODO check it
    # x = tf.reshape(
    #     tf.transpose(x, (1, 2, 0, 3)), # [HWBI]
    #     (1, in_shape[1], in_shape[2], -1)
    # ) # [1 H W BI]
    w = tf.reshape(
        tf.transpose(w, (1, 2, 3, 0, 4)), # [kkIBO]
        (kernel_size, kernel_size, in_shape[-1], -1)
    ) # [k k I BO]

    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    x = tf.reshape(x, (in_shape[1], in_shape[2], -1, filters))
    x = tf.transpose(x, (2, 0, 1, 3))

    return x
