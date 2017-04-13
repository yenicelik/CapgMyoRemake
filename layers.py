import numpy as np
import tensorflow as tf


def layer_conv(inputs, W, b, is_training):
    """
    :param inputs:
    :param W:
    :param b:
    :param is_training:
    :return:
    """
    #TODO: Not entirely sure if padding is 'SAME' in this case..
    inputs = tf.nn.conv2d(
                            inputs,
                            W,
                            filter=[1, 3, 3, 1],
                            strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
    inputs += b
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    return inputs


def layer_locally_connected(inputs, W, b, is_training):
    """
    There are no locally connected layers in tf, so we must use a work-aroung with tf.batch_matrix_band_part(input, num_lower, num_upper)
    :param inputs:
    :param W:
    :param b:
    :param is_training:
    :return:
    """
    inputs = tf.matmul(inputs, W) + b
    inputs = tf.batch_matrix_band_part(input, 1, 1)
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    return inputs


def layer_affine(inputs, W, b, is_training):
    inputs = tf.matmul(inputs, W) + b
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    return inputs