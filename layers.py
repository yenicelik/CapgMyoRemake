import numpy as np
import tensorflow as tf


def layer_conv(inputs, W, b, is_training):
    """
    :param inputs: [image number, X-axis, Y-axis, Channels of the image]
    :param W:
    :param b:
    :param is_training:
    :return:
    """
    #TODO: Not entirely sure if padding is 'SAME' in this case..
    inputs = tf.nn.conv2d(
                            input=inputs,
                            filter=W,
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

#TODO: Implement locally connected layer!
def layer_local(inputs, W, b, is_training):
    inputs = tf.multiply(inputs, W) + b
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
    )
    inputs = tf.nn.relu(inputs)

    return inputs

