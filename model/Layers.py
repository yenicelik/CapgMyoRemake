import tensorflow as tf


def layer_conv(inputs, W, b, is_training):
    """
    Implements a convolutional layer with batch-normalization and ReLU as a non-linear activation function.
    :param inputs: [image number, X-axis, Y-axis, Channels of the image]
    :param W: Convolutional feature/weight tensor
    :param b: Convolutional bias tensor
    :param is_training: Whether the current session is used for training (True) or for testing (False)
    :return: A tensorflow layer object
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
    """
    Implements a fully connects /affine layer with batch-normalization and ReLU as a non-linear activation function.
    :param inputs: [image number, -1]
    :param W: Feature/weight tensor
    :param b: Bias tensor
    :param is_training: Whether the current session is used for training (True) or for testing (False)
    :return: A tensorflow layer object
    """
    inputs = tf.matmul(inputs, W) + b
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    return inputs

def layer_local(inputs, W, b, is_training):
    """
    Implements a locally connected layer with convolutional size of 1x1. This is equivalent to a component-wise multiplication. This is with batch-normalization and ReLU as a non-linear activation function.
    :param inputs: [image number, -1]
    :param W: Feature/weight tensor
    :param b: Bias tensor
    :param is_training: Whether the current session is used for training (True) or for testing (False)
    :return: A tensorflow layer object
    """

    inputs = tf.multiply(inputs, W) + b
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
    )
    inputs = tf.nn.relu(inputs)

    return inputs

