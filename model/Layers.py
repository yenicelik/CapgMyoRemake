import tensorflow as tf


def layer_conv(inputs, W, b, is_training):
    inputs = tf.nn.conv2d(
                            input=inputs,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
    inputs += b
    inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=0.9)
    inputs = tf.nn.relu(inputs)
    return inputs

def layer_affine(inputs, W, b, is_training):
    inputs = tf.matmul(inputs, W) + b
    inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=0.9)
    inputs = tf.nn.relu(inputs)
    return inputs

#TODO: write the local layers correctly
def layer_local(inputs, W, b, is_training):
    inputs = tf.multiply(inputs, W) + b
    inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=0.9)
    inputs = tf.nn.relu(inputs)
    return inputs