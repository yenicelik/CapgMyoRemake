import tensorflow as tf
import numpy as np

def print_layershape(layername, inputs, verbose=False):
    """
    :param layername: The name of the layer that we currently print
    :param inptus: The layer object, such that we can extract the shape
    :param verbose: Whether we want to print it or not (for simplicity of use)
    :return: -
    """
    if verbose:
        print(layername, "\t\t\t\t", str(inputs.get_shape()) )

def init_graph():
    """
    :return:
    """
    W, b = initialize_parameters()
    build_forward_model(W, b, )
    build_loss_model()


def initialize_parameters():


    #TODO: Iteratively check if these are the correct weights!
    Weights = {
                "W_Conv1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Conv1"),
                "W_Conv2": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Conv2"),
                "W_Local1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Local1"),
                "W_Local2": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Local2"),
                "W_Affine1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Affine1"),
                "W_Affine2": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Affine2"),
                "W_Affine3": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="W_Affine3"),
    }

    #TODO: Do local layers have a bias term?
    Bias = {
                "b_Conv1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Conv1"),
                "b_Conv2": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Conv2"),
                "b_Local1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Local1"),
                "b_Local2": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Local2"),
                "b_Affine1": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Affine1"),
                "b_Affine2": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Affine2"),
                "b_Affine3": tf.Variable(tf.random_normal([8, 8, 1, 16], 0.00, 0.01), name="b_Affine3")
    }

    return Weights, Bias

def build_forward_model(W, b, verbose=True, is_training=False):
    """
    :param W:
    :param b:
    :param verbose:
    :param is_training:
    :return:

    The forward model looks as follows:
    0. Input: (16, 8)
        BatchNorm
    1. Convolutional 1: 64 filters, each 3x3 wide, with stride 1
        BatchNorm
        ReLU
    2. Convolutional 2: 64 filters, each 3x3 wide, with stride 1
        BatchNorm
        ReLU
    3. Local 1: 64 non-overlapping filters, 1x1 wide
        BatchNorm
        ReLU
    4. Local 2: 64 non-overlapping filters, 1x1 wide
        BatchNorm
        ReLU
        Dropout(0.5)
    5: Affine 1: 512 units
        BatchNorm
        ReLU
        Dropout(0.5)
    6: Affine 2: 512 units
        BatchNorm
        ReLU
        Dropout(0.5)
    7: Affine 3: 128 units
        BatchNorm
        ReLU
    8: Affine 4: Connecting to number of gestures
        BatchNorm
        ReLU
    9: Softmax

    """

    ##Input
    input = tf.placeholder(shape=[16, 8], dtype=tf.float32)
    inputs = tf.reshape(input, [1, 16, 8, 1]) #must be a 4D input into the CNN layer
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )

    print_layershape("Input", input, verbose=verbose)

    ###########################
    ##Conv1 (64, stride=1, 3x3)
    #TODO: Not entirely sure if padding is 'SAME' in this case..
    inputs = tf.nn.conv2d(
                            inputs,
                            W['W_Conv1'],
                            filter=[1, 3, 3, 1],
                            strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
    inputs += b['b_Conv1']
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    ##---------- PRINT
    print_layershape("Conv1", inputs, verbose=verbose)

    ###########################
    ##Conv2 (64, stride=1, 3x3)
    inputs = tf.nn.conv2d(
                            input,
                            W['W_Conv2'],
                            filter=[1, 3, 3, 1],
                            strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
    inputs += b['b_Conv2']
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    ##---------- PRINT
    print_layershape("Conv2", inputs, verbose=verbose)

    ###########################
    ##Locally connected 1 (there are no locally connected layers, so we must use a work-around with tf.batch_matrix_band_part(input, num_lower, num_upper, name=None)
    inputs = tf.matmul(inputs, W['W_Local1']) + b['b_Local1']
    inputs = tf.batch_matrix_band_part(input, 1, 1)
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)

    ##---------- PRINT
    print_layershape("Local1", inputs, verbose=verbose)

    ###########################
    ##Locally connected 2
    inputs = tf.matmul(inputs, W['W_Local2']) + b['b_Local2']
    inputs = tf.batch_matrix_band_part(inputs, 1, 1)
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.dropout(inputs, 0.5)

    ##---------- PRINT
    print_layershape("Local2", inputs, verbose=verbose)

    ###########################
    ##Affine 1 (512 units)
    inputs = tf.matmul(inputs, W['W_Affine1']) + b['b_Affine1']
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.dropout(inputs, 0.5)

    ##---------- PRINT
    print_layershape("Affine1", inputs, verbose=verbose)

    ###########################
    ##Affine 2 (512 units)
    inputs = tf.matmul(inputs, W['W_Affine2']) + b['b_Affine2']
    tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.dropout(inputs, 0.5)

    ##---------- PRINT
    print_layershape("Affine2", inputs, verbose=verbose)

    ###########################
    ##Affine 3 (128 units)
    inputs = tf.matmul(inputs, W['W_Affine3']) + b['b_Affine3']
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    logits = tf.nn.relu(inputs)

    ##---------- PRINT
    print_layershape("Affine3", inputs, verbose=verbose)

    ###########################
    ##Softmax (finally the output)
    predict = tf.nn.softmax(inputs) if not is_training else None

    return input, logits, predict


def build_loss_model(W, b, logits, verbose=True, is_training=False):
    """
    :param W:
    :param b:
    :param verbose:
    :return:
    Loss: Softmax Cross Entropy (not sure what loss function they used, but I assume this is close if they use a softmax function at the end)
    Trainer: GradientDescentOptimizer with lr=0.1
    updateModel: Minimize Loss
    """
    loss = tf.nn.softmax_cross_entropy(None, None)
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    return loss, trainer, updateModel


