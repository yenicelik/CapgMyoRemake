import tensorflow as tf
import numpy as np
from Layers import *

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
    build_model(W, b, )



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




def build_model(W, b, verbose=True, is_training=False):
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

    ## 0.Layer: Input
    input = tf.placeholder(shape=[16, 8], dtype=tf.float32)
    inputs = tf.reshape(input, [1, 16, 8, 1]) #must be a 4D input into the CNN layer
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    print_layershape("Input", input, verbose=verbose)



    ## 1. Layer: Conv1 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv1'], b['b_Conv1'], is_training)
    print_layershape("Conv1", inputs, verbose=verbose)

    ## 2. Layer: Conv2 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv2'], b['b_Conv2'], is_training)
    print_layershape("Conv2", inputs, verbose=verbose)



    ## 3. Layer: Locally connected 1
    inputs = layer_locally_connected(inputs, W['W_Local1'], b['b_Local1'], is_training)
    print_layershape("Local1", inputs, verbose=verbose)

    ## 4. Layer: Locally connected 2
    layer_locally_connected(inputs, W['W_Local2'], b['b_Local2'], is_training)
    inputs = tf.nn.dropout(inputs, 0.5)
    print_layershape("Local2", inputs, verbose=verbose)



    ## 5. Layer: Affine 1 (512 units)
    inputs = layer_affine(inputs, W['W_Affine1'], b['b_Affine1'], is_training)
    inputs = tf.nn.dropout(inputs, 0.5)
    print_layershape("Affine1", inputs, verbose=verbose)

    ## 6. Layer: Affine 2 (512 units)
    inputs = layer_affine(inputs, W['W_Affine2'], b['b_Affine2'])
    inputs = tf.nn.dropout(inputs, 0.5)
    print_layershape("Affine2", inputs, verbose=verbose)

    ## 7. Layer: Affine 3 (128 units)
    inputs = layer_affine(inputs, W['W_Affine3'], b['b_Affine3'])
    print_layershape("Affine3", inputs, verbose=verbose)


    ## 8. Layer: Softmax, or loss otherwise
    #Depending on train/evaluate
    predict = tf.nn.softmax(inputs) if not is_training else None #We can calculate the loss directly if we only train

    #Depending on train/evaluate
    loss = None
    #loss = tf.nn.softmax_cross_entropy_with_logits(None, None)

    #TODO: Parameterize the learning rate!
    #TODO: Choose the optimal algorithm to find the optimal function model
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)


    #We must return 'references' to the individual objects
    return input, loss, predict, trainer, updateModel

