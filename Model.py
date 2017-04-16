from __future__ import print_function
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
    print(type(inputs))
    if verbose:
        print(layername, "\t\t\t\t", str(inputs.get_shape()) )

def init_graph():
    """
    :return:
    """
    W, b, global_step = initialize_parameters()
    X_input, y_input, loss, predict, updateModel, global_step = build_model(W, b, global_step, verbose=True, is_training=True)

    model_dict = {
                    "X_input": X_input,
                    "y_input": y_input,
                    "loss": loss,
                    "predict": predict,
                    "updateModel": updateModel,
                    "globalStepTensor": global_step
    }

    return W, b, model_dict



def initialize_parameters():

    #Global variable to capture the number of steps made
    global_step = tf.Variable(1, trainable=False, name='global_step')

    #TODO: Iteratively check if these are the correct weights!
    Weights = {
                #Is the last number the batch_size? I think it is..
                "W_Conv1": tf.Variable(tf.random_normal([3, 3, 1, 64], 0.00, 0.01), name="W_Conv1"),
                "W_Conv2": tf.Variable(tf.random_normal([3, 3, 64, 64], 0.00, 0.01), name="W_Conv2"), #not sure if we sum over all 64 channels?!
                #TODO: Implement locally connected layer! Currently, we use affine layers instead of locally connected layers!
                #This layer is terribly wrong!
                #"W_Local1": tf.Variable(tf.random_normal([16 * 64 * 8, 512], 0.00, 0.01), name="W_Local1"),
                #"W_Local2": tf.Variable(tf.random_normal([16 * 64 * 8, 512], 0.00, 0.01), name="W_Local2"),
                "W_Affine1": tf.Variable(tf.random_normal([16*64*8, 512], 0.00, 0.01), name="W_Affine1"),
                "W_Affine2": tf.Variable(tf.random_normal([512, 128], 0.00, 0.01), name="W_Affine2"),
                "W_Affine3": tf.Variable(tf.random_normal([128, 12], 0.00, 0.01), name="W_Affine3"),
    }

    #TODO: Do local layers have a bias term?
    Bias = {
                "b_Conv1": tf.Variable(tf.random_normal([1, 16, 8, 64], 0.00, 0.01), name="b_Conv1"), #this is not correct, as it should be a per-filter weight!!!
                "b_Conv2": tf.Variable(tf.random_normal([1, 16, 8, 64], 0.00, 0.01), name="b_Conv2"),
                #"b_Local1": tf.Variable(tf.random_normal([1, 16, 8, 8], 0.00, 0.01), name="b_Local1"),
                #"b_Local2": tf.Variable(tf.random_normal([1, 16, 8, 8], 0.00, 0.01), name="b_Local2"),
                "b_Affine1": tf.Variable(tf.random_normal([1, 512], 0.00, 0.01), name="b_Affine1"),
                "b_Affine2": tf.Variable(tf.random_normal([1, 128], 0.00, 0.01), name="b_Affine2"),
                "b_Affine3": tf.Variable(tf.random_normal([1, 12], 0.00, 0.01), name="b_Affine3")
    }

    return Weights, Bias, global_step




def build_model(W, b, global_step, verbose=True, is_training=False):
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
    #TODO: Input None into the shape; also, input the time-window into the shape!
    X_input = tf.placeholder(shape=[None, 16, 8], dtype=tf.float32)
    if is_training:
        y_input = tf.placeholder(shape=[None, 12], dtype=tf.int8)
    inputs = tf.reshape(X_input, (-1, 16, 8, 1)) #must be a 4D input into the CNN layer
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    print_layershape("Input", inputs, verbose=verbose)



    ## 1. Layer: Conv1 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv1'], b['b_Conv1'], is_training)
    print_layershape("Conv1", inputs, verbose=verbose)

    ## 2. Layer: Conv2 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv2'], b['b_Conv2'], is_training)
    print_layershape("Conv2", inputs, verbose=verbose)



    #TODO: Implement locally connected layer! Currently, we use affine layers instead of locally connected layers!
    # ## 3. Layer: Locally connected 1
    # inputs = layer_affine(inputs, W['W_Local1'], b['b_Local1'], is_training)
    # print_layershape("Local1", inputs, verbose=verbose)
    #
    # ## 4. Layer: Locally connected 2
    # inputs = layer_affine(inputs, W['W_Local2'], b['b_Local2'], is_training)
    # inputs = tf.nn.dropout(inputs, 0.5)
    # print_layershape("Local2", inputs, verbose=verbose)



    ## 5. Layer: Affine 1 (512 units)
    inputs = tf.reshape(inputs, (-1, 16 * 8 * 64))
    inputs = layer_affine(inputs, W['W_Affine1'], b['b_Affine1'], is_training)
    inputs = tf.nn.dropout(inputs, 0.5)
    print_layershape("Affine1", inputs, verbose=verbose)

    ## 6. Layer: Affine 2 (512 units)
    inputs = layer_affine(inputs, W['W_Affine2'], b['b_Affine2'], is_training)
    inputs = tf.nn.dropout(inputs, 0.5)
    print_layershape("Affine2", inputs, verbose=verbose)

    ## 7. Layer: Affine 3 (128 units)
    inputs = layer_affine(inputs, W['W_Affine3'], b['b_Affine3'], is_training)
    print_layershape("Affine3", inputs, verbose=verbose)


    ## 8. Layer: Softmax, or loss otherwise
    predict = tf.nn.softmax(inputs) #should be an argmax, or should this even go through

    loss = None
    trainer = None
    updateModel = None

    if is_training:
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=predict)
        #TODO: Parameterize the learning rate!
        #TODO: Choose the optimal algorithm to find the optimal function model
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1) #think about using Adam
        updateModel = trainer.minimize(loss)

    #To see how many steps we've been through
    increment_global_step = tf.assign(global_step, global_step+1)

    #We must return 'references' to the individual objects
    return X_input, y_input, loss, predict, updateModel, increment_global_step

