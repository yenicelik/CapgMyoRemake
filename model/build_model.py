from __future__ import print_function

from model.layers import *
import tensorflow as tf
from LocallyConnected import LocallyConnected_1x1

import sys
import logging
logging = logging.getLogger(__name__)


#TODO: apply weight decay with 0.0001
def init_graph():

    ###############################
    # VARIABLES
    ###############################
    global_step = tf.get_variable("global_step", shape=[], trainable=False, initializer=tf.constant_initializer(1), dtype=tf.int64)

    W = {
        #Is the last number the batch_size? I think it is..
        "W_Conv1": tf.get_variable("W_Conv1", shape=[3, 3, 1, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Conv2": tf.get_variable("W_Conv2", shape=[3, 3, 64, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine1": tf.get_variable("W_Affine1", shape=[16 * 8 * 64, 512],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine2": tf.get_variable("W_Affine2", shape=[512, 512],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine3": tf.get_variable("W_Affine3", shape=[512, 128],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine4": tf.get_variable("W_Affine4", shape=[128, 10],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        )
    }

    b = {
        "b_Conv1": tf.get_variable("b_Conv1", shape=[1, 16, 8, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Conv2": tf.get_variable("b_Conv2", shape=[1, 16, 8, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine1": tf.get_variable("b_Affine1", shape=[1, 512],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine2": tf.get_variable("b_Affine2", shape=[1, 512],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine3": tf.get_variable("b_Affine3", shape=[1, 128],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine4": tf.get_variable("b_Affine4", shape=[1, 10],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        )
    }

    ###############################
    # INPUTS
    ###############################
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    X_input = tf.placeholder(shape=[None, 16, 8], dtype=tf.float32, name="X_input")
    y_input = tf.placeholder(shape=[None, 10], dtype=tf.int8, name="y_input")

    ###############################
    # HIDDEN LAYERS
    ###############################
    ## 0. Layer: BatchNorm after input
    inputs = tf.reshape(X_input, (-1, 16, 8, 1)) #must be a 4D input into the CNN layer
    inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=0.9)
    print_layershape("Input", inputs)

    ## 1. Layer: Conv1 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv1'], b['b_Conv1'], is_training)
    print_layershape("Conv1", inputs)

    ## 2. Layer: Conv2 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv2'], b['b_Conv2'], is_training)
    print_layershape("Conv2", inputs)


    ## 3. Layer: Locally connected 1
    inputs = LocallyConnected_1x1(
                    inputs=inputs,
                    layerName="Local1",
                    nInputPlane=64,
                    nFilters=64,
                    is_training=is_training,
                    iW=8,
                    iH=16
    ).build_operation(inputs, is_training)
    print_layershape("Local1", inputs)

    ## 4. Layer: Locally connected 2
    inputs = LocallyConnected_1x1(
                    inputs=inputs,
                    layerName="Local2",
                    nInputPlane=64,
                    nFilters=64,
                    is_training=is_training,
                    iW=8,
                    iH=16
    ).build_operation(inputs, is_training)
    inputs = tf.nn.dropout(inputs, keep_prob)
    print_layershape("Local2", inputs)


    ## 5. Layer: Affine 1 (512 units)
    inputs = tf.reshape(inputs, (-1, 16 * 8 * 64)) #Need to reshape here
    inputs = layer_affine(inputs, W['W_Affine1'], b['b_Affine1'], is_training)
    inputs = tf.nn.dropout(inputs, keep_prob)
    print_layershape("Affine1", inputs)

    ## 6. Layer: Affine 2 (512 units)
    inputs = layer_affine(inputs, W['W_Affine2'], b['b_Affine2'], is_training)
    inputs = tf.nn.dropout(inputs, keep_prob)
    print_layershape("Affine2", inputs)

    ## 7. Layer: Affine 3 (128 units)
    inputs = layer_affine(inputs, W['W_Affine3'], b['b_Affine3'], is_training)
    print_layershape("Affine3", inputs)


    ## 8. Layer: Affine 4 and softmax
    logits = tf.matmul(inputs, W['W_Affine4']) + b['b_Affine4']
    print_layershape("Affine4", logits)
    predict = tf.nn.softmax(logits) #should be an argmax, or should this even go through

    ###############################
    # LOSS FUNCTIONS AND OPTIMIZERS
    ###############################
    #Weight decay
    vars   = tf.trainable_variables()
    wdecay = 0.0001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in vars if v.name[0] == 'W'])


    ## Output: Loss functions and model trainers
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=y_input, logits=logits) + wdecay)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        trainer = tf.train.AdamOptimizer() #tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #tf.train.AdamOptimizer()
        updateModel = trainer.minimize(loss, global_step=global_step)

    ## Accuracy
    correct_pred = tf.equal(tf.argmax(y_input, 1), tf.argmax(predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

    #Compiling all into a dictionary
    model = {
        "X": X_input,
        "y": y_input,
        "lr": learning_rate,
        "is_training": is_training,
        "loss": loss,
        "predict": predict,
        "updateModel": updateModel,
        "keep_prob": keep_prob,
        "trainer": trainer,
        "accuracy": accuracy,
    }

    return model


def print_layershape(layername, inputs):
    print("{}:\t\t\t {}".format(layername, str(inputs.get_shape()) ))
    logging.info("{} \t\t\t {}".format(layername, str(inputs.get_shape())))

