from __future__ import print_function

from model.Layers import *
from datahandler.DataLoader import DataLoader

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)


def print_layershape(layername, inputs):
    print("\t\t\t\t".format(layername, str(inputs.get_shape()) ))
    logging.info("{} \t\t\t\t {}".format(layername, str(inputs.get_shape())))

#TODO: this one-hot dimension should maybe be a global variable!
#TODO: have a more robust load for the individual weights. That means, have a loader object that detects the variables by names, and individually puts them back. if no variable is found, initialize the variable from scratch
def init_graph():
    """
    :param is_training: Whether the session we're running is a training session or not.
    :return: The initized weights-dictionary, bias-dictionary and a model-dictionary that captures all input and output of the created graph.
    """
    logging.debug("-> {} function".format(init_graph.__name__))
    W, b, global_step = initialize_parameters()

    (X_input,
     y_input,
     loss,
     predict,
     updateModel,
     global_step,
     keep_prob,
     learning_rate,
     is_training) = build_model(W, b, global_step)

    model_dict = {
                    "X_input": X_input,
                    "y_input": y_input,
                    "loss": loss,
                    "predict": predict,
                    "updateModel": updateModel,
                    "globalStepTensor": global_step,
                    "keepProb": keep_prob,
                    "learningRate": learning_rate,
                    "isTraining": is_training
    }

    logging.info("Model dictionary looks like: {}".format(model_dict))
    logging.debug("<- {} function".format(init_graph.__name__))
    return W, b, model_dict


def initialize_parameters():
    logging.debug("-> {} function".format(initialize_parameters.__name__))

    #Global variable to capture the number of steps made
    global_step = tf.Variable(1, trainable=False, name='global_step')

    #TODO FUTURE: Time-video as input, for possible Seq2Seq model
    Weights = {
                #Is the last number the batch_size? I think it is..
                "W_Conv1": tf.Variable(tf.random_normal([3, 3, 1, 64], 0.00, 0.01), name="W_Conv1"),
                "W_Conv2": tf.Variable(tf.random_normal([3, 3, 64, 64], 0.00, 0.01), name="W_Conv2"),
                #not sure if we sum over all 64 channels?!
                "W_Local1": tf.Variable(tf.random_normal([1, 16 * 8 * 64], 0.00, 0.01), name="W_Local1"),
                "W_Local2": tf.Variable(tf.random_normal([1, 16 * 8 * 64], 0.00, 0.01), name="W_Local2"),
                "W_Affine1": tf.Variable(tf.random_normal([16*8*64, 512], 0.00, 0.01), name="W_Affine1"),
                "W_Affine2": tf.Variable(tf.random_normal([512, 128], 0.00, 0.01), name="W_Affine2"),
                "W_Affine3": tf.Variable(tf.random_normal([128, 10], 0.00, 0.01), name="W_Affine3"),
    }

    #TODO: Do local layers have a bias term?
    Bias = {
                "b_Conv1": tf.Variable(tf.random_normal([1, 16, 8, 64], 0.00, 0.01), name="b_Conv1"),
                "b_Conv2": tf.Variable(tf.random_normal([1, 16, 8, 64], 0.00, 0.01), name="b_Conv2"),
                "b_Local1": tf.Variable(tf.random_normal([1, 8192], 0.00, 0.01), name="b_Local1"),
                "b_Local2": tf.Variable(tf.random_normal([1, 8192], 0.00, 0.01), name="b_Local2"),
                "b_Affine1": tf.Variable(tf.random_normal([1, 512], 0.00, 0.01), name="b_Affine1"),
                "b_Affine2": tf.Variable(tf.random_normal([1, 128], 0.00, 0.01), name="b_Affine2"),
                "b_Affine3": tf.Variable(tf.random_normal([1, 10], 0.00, 0.01), name="b_Affine3")
    }

    logging.info("Weights are of shape: {}".format([(key, str(w.get_shape())) for key, w in Weights.iteritems()]))
    logging.info("bias are of shape: {}".format([(key, str(b.get_shape())) for key, b in Bias.iteritems()]))

    logging.debug("<- {} function".format(initialize_parameters.__name__))
    return Weights, Bias, global_step


def build_model(W, b, global_step):
    """
    :param W: The weight dictionary. All dimensions of all weights must match.
    :param b: The bias dictionary. All dimensions of all weights must match.
    :param verbose: Whether we want to print out the model-architecture during initialization.
    :param is_training: Whether this is a training or a testing session. I am pretty sure we should move this as a tensorflow variable now.
    :return: A reference to X_input, y_input, loss, predict, updateModel, increment_global_step, keep_prob, each being a reference to the respective layer created within the layer (usually a placeholder, loss/predict function or optimizer)
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
    logging.debug("-> {} function".format(build_model.__name__))
    #Initialize variables here.
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    ## 0.Layer: Input
    X_input = tf.placeholder(shape=[None, 16, 8], dtype=tf.float32)
    y_input = tf.placeholder(shape=[None, 10], dtype=tf.int8)
    print_layershape("X_input", X_input)
    print_layershape("y_input", y_input)

    inputs = tf.reshape(X_input, (-1, 16, 8, 1)) #must be a 4D input into the CNN layer
    inputs = tf.contrib.layers.batch_norm(
                            inputs,
                            center=False,
                            scale=False,
                            is_training=is_training
                        )
    print_layershape("Input", inputs)

    ## 1. Layer: Conv1 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv1'], b['b_Conv1'], is_training)
    print_layershape("Conv1", inputs)

    ## 2. Layer: Conv2 (64, stride=1, 3x3)
    inputs = layer_conv(inputs, W['W_Conv2'], b['b_Conv2'], is_training)
    print_layershape("Conv2", inputs)


    # ## 3. Layer: Locally connected 1
    inputs = tf.reshape(inputs, (-1, 16 * 8 * 64))
    inputs = layer_local(inputs, W['W_Local1'], b['b_Local1'], is_training)
    print_layershape("Local1", inputs)

    # ## 4. Layer: Locally connected 2
    inputs = layer_local(inputs, W['W_Local2'], b['b_Local2'], is_training)
    inputs = tf.nn.dropout(inputs, keep_prob)
    print_layershape("Local2", inputs)



    ## 5. Layer: Affine 1 (512 units)
    inputs = layer_affine(inputs, W['W_Affine1'], b['b_Affine1'], is_training)
    inputs = tf.nn.dropout(inputs, keep_prob)
    print_layershape("Affine1", inputs)

    ## 6. Layer: Affine 2 (512 units)
    inputs = layer_affine(inputs, W['W_Affine2'], b['b_Affine2'], is_training)
    inputs = tf.nn.dropout(inputs, 0.5)
    print_layershape("Affine2", inputs)

    ## 7. Layer: Affine 3 (128 units)
    inputs = layer_affine(inputs, W['W_Affine3'], b['b_Affine3'], is_training)
    print_layershape("Affine3", inputs)


    ## 8. Layer: Softmax, or loss otherwise
    predict = tf.nn.softmax(inputs) #should be an argmax, or should this even go through


    ## Output: Loss functions and model trainers
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=predict)
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    updateModel = trainer.minimize(loss)

    #To see how many steps we've been through (this will be saved in-between sessions)
    increment_global_step = tf.assign(global_step, global_step+1)

    #We must return 'references' to the individual objects
    logging.debug("<- {} function".format(build_model.__name__))
    return X_input, y_input, loss, predict, updateModel, increment_global_step, keep_prob, learning_rate, is_training


if __name__ == "__main__":

    dataLoader = DataLoader("../datahandler/Datasets/Preprocessed/DB-a")
    X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    W, b, model_dict = init_graph()

    parameter = {
        'LEARNING_RATE': 500,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000
    }

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        logging.debug("Entering session")
        sess.run(init_op)
        logging.debug("Initial_op success. Now running session on data")
        sess.run(
            # Describe what we want out of the model
            [
                model_dict['loss'],
                model_dict['predict'],
                model_dict['updateModel'],
                model_dict['globalStepTensor']

            ],
            # Describe what we input in the model
            feed_dict={
                model_dict['X_input']: X_train,
                model_dict['y_input']: y_train,
                model_dict['keepProb']: 0.5,
                model_dict['learningRate']: parameter['LEARNING_RATE'],
                model_dict['isTraining']: True
            }
        )
        logging.debug("Run on data successful")






