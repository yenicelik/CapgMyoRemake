from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from datahandler.BatchLoader import BatchLoader
from datahandler.Importer import *
from datahandler.DataLoader import *
import tensorflow as tf

import logging
logging = logging.getLogger(__name__)



#TODO: implement a 'sliding' windows over the framesize. Currently it works only to vote over frames of 1000 frames!
#TODO: remove verbosity
def voting(sess, model_dict, parameter, X, y, framesize=1000, verbose=False):
    logging.debug("Entering voting function")

    if 1000 % framesize != 0:
        logging.error("Framesize not divisible by 1000! Currently is: {}".format(framesize))
        sys.exit(11)

    batchLoder = BatchLoader(X, y, framesize, shuffle=False)
    predict_list = []
    actual_list = []
    epoch_done = False

    #TODO: what exactly do we do here? We vote over multiple frames to get a final, single frame
    while not epoch_done:
        logging.debug("Starting next voting session")

        X_batch, y_batch, epoch_done = batchLoder.load_batch()

        logging.debug("X_batch has shape: {}".format(X_batch.shape))
        logging.debug("y_batch has shape: {}".format(y_batch.shape))

        if not np.array_equal(y_batch, np.full(y_batch.shape, y_batch[0])): #i think it will fail on this part due to the dimensions!
            print("y_batch should be one-dimensional, but is of size: ", y_batch.shape)
            logging.error("y_batch should be one-dimensional, but is of size: ".format(y_batch.shape))
            logging.error("Not all samples are from the same sample! No majority voting possible!")
            sys.exit(11)

        #Do i need this?
        # X_tmp = np.reshape(X_batches[i], (-1, 16, 8))
        # y_tmp = y_batches[i]

        logging.debug("Entering sess.run from within the voting function")
        logit = sess.run(
                            #Describe what we want out of the model
                            [
                                model_dict['predict'],
                            ],
                            #Describe what we input in the model
                            feed_dict = {
                                model_dict['X_input']: X_batch,
                                model_dict['keepProb']: 1.,
                                model_dict['learningRate']: parameter['LEARNING_RATE'],
                                model_dict['isTraining']: False
                            }
        )
        logging.debug("Finished sess.run from within the voting function")

        logit = logit[0]

        predict = np.argmax(logit, axis=1) #should take the maximum value out of all values!
        logging.debug("Predict has shape: {} after choosing the maximum value (argmaxing)".format(predict.shape))

        predict = np.bincount(predict)
        predict = np.argmax(predict)
        logging.debug("Predict has shape: {} after having majority-voted (taking the most common occurence)".format(predict.shape))

        actual = np.bincount(np.argmax(y_batch, axis=1)) #should return a random element which is equivalent to all elements #Do i need to change this / process this?
        actual = np.argmax(actual)
        logging.debug("Actual has shape: {} after having majority-voted (taking the most common occurence)".format(actual.shape))

        predict_list.append(predict)
        actual_list.append(actual)

    difference_vector = [1 if pred == act else 0 for pred, act in zip(predict_list, actual_list)]
    accuracy = (np.sum(difference_vector) / float(len(predict_list)))

    ##############################
    # Confusion matrix
    ##############################
    logging.warning("Accuracy is: {:.3f}%".format(accuracy*100))
    logging.warning("Random baseline: {:.3f}%".format(1./32*100))

    return accuracy



#TODO: implement majority-voting over time-frame
def test_accuracy(sess, model_dict, parameter, X, y, verbose=False, show_confusion_matrix=False):
    """
    :param sess: The tensorflow session that we are going to use.
    :param model_dict: The model-dictionary that we need to refer to as the tensorflow-graph
    :param parameter: The parameter dictionary, holding different values needed.
    :param X: The test-set.
    :param y: The respective labels of the test-set.
    :param verbose: Whether the sample prediction and actual y-values should be printed.
    :return: Nothing
    """
    plt.interactive(False)
    plt.ion()
    #
    # plt.plot([1, 2, 3])
    # plt.show()

    #TODO: why should the batch_size be specified? I mean, in the end we want the entire X to be tested, right? at least the entire X that was input into this function..
    batchLoader = BatchLoader(X, y, parameter['BATCH_SIZE'], shuffle=False)
    X_batch, y_batch, epoch_done = batchLoader.load_batch()

    loss, logits = sess.run(
                        #Describe what we want out of the model
                        [
                            model_dict['loss'],
                            model_dict['predict'],
                        ],
                        #Describe what we input in the model
                        feed_dict = {
                            model_dict['X_input']: X_batch,
                            model_dict['y_input']: y_batch,
                            model_dict['keepProb']: 1.,
                            model_dict['learningRate']: parameter['LEARNING_RATE'],
                            model_dict['isTraining']: False
                        }
                    )

    predict = np.argmax(logits, axis=1)
    actual = np.argmax(y_batch, axis=1)

    if verbose:
        print("predict is: ", predict)
        print("actual is: ", actual)

    difference = [1 if pred == act else 0 for pred, act in zip(predict, actual)]
    accuracy = (np.sum(difference) / float(X_batch.shape[0]))

    ##############################
    # Confusion matrix
    ##############################
    print("Accuracy is: {:.3f}%".format(accuracy*100))
    print("Random baseline: {:.3f}%".format(1./32*100))

    if show_confusion_matrix:
        cm = confusion_matrix(actual, predict)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
        plt.show(block=True)

    return accuracy


