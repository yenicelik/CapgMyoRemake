from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from model.BuildGraph import *

from datahandler.BatchLoader import BatchLoader
from datahandler.DataLoader import *

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

def test_model_accuracy_voting(X, y, parameter, voting_window=1000, sess=None, model_dict=None):
    """
    :param X: The data for which labels should be predicted / The test data
    :param y: The true labels of the data
    :param parameter: The common parameter dictionary
    :param voting_window: Over how many frames voting should occur. Default 1000. Maximum 1000.
    :param sess: The session (reference to the tensorflow session). If None, random values are generated to test the correctness of the function
    :param model_dict: The model dictionary (reference to the tensorflow object)
    :return: The accuracy of the model.
    """
    logging.debug("-> {} function".format(test_model_accuracy.__name__))
    #We currently only take the first 'voting_window' number of frames, vote based on them. We ignore the result
    batchLoder = BatchLoader(X, y, 1000, shuffle=False) #we must split the data by 1000's
    predict_list = []
    actual_list = []
    epoch_done = False

    if voting_window > 1000:
        logging.error("Voting window {} is too big (max 1000)".format(voting_window))
        sys.exit(69)
    if X.shape[0] != y.shape[0]:
        logging.error("X.shape {} is different from y.shape {}".format(X.shape[0], y.shape[0]))
        sys.exit(69)
    if X.shape[0] % 1000 != 0 or y.shape[0] % 1000 != 0:
        logging.error("X {} or y {} are not divisible by 1000!".format(X.shape[0], y.shape[0]))

    while not epoch_done:
        logging.debug("Starting next voting session")
        X_batch, y_batch, epoch_done = batchLoder.load_batch()
        X_batch = X_batch[:voting_window]
        y_batch = y_batch[:voting_window]
        if not ((y_batch - y_batch[0,:]) == 0).all():
            logging.error("y_batch be equal everywhere, but is not. As such, we have different gestures collected! y_batch: {}".format(y_batch))
            sys.exit(69)

        #Running function
        logging.debug("-> sess.run function")
        try:
            if sess is None:
                loss = np.random.randint(0, 1000)
                logits = np.random.rand(X_batch.shape[0], 10) #10 should be the total number of different classes. This should be a global variable maybe
            else:
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
        except Exception as e:
            logging.error("Tensorflow threw an error: {}".format(e))
            sys.exit(69)
        logging.debug("<- sess.run function")

        #Majority-Voting
        predict = np.argmax(logits, axis=1)
        predict = np.bincount(predict)
        predict = np.argmax(predict)
        actual = np.bincount(np.argmax(y_batch, axis=1)) #should return a random element which is equivalent to all elements #Do i need to change this / process this?
        actual = np.argmax(actual)
        predict_list.append(predict)
        actual_list.append(actual)

    if len(predict_list) != len(actual_list):
        logging.error("predict_list length {} is not equal to actual_list length {}".format(len(predict_list), len(actual_list)))

    difference_vector = [1 if pred == act else 0 for pred, act in zip(predict_list, actual_list)]
    accuracy = (np.sum(difference_vector) / float(len(predict_list)))

    logging.warning("Voting accuracy over {} frames is: {:.3f}%".format(voting_window, accuracy*100))
    logging.warning("Random baseline: {:.3f}%".format(1./10*100))

    return accuracy


def test_model_accuracy(X, y, parameter, show_confusion_matrix, sess=None, model_dict=None):
    """
    :param X: The data for which labels should be predicted / The test data
    :param y: The true labels of the data
    :param parameter: The common parameter dictionary
    :param show_confusion_matrix: True or False; Whether or not the confusion matrix should be shown
    :param sess: The session (reference to the tensorflow session)
    :param model_dict: The model dictionary (reference to the tensorflow object)
    :return: The model accuracy
    """
    logging.debug("-> {} function".format(test_model_accuracy.__name__))

    #We run the model
    logging.debug("-> sess.run function")
    #TODO: placeholders that are going to be replaced by the actual values
    try:
        if sess is None:
            loss = np.random.randint(0, 1000)
            logits = np.random.rand(X.shape[0], 10) #10 should be the total number of different classes. This should be a global variable maybe
        else:
            loss, logits = sess.run(
                    #Describe what we want out of the model
                    [
                        model_dict['loss'],
                        model_dict['predict'],
                    ],
                    #Describe what we input in the model
                    feed_dict = {
                        model_dict['X_input']: X,
                        model_dict['y_input']: y,
                        model_dict['keepProb']: 1.,
                        model_dict['learningRate']: parameter['LEARNING_RATE'],
                        model_dict['isTraining']: False
                    }
                )
    except Exception as e:
        logging.error("Tensorflow threw an error: {}".format(e))
        sys.exit(69)
    logging.debug("<- sess.run function")

    #Both reduced from one-hot to arg-vector
    predict = np.argmax(logits, axis=1)
    actual = np.argmax(y, axis=1)
    if len(predict) != len(actual):
        logging.error("Predict length {} doesn't equal actualy length {}".format(len(predict), len(actual)))
        sys.exit(69)
    difference_vector = [1 if pred == act else 0 for pred, act in zip(predict, actual)]
    accuracy = np.sum(difference_vector) / float(len(difference_vector))
    logging.info("Accuracy is: {:.3f}%".format(accuracy*100))
    logging.info("Random baseline: {:.3f}%".format(1./10*100))

    #Confusion Matrix
    if show_confusion_matrix:
        cm = confusion_matrix(actual, predict)
        logging.debug("Showing confusion matrix: {}".format(cm))
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
        plt.show(block=True)

    logging.debug("<- {} function".format(test_model_accuracy.__name__))
    return accuracy



if __name__ == '__main__':
    dataLoader = DataLoader("../datahandler/Datasets/Preprocessed/DB-a")
    X_pretrain, y_pretrain, cross_session_dataset = dataLoader.get_cross_session_dataset_given_subject_id(1)

    parameter = {
        'BATCH_SIZE': 500
    }

    sess = None

    W, b, model_dict, tmpval= init_graph()

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # acc = test_model_accuracy(X_pretrain, y_pretrain, parameter, show_confusion_matrix=False)
        acc = test_model_accuracy_voting(X_pretrain, y_pretrain, parameter)

        print("Accuracy is {}".format(acc*100))


