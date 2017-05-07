from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from model.build_model import *

from datahandler.BatchLoader import BatchLoader

import logging
logging = logging.getLogger(__name__)

def test_model_accuracy_voting(X, y, parameter, sess, model):
    batchLoder = BatchLoader(X, y, 1000, shuffle=False)

    predict_list = []
    actual_list = []
    epoch_done = False
    while not epoch_done:
        X_batch, y_batch, epoch_done = batchLoder.load_batch()

        logits = sess.run(
                #Describe what we want out of the model
                [
                    model['predict']
                ],
                #Describe what we input in the model
                feed_dict = {
                    model['X']: X_batch,
                    model['y']: y_batch,
                    model['keep_prob']: 1.,
                    #model['lr']: parameter['LEARNING_RATE'],
                    model['is_training']: False
                }
        )

        #Majority-Voting
        predict = np.argmax(logits[0], axis=1)
        predict = np.bincount(predict)
        predict = np.argmax(predict)

        actual = np.argmax(y_batch, axis=1)
        actual = np.bincount(actual)
        actual = np.argmax(actual)

        predict_list.append(predict)
        actual_list.append(actual)

    difference_vector = [1 if pred == act else 0 for pred, act in zip(predict_list, actual_list)]
    accuracy = (np.sum(difference_vector) / float(len(difference_vector)))

    return accuracy


def test_model_accuracy(X, y, parameter, sess, model):
    batchLoader = BatchLoader(X, y, 1000, shuffle=False)

    acc_list = []
    epoch_done = False
    while not epoch_done:
        X_batch, y_batch, epoch_done = batchLoader.load_batch()

        accuracy = sess.run(
                #Describe what we want out of the model
                [
                    model['accuracy']
                ],
                #Describe what we input in the model
                feed_dict = {
                    model['X']: X_batch,
                    model['y']: y_batch,
                    model['keep_prob']: 1.,
                    model['is_training']: False
                }
            )
        acc_list.append(accuracy)

    final_acc = (np.sum(acc_list) / float(len(acc_list)))

    return final_acc




