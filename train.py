#External dependencies
import numpy as np
import tensorflow as tf
import random
import sys
import datetime
import time

#Project dependencies
from Model import init_graph
from Importer import *
from BatchLoader import BatchLoader


"""
parameter is a dictionary consisting of
'NUM_EPISODES':

"""

#TODO: Epoch == Full pass through the data
#TODO: the model should be saved on the harddrive frequently!

def train(parameter, model_dict, X, y):
    """ Trains the network to the given environment. Saves weights to saver
        In: parameter (Dictionary with settings)
        In: saver (tf.saver where weights and bias will be saved to)
        In: forward_dict (Dictionary referencing to the tensorflow model)
        In: loss_dict (Dictionary referencing to the tensorflow model)
        Out: rewards_list (reward for instantenous run)
        Out: steps_list (number of steps 'survived' in given episode)
    """
    loss_list = []

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in xrange(parameter['NUM_EPOCHS']):

            start_time = datetime.datetime.now()

            loss = run_epoch(
                                cur_epoch=epoch,
                                sess=sess,
                                parameter=parameter,
                                model_dict=model_dict,
                                X=X,
                                y=y
            )

            end_time = datetime.datetime.now()
            total_time = end_time - start_time

            loss_list.extend(loss)

            percentage = float(epoch) / parameter['NUM_EPOCHS']

            print("Progress: {0:.3f}%%".format(percentage * 100))
            print("EST. time per episode: " + str(total_time))
            print("Epochs left: {0:d}".format(parameter['NUM_EPOCHS'] - epoch))
            print("Average loss of current epoch: " + str(np.sum(loss_list)/parameter['NUM_EPOCHS']))
            print("")

    return loss_list


#TODO: update hyperparameters depending to update rule (gradient descent rule etc.)
#TODO: potentially create a cross-validation option to see how the model performs /CV loss vs Training loss
def run_epoch(sess, cur_epoch, parameter, model_dict, X, y):
    """ Run one episode of  environment
        In: cur_episode
        In: parameter
        Out: total_reward (accumulated over episode)
        Out: steps (needed until termination)
    """

    loss_list = []
    batchLoader = BatchLoader(X, y, parameter['BATCH_SIZE'], shuffle=False)

    epoch_done = False
    while not epoch_done:
        X_batch, y_batch, epoch_done = batchLoader.load_batch()


        loss, predict, _ = sess.run(
                        #Describe what we want out of the model
                        [
                            model_dict['loss'],
                            model_dict['predict'],
                            model_dict['updateModel']
                        ],
                        #Describe what we input in the model
                        feed_dict = {
                            model_dict['X_input']: X_batch,
                            model_dict['y_input']: y_batch
                        }
                    )

        loss_list.append(loss/parameter['BATCH_SIZE'])

    return loss_list


