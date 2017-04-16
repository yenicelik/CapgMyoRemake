from __future__ import print_function
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
from Saver import *


"""
parameter is a dictionary consisting of
'NUM_EPISODES':

"""

#TODO: Epoch == Full pass through the data
#TODO: the model should be saved on the harddrive frequently!

def train(sess, parameter, model_dict, X, y, saverObj):
    """ Trains the network to the given environment. Saves weights to saver
        In: parameter (Dictionary with settings)
        In: saver (tf.saver where weights and bias will be saved to)
        In: forward_dict (Dictionary referencing to the tensorflow model)
        In: loss_dict (Dictionary referencing to the tensorflow model)
        Out: rewards_list (reward for instantenous run)
        Out: steps_list (number of steps 'survived' in given episode)
    """

    for epoch in xrange(parameter['NUM_EPOCHS']):

        loss_list = []

        start_time = datetime.datetime.now()

        loss = run_epoch(
                            cur_epoch=epoch,
                            sess=sess,
                            parameter=parameter,
                            model_dict=model_dict,
                            X=X,
                            y=y,
                            saverObj=saverObj
        )

        end_time = datetime.datetime.now()
        total_time = end_time - start_time



        loss_list.extend(loss)

        percentage = float(epoch) / parameter['NUM_EPOCHS']

        saverObj.save_session(sess, parameter['SAVE_DIR'])

        print("Progress: {0:.3f}%%".format(percentage * 100))
        print("EST. time per episode: " + str(total_time))
        print("Epochs left: {0:d}".format(parameter['NUM_EPOCHS'] - epoch))
        print("Average loss of current epoch: " + str(np.sum(loss_list)/parameter['NUM_EPOCHS']))
        print("")

    return loss_list


#TODO: update hyperparameters depending to update rule (gradient descent rule etc.)
#TODO: potentially create a cross-validation option to see how the model performs /CV loss vs Training loss
def run_epoch(sess, cur_epoch, parameter, model_dict, X, y, saverObj):
    """ Run one episode of  environment
        In: cur_episode
        In: parameter
        Out: total_reward (accumulated over episode)
        Out: steps (needed until termination)
    """

    loss_list = []
    batchLoader = BatchLoader(X, y, parameter['BATCH_SIZE'], shuffle=True)

    epoch_done = False
    save_iter = 0
    while not epoch_done:
        X_batch, y_batch, epoch_done = batchLoader.load_batch()

        loss, predict, _, _ = sess.run(
                        #Describe what we want out of the model
                        [
                            model_dict['loss'],
                            model_dict['predict'],
                            model_dict['updateModel'],
                            model_dict['globalStepTensor']

                        ],
                        #Describe what we input in the model
                        feed_dict = {
                            model_dict['X_input']: X_batch,
                            model_dict['y_input']: y_batch
                        }
                    )

        print("Step progress: ",  100. * batchLoader.batch_counter/ batchLoader.number_of_batches )
        print("Training Loss: ", np.sum(loss)/batchLoader.batch_size)

        if save_iter % parameter['SAVE_EVERY'] == 0:
            saverObj.save_session(sess, parameter['SAVE_DIR'], tf.train.global_step(sess, global_step_tensor=model_dict['globalStepTensor'])) #step in terms of batches #cur_epoch * batchLoader.number_of_batches + batchLoader.batch_counter

        save_iter += 1

        loss_list.append(loss/parameter['BATCH_SIZE'])

    return loss_list


