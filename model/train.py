from __future__ import print_function

from datahandler.DataLoader import DataLoader
from datahandler.BatchLoader import BatchLoader
from model.BuildGraph import *
from datahandler.TFSaver import TFSaver
import numpy as np
import tensorflow as tf

import sys
import logging
logging = logging.getLogger(__name__)

# parameter = {
#            'NUM_EPOCHS': 1,
#            'BATCH_SIZE': 100,
#            'SAVE_DIR': 'saves/',
#            'SAVE_EVERY': 500 #number of batches after which to save
#    }


# TODO: the model should be saved on the harddrive frequently!
# TODO: implement cross-validation for checking model capacity and general accuracy
def train(X, y, parameter, sess=None, model_dict=None, saverObj=None):
    logging.debug("-> {} function".format(train.__name__))

    total_loss_list = []

    for epoch in range(parameter['NUM_EPOCHS']):
        logging.debug("Entering Epoch {}".format(epoch))

        epoch_loss_list = run_epoch(X, y, parameter, epoch, sess=sess, model_dict=model_dict)

        total_loss_list.extend(epoch_loss_list)
        progress = float(epoch) / parameter['NUM_EPOCHS']

        if saverObj is not None:
            logging.info("Saving...")
            saverObj.save_session(sess, global_step=model_dict['globalStepTensor'])
            logging.debug("Save success at {}".format(parameter['SAVE_DIR']))

        logging.info("Progress: {0:.3f}%".format(progress * 100))
        logging.info("Average loss {0:.5f}".format(float(sum(total_loss_list))/len(total_loss_list)))
        print("Epoch Progress: {0:.3f}%".format(progress * 100))
        print("Average loss {0:.5f}".format(float(sum(total_loss_list))/len(total_loss_list)))

    logging.debug("<- {} function".format(train.__name__))
    return total_loss_list



def run_epoch(X, y, parameter, cur_epoch, sess=None, model_dict=None):
    """ Run one episode of  environment
        In: cur_episode
        In: parameter
        Out: total_reward (accumulated over episode)
        Out: steps (needed until termination)
    """
    logging.debug("-> {} function".format(run_epoch.__name__))

    epoch_done = False
    loss_list = []
    batchLoader = BatchLoader(X, y, parameter['BATCH_SIZE'], shuffle=True)
    learning_rate = adapt_lr(cur_epoch, parameter['LEARNING_RATE'])

    while not epoch_done:
        if batchLoader.batch_counter % max(batchLoader.no_of_batches / 10, 1) == 0:
            logging.debug("Step progress: {:3f}% in epoch {}".format(100. * batchLoader.batch_counter / batchLoader.no_of_batches, cur_epoch))

        X_batch, y_batch, epoch_done = batchLoader.load_batch()

        if X_batch.shape[0] == 0 or y_batch.shape[0] == 0:
            logging.warning("X_batch {} or y_batch {} is of size 0!".format(X_batch.shape, y_batch))
        if X_batch.shape[0] != y_batch.shape[0]:
            logging.warning("X_batch shape {} does not equal y_batch shape {}".format(X_batch.shape, y_batch.shape))
            sys.exit(69)

        logging.debug("-> sess.run function")
        try:
            if sess is None:
                logging.debug("USING RANDOM")
                loss = np.random.rand(y_batch.shape[0])
                predict = np.random.rand(X_batch.shape[0], 10) #10 should be the total number of different classes. This should be a global variable maybe
            else:
                loss, predict, _, _ = sess.run(
                # Describe what we want out of the model
                [
                    model_dict['loss'],
                    model_dict['predict'],
                    model_dict['updateModel'],
                    model_dict['globalStepTensor']

                ],
                # Describe what we input in the model
                feed_dict={
                    model_dict['X_input']: X_batch,
                    model_dict['y_input']: y_batch,
                    model_dict['keepProb']: 0.5,
                    model_dict['learningRate']: learning_rate,
                    model_dict['isTraining']: True
                }
            )
        except Exception as e:
            logging.error("Tensorflow threw an error: {}".format(e))
        logging.debug("<- sess.run function")

        loss_list.append(float(sum(loss)) / len(loss))


    logging.debug("<- {} function".format(run_epoch.__name__))
    return loss_list



def adapt_lr(epoch, para_lr):
    logging.debug("-> {} function".format(adapt_lr.__name__))
    if epoch >= 24:
        learning_rate = para_lr / 100
        logging.debug("Diving learning rate by 100. Before: {}; After: {}".format(para_lr, learning_rate))
    elif epoch >= 16:
        learning_rate = para_lr / 10
        logging.debug("Diving learning rate by 10. Before: {}; After: {}".format(para_lr, learning_rate))
    else:
        learning_rate = para_lr

    logging.debug("<- {} function".format(adapt_lr.__name__))
    return learning_rate



if __name__ == '__main__':
    dataLoader = DataLoader("../datahandler/Datasets/Preprocessed/DB-a")
    X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    W, b, model_dict= init_graph()

    features = W.copy()
    features.update(b)
    features.update({'global_step': model_dict['globalStepTensor']})
    TfSaver = TFSaver("saves/", features)
    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000,
        'SAVE_DIR': ""
    }

    init_op = tf.global_variables_initializer()

    acc_list = []
    with tf.Session() as sess:
        sess.run(init_op)
        train(X_train[:2000], y_train[:2000], parameter, sess=sess, model_dict=model_dict, saverObj=TfSaver)
