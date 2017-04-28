from __future__ import print_function

from datahandler.DataLoader import DataLoader
from datahandler.BatchLoader import BatchLoader
import numpy as np

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

        epoch_loss_list = run_epoch(X, y, parameter, epoch)

        total_loss_list.extend(epoch_loss_list)
        progress = float(epoch) / parameter['NUM_EPOCHS']

        if saverObj is not None:
            logging.info("Saving...")
            saverObj.save_session(sess, parameter['SAVE_DIR'])
            logging.debug("Save success at {}".format(parameter['SAVE_DIR']))

        logging.info("Progress: {0:.3f}%".format(progress * 100))
        logging.info("Average loss {0:.3f}".format(float(sum(total_loss_list))/len(total_loss_list)))
        print("Epoch Progress: {0:.3f}%".format(progress * 100))
        print("Average loss {0:.3f}".format(float(sum(total_loss_list))/len(total_loss_list)))

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
        if batchLoader.batch_counter % (batchLoader.no_of_batches / 10) == 0:
            logging.debug("Step progress: {:3f}% in epoch {}".format(100. * batchLoader.batch_counter / batchLoader.no_of_batches, cur_epoch))

        X_batch, y_batch, epoch_done = batchLoader.load_batch()

        if X_batch.shape[0] == 0 or y_batch.shape[0] == 0:
            logging.warning("X_batch {} or y_batch {} is of size 0!".format(X_batch.shape, y_batch))
        if X_batch.shape[0] != y_batch.shape[0]:
            logging.warning("X_batch shape {} does not equal y_batch shape {}".format(X_batch.shape, y_batch.shape))
            sys.exit(69)

        logging.debug("-> sess.run function")
        try:
            print("Running step {} now!".format(batchLoader.batch_counter))
            print("X_batch.shape: {}".format(X.shape))
            print("y_batch.shape: {}".format(y.shape))
            print("learning_rate: {}".format(learning_rate))
            print("Leaving running epoch!")
            loss = np.random.randint(0, 1000)
            logit = np.random.rand(X_batch.shape[0], 10) #10 should be the total number of different classes. This should be a global variable maybe
        except Exception as e:
            logging.error("Tensorflow threw an error: {}".format(e))

        logging.debug("<- sess.run function")

        loss = np.random.rand(y_batch.shape[0])
        loss_list.append(float(sum(loss)) / parameter['BATCH_SIZE'])


    logging.debug("<- {} function".format(run_epoch.__name__))
    return loss_list


def step():
    pass


def adapt_lr(epoch, para_lr):
    logging.debug("-> {} function".format(adapt_lr.__name__))
    if epoch >= 24:
        learning_rate = para_lr / 100
        logging.debug("Diving learning rate by 100. Before: {}; After: {}".format(para_lr, learning_rate))
    elif epoch >= 16:
        learning_rate = parameter['LEARNING_RATE'] / 10
        logging.debug("Diving learning rate by 10. Before: {}; After: {}".format(para_lr, learning_rate))
    else:
        learning_rate = para_lr

    logging.debug("<- {} function".format(adapt_lr.__name__))
    return learning_rate



if __name__ == '__main__':
    dataLoader = DataLoader("../datahandler/Datasets/Preprocessed/DB-a")
    X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    parameter = {
        'LEARNING_RATE': 500,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000
    }
    train(X_train, y_train, parameter)
