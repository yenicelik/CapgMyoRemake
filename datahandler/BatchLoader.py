from __future__ import print_function

import sys
import numpy as np

import logging
logging = logging.getLogger(__name__)


class BatchLoader(object):

    def __init__(self, X, y, batch_size, shuffle):
        logging.debug("-> {} function".format(self.__init__.__name__))
        self.no_of_batches = X.shape[0] / batch_size
        self.batch_counter = 0
        self.samples = X.shape[0]
        self.batch_size = batch_size

        if self.no_of_batches * batch_size != X.shape[0]:
            logging.error("The data {} is not divisible by the batch_size {} ({} batches)!".format(X.shape[0], batch_size, self.no_of_batches))
            sys.exit(69)

        oldshape = X.shape
        if shuffle:
            logging.debug("Shuffling dataset")
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices,:]
            y = y[indices]

        #Making sure shuffled shape is equivalent to old shape
        if oldshape != X.shape:
            logging.error("Shuffle has changed the shape!")

        logging.debug("X has shape: {}".format(X.shape))
        logging.debug("y has shape: {}".format(y.shape))
        logging.info("There are {} batches, each of size {}".format(self.no_of_batches, batch_size))

        self.X_arr = np.split(X, self.no_of_batches, axis=0)
        self.y_arr = np.split(y, self.no_of_batches, axis=0)
        logging.debug("<- {} function".format(self.__init__.__name__))


    def load_batch(self):
        """
        :return: A (random) batch of X with the corresponding labels y, and a signal wether one epoch has passed
        """
        outX = self.X_arr[self.batch_counter]
        outy = self.y_arr[self.batch_counter]
        epoch_passed = False
        self.batch_counter += 1

        if self.batch_counter >= self.no_of_batches:
            if self.batch_counter * self.batch_size != self.samples:
                logging.error("Not all data-samples have been processed, but epoch is signalled as done!")
                sys.exit(69)
            epoch_passed = True
            logging.debug("{}".format(self.batch_counter))
            logging.debug("Epoch has passed in load_batch!")
            self.batch_counter = self.batch_counter % self.no_of_batches

        return outX, outy, epoch_passed

