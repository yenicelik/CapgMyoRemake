import numpy as np
import tensorflow as tf
import sys

class BatchLoader(object):
    """
        Takes in training data X with corresponding labels y and a batch_size. Potentially shuffles them.
        Allows to get batches until an epoch has passed
    """

    def __init__(self, X, y, batch_size, shuffle=True):
        """
        :param X: The training data
        :param y: The corresponding labels of the training data
        :param batch_size: The batch size
        :param shuffle: Whether the input data should be shuffled or not
        :return: -
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.number_of_batches = self.X.shape[0] / self.batch_size
        self.batch_counter = 0

        #Check if all batches are of equal size
        if self.number_of_batches * self.batch_size != self.X.shape[0]:
            print("Number of batches: ", self.number_of_batches)
            print("Batch size: ", self.batch_size)
            print("Data samples: ", self.X.shape[0])
            print("The data is not divisible by the batch_size!")
            sys.exit(11)

        #Potentially shuffle dataset
        if shuffle:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices,:]
            y = y[indices]

        print(X.shape)
        print(y.shape)
        print(self.number_of_batches)
        self.X_arr = np.split(X, self.number_of_batches, axis=0)
        self.y_arr = np.split(y, self.number_of_batches, axis=0)


    def load_batch(self):
        """
        :return: A (random) batch of X with the corresponding labels y, and a signal whether one epoch has passed
        """

        outX = self.X_arr[self.batch_counter]
        outy = self.y_arr[self.batch_counter]

        self.batch_counter += 1

        epoch_passed = False
        if self.batch_counter >= self.number_of_batches:
            epoch_passed = True
            self.batch_counter = self.batch_counter % self.number_of_batches

        return outX, outy, epoch_passed




if __name__ == '__main__':
    pass
    # X = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    # y = np.asarray([1, 2, 3, 4, 5, 6])
    # print(X.shape)
    # print(y.shape)
    # dl = DataLoader(X, y, 2, shuffle=True)
    #
    # epoch_done = False
    # while not epoch_done:
    #     print("\n")
    #     print(dl.batch_counter)
    #     X_batch, y_batch, epoch_done = dl.load_batch()
    #     print(X_batch)
    #     print(y_batch)
    #     print(epoch_done)