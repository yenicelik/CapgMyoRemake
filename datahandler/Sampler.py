from __future__ import print_function

import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from Importer import Importer

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)


class Sampler(Importer):

    def __init__(self, data_parent_directory):
        """
        :param data_parent_directory: The parent directory in which all the datasets are located
        :return: -
        """
        logging.debug("-> {} function".format(self.__init__.__name__))
        super(Sampler, self ).__init__(data_parent_directory)
        logging.debug("<- {} function".format(self.__init__.__name__))


    def sample_data(self):
        """
        :return: -
        """
        logging.debug("-> {} function".format(self.sample_data.__name__))
        plt.ion()

        #select random data sample
        index = np.random.randint(0, (self.super_matrix.shape[0]/1000)-1)
        sample_X = self.super_matrix[index:(index+1000),4:]
        sample_y = self.super_matrix[index:(index+1000),1]

        self._play_data(sample_X, sample_y, 1000)
        logging.debug("<- {} function".format(self.sample_data.__name__))

    #TODO: There is a bug in which the .pause command seems to accumulates
    def _play_data(self, data, label, framerate):
        """
        :param data: Sample X value to be displayed
        :param label: Respective label of the video
        :param framerate: How many frames per second to be displayed
        :return:
        """
        logging.debug("-> {} function".format(self._play_data.__name__))
        video = []
        for raw_frame in data:
            frame = np.reshape(raw_frame, (16, 8))
            video.append(frame)

        fig = plt.gcf()
        fig.canvas.set_window_title("Gesture: " + str(np.argmax(label)))

        i = 0
        for frame in video:
            i += 1
            if i >= 30:
                break

            plt.imshow(frame, cmap="gray")
            plt.pause(1./framerate)

        raw_input("Press Enter to continue...")
        logging.debug("<- {} function".format(self._play_data.__name__))


if __name__ == "__main__":

    sampler = Sampler("Datasets/Preprocessed/DB-a")
    sm = sampler.get_super_matrix()
    sampler.sample_data()

