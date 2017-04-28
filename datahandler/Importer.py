from __future__ import print_function

import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)


#TODO: It would be the easiest to choose the correct classes when reading in the files i think... anything else is gonna be really slow. This is acceptable, but once the filenumbers are retrieved, one can do simple string-operations to filter out the necessary files. On the other side, \
# one needs to distuinguish between multiple sessions again. Not sure if generalization can be easily applied.
#TODO: in the batch_loader, we cannot guarantee we always have a satisfying batch_number to not cause an index-error. If the batch-size is a divider of 1000, this shouldn't be a problem for now. Beware of empty batches in case there is no gesture for a specific person
#TODO: implement Importer that lets you choose between: Intra-session, Cross-session, Cross-subject
#TODO: Potentially, it could be wise to also have an axis for the frame-order - in case we need to
#TODO: change this into a hard-coded style, where additional classes are input depending on how many previous classes exist (a function that counts classes, and returns the number of different classes given a certain dataset

class Importer(object):
    """
        Will finally contain an array, the 'super-matrix':
           0  1  2  3  4  5  6  ...  n
        -----------------------------------
        |  s  g  t  f  d  d  d  ...  d
        |  s  g  t  f  d  d  d  ...  d
        |  s  g  t  f  d  d  d  ...  d
        |  s  g  t  f  d  d  d  ...  d
        |  s  g  t  f  d  d  d  ...  d
        | ...
        Where:
         s is the respective subject ID
         g is the respective gesture ID
         t is the respective trial
         f is the respective frame order
         d is the respective emg frame data (16 * 8)
    """
    #TODO: implement a 'frame_number' property

    def __init__(self, data_parent_directory):
        """
        :param data_parent_directory: The parent directory in which all the datasets are located
        :return: -
        """
        logging.debug("-> {} function".format(self.__init__.__name__))
        #Get the dictionary data
        filepaths = self._get_filepaths_in_directory(data_parent_directory)
        data_array = self._get_data_from_filepaths(filepaths)

        #Extract the values
        #TODO: IMPORTANT! Check if these lists have the correct order. These must be aligned. How could we possible check these?
        subjects = self._extract_dict_property(data_array, "subject")
        trials = self._extract_dict_property(data_array, "trial")
        emgframes = self._extract_dict_property(data_array, "data")
        emgframes = np.reshape(emgframes, (-1, 16 * 8))

        logging.debug("Subjects has shape: {}".format(subjects.shape))
        logging.debug("Trials has shape: {}".format(trials.shape))
        logging.debug("EMGframes has shape: {}".format(emgframes.shape))

        frameorders = np.arange(0, 1000) #TODO: is this the maximum amount of frames? or: are all measurements taken over 1000 frames?
        frameorders = np.tile(frameorders, len(subjects))

        gestures = self._extract_dict_property(data_array, "gesture") #a function that turns indecies into sorted indecies.
        logging.debug("Gestures has shape: {}".format(subjects.shape))
        translate_dict = dict(enumerate(np.unique(gestures).tolist()))
        translate_dict = dict((v,k) for k,v in translate_dict.iteritems())
        logging.info("Gesture dict looks as follows: {}".format(translate_dict))
        translate = lambda x : translate_dict.get(x)
        gestures = map(translate, gestures)
        gestures = np.asarray(gestures)

        #need to reshape before we can apply anything
        subjects = np.reshape(subjects, (len(subjects), 1))
        gestures = np.reshape(gestures, (len(gestures), 1))
        trials = np.reshape(trials, (len(trials), 1))
        frameorders = np.reshape(frameorders, (len(frameorders), 1))

        if subjects.shape[0] != gestures.shape[0] or \
            gestures.shape[0] != trials.shape[0] or \
             trials.shape[0] != frameorders.shape[0]/1000:
            logging.error("Subjects {}, gestures {}, trials {} or frameorders {} have different shape!".format(
                subjects.shape[0],
                gestures.shape[0],
                trials.shape[0],
                frameorders.shape[0])
            )
            sys.exit(69)

        #Create the super-matrix
        sm = np.concatenate((subjects, gestures, trials), axis=1)
        sm = np.repeat(sm, 1000, axis=0)
        sm = np.concatenate((sm, frameorders), axis=1)
        sm = np.reshape(sm, (-1, 4))
        if sm.shape[0] != emgframes.shape[0]:
            logging.error("EMGframes {} and sm {} have different shape!".format(emgframes.shape, sm.shape))
            sys.exit(69)
        self.super_matrix = np.concatenate((sm, emgframes), axis=1)

        logging.debug("<- {} function".format(self.__init__.__name__))


    def get_super_matrix(self):
        """
        :return: The super matrix as described above
        """
        logging.debug("<-> {} function".format(self.get_super_matrix.__name__))
        return self.super_matrix


    def _get_filepaths_in_directory(self, parent_directory):
        """
        :param parent_directory: The parent directory from which all datasets are to be gathered
        :return: An array of filepaths (string) that are in the .mat format, and are included in the parent_directory
        """
        logging.debug("-> {} function".format(self._get_filepaths_in_directory.__name__))

        out = []
        for path, _, files in os.walk(parent_directory):
            for filename in files:
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    logging.debug("Found file {}".format(filename))
                    filepath = os.path.join(path, filename)
                    out.append(filepath)

        #Post-conditions
        if len(out) == 0:
            logging.error("No files in .mat-format were found!")

        logging.debug("<- {} function".format(self._get_filepaths_in_directory.__name__))
        return out


    def _get_data_from_filepaths(self, filepaths):
        """
        :param filepaths: A single filepath string, or an array of filepath strings
        :return: An array of data saved within the .mat format. In this case, these are dictionaries
        """
        logging.debug("-> {} function".format(self._get_data_from_filepaths.__name__))
        #Pre-Conditions
        if len(filepaths) == 0:
            logging.error("No filepaths are given! Quitting from get_data_from_filepath")
            logging.error(os.getcwd())
            sys.exit(69)

        out = []
        #Whether it is an array or a single string
        if type(filepaths) == type([1, 2]):
            logging.debug("Filepaths is an array")
            for filepath in filepaths:
                filedata = sio.loadmat(filepath)
                out.append(filedata)
        else:
            logging.debug("Filepaths is a single string with the value {}".format(filepaths))
            out = [sio.loadmat(filepaths)]

        #Post-Conditions
        if len(out) == 0:
            logging.error("We have no output")
            sys.exit(69)
        if len(out) != len(filepaths):
            logging.error("Something went wrong! We have {} data-elements, but {} filepaths!".format(len(out), len(filepaths)))
            sys.exit(69)

        logging.debug("<- {} function".format(self._get_data_from_filepaths.__name__))
        return out


    def _extract_dict_property(self, data_arr, prop):
        """
        :param data_arr: An array of dictionaries.
        :param prop: The property of the dictionary one wants to extract for every element.
        :return: An array, containing the the exctracted 'prop' from every element of 'data_arr'
        """
        logging.debug("-> {} function".format(self._extract_dict_property.__name__))
        #Pre-conditions
        if len(data_arr) == 0:
            logging.warning("data_arr is empty!")

        out = []
        for data in data_arr:
            tmp = data[prop]
            tmp = np.squeeze(tmp)
            out.append(tmp)

        #Post conditions
        if len(out) != len(data_arr):
            logging.error("We have {} data-elements, but {} input-elements!".format(len(out), len(data_arr)))

        logging.debug("<- {} function".format(self._extract_dict_property.__name__))
        # return np.asarray(out)
        return np.asarray(out)


    def _pop_dict_property(self, data_arr, prop):
        """
        :param data_arr: An array of dictionaries.
        :param prop: The property of the dictionary one wants to remove.
        :return: An array, containing dictionaries that don't have the property 'prop' anymore
        """
        logging.debug("-> {} function".format(self._pop_dict_property.__name__))
        #Pre-conditions
        if len(data_arr) == 0:
            logging.warning("data_arr is empty!")

        out = []
        for data in data_arr:
            tmp = data
            tmp.pop(prop, None)
            out.append( tmp )

        #Post-conditions
        if len(out) != len(data_arr):
            logging.error("We have {} data-elements, but {} input-elements!".format(len(out), len(data_arr)))

        logging.debug("<- {} function".format(self._pop_dict_property.__name__))
        return out


if __name__ == "__main__":

    importer = Importer("Datasets/Preprocessed/DB-a")
    sm = importer.get_super_matrix()
    logging.debug("{}".format(sm))
    logging.debug("Super matrix has shape: {}".format(sm.shape))







