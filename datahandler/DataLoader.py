from __future__ import print_function

from Importer import Importer

import numpy as np
import sys

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)


#TODO: check which functions are actually used, and remove the other ones.
class DataLoader(Importer):

    def __init__(self, data_parent_directory):
        logging.debug("-> {} function".format(self.__init__.__name__))
        super(DataLoader, self ).__init__(data_parent_directory)

        #Data might be incomplete for certain repeats. Thus, we must iterate over all of them
        self.subjects = np.unique(self.super_matrix[:, 0].astype(int))
        self.gestures = np.unique(self.super_matrix[:, 1].astype(int))
        self.trials = np.unique(self.super_matrix[:, 2].astype(int))

        logging.debug("We have {} as trials".format(self.trials))
        logging.debug("We have {} as subjects".format(self.subjects))
        logging.debug("We have {} as gestures".format(self.gestures))

        logging.debug("<- {} function".format(self.__init__.__name__))


    def _get_X_and_y(self, inp_matrix):
        """
        :param inp_matrix: The partial super_matrix from which y and X values should be returned
        :return: Returns the entire dataset X, and their respective y one-hot labels
        """
        logging.debug("-> {} function".format(self._get_X_and_y.__name__))
        #Exctract raw values
        X = inp_matrix[:, 4:]
        y = inp_matrix[:, 1].astype(int)

        #Turn y into one-hot and X into frameshape
        X_frames = np.reshape(X, (-1, 16, 8)) #TODO: check if this causes any trouble!
        y_hot = np.zeros((y.shape[0], len(self.gestures)))
        y_hot[np.arange(y.shape[0]), y] = 1

        #Post-Condition
        for i in xrange(y.shape[0]):
            if np.sum(y_hot[i, :]) != 1:
                logging.error("Something wrong with one-hot encoding!")
                sys.exit(69)

        logging.debug("<- {} function".format(self._get_X_and_y.__name__))
        return X_frames, y_hot


    def get_cross_subject_dataset(self):
        """
        We have little data, so we assume the user wants to conduct leave-one-out cross-validation
        :return: An array of tuples (X, y) for each subject. X represents the emgframes, y the gesture-label in one-hot representation.
        """
        logging.debug("-> {} function".format(self.get_cross_subject_dataset.__name__))
        out = []

        #Split the array by subjects and extract emgframes and label
        tmp_out = np.split(self.super_matrix, np.where(np.diff(self.super_matrix[:, 0]))[0]+1)
        for ele in tmp_out:
            if ele.size == 0:
                logging.warning("Element has size 0, but will be added to the cross-subject training set!")
            tmp = self._get_X_and_y(ele)
            out.append(tmp)

        #Post-Conditions
        if len(out) == 0:
            logging.error("The array to be output is empty!")
            sys.exit(69)
        total_length = sum([x[0].shape[0] for x in out])
        if total_length == 0:
            logging.error("All elements of the array to be output is empty!")
            sys.exit(69)
        if total_length != self.super_matrix.shape[0]:
            logging.error("Some data was lost, or not all rows have a subject-id associated with it!")
            sys.exit(69)

        logging.debug("<- {} function".format(self.get_cross_subject_dataset.__name__))
        return out


    #TODO: read how they did this in the paper!
    def get_cross_session_dataset_given_subject_id(self, subject_id):
        """
        :param subject_id: The subject id on which we will perform cross-session.
        :return:
        """
        logging.debug("-> {} function".format(self.get_cross_session_dataset_given_subject_id.__name__))
        X_pretrain, y_pretrain = self._get_cross_session_pretrain_given_subject_id(subject_id)
        out = self._get_cross_session_train_given_subject_id(subject_id)

        #Post-Condition
        total_length = sum([x[0].shape[0] for x in out])
        total_length += X_pretrain.shape[0]
        if total_length != self.super_matrix.shape[0]:
            logging.error("Some data was lost, or not all rows have a trial-id associated with it!")
            sys.exit(69)

        logging.debug("<- {} function".format(self.get_cross_session_dataset_given_subject_id.__name__))
        return X_pretrain, y_pretrain, out


    def _get_cross_session_pretrain_given_subject_id(self, subject_id):
        """
        :param subject_id: The subject id on which we will perform cross-session.
        :return:
        """
        logging.debug("-> {} function".format(self._get_cross_session_pretrain_given_subject_id.__name__))
        X = None
        y = None
        found_elements = self.super_matrix[
            (self.super_matrix[:,0] != subject_id)
        ]

        #Post-Conditions
        if found_elements.size != 0:
            X, y = self._get_X_and_y(found_elements)
        if X is None or y is None:
            logging.error("There are no subjects with id {} found".format(subject_id))

        logging.debug("<- {} function".format(self._get_cross_session_pretrain_given_subject_id.__name__))
        return X, y


    def _get_cross_session_train_given_subject_id(self, subject_id):
        """
        :param subject_id: The subject id on which we will perform cross-session.
        :return: Return a tuple of the form (X, y), where X is a tensor of the frames and y contains the corresponding gesture labels.
        """
        logging.debug("-> {} function".format(self._get_cross_session_train_given_subject_id.__name__))
        out = []
        #Take subsection of super_matrix which contains only samples from a specific subject
        all_elements = self.super_matrix[
            (self.super_matrix[:,0] == subject_id)
        ]

        #Split into cross-validat'able' array
        #Split the array by subjects and extract emgframes and label
        tmp_out = np.split(all_elements, np.where(np.diff(all_elements[:, 2]))[0]+1)
        for ele in tmp_out:
            if ele.size == 0:
                logging.warning("Element has size 0, but will be added to the cross-session training set!")
            tmp = self._get_X_and_y(ele)
            out.append(tmp)

        logging.debug("<- {} function".format(self._get_cross_session_train_given_subject_id.__name__))
        return out

    def get_intra_session_dataset_given_sid_tid(self, subject_id, trial_id):
        logging.debug("-> {} function".format(self.get_intra_session_dataset_given_sid_tid.__name__))
        out = []
        #Generate pretrain dataset (including everything but the chosen subject and trial id
        X_pretrain1, y_pretrain1 = self._get_cross_session_pretrain_given_subject_id(subject_id)
        X_pretrain2, y_pretrain2 = self._get_intra_session_pretrain_given_sid_tid(subject_id, trial_id)
        X_pretrain = np.concatenate((X_pretrain1, X_pretrain2), axis=0)
        y_pretrain = np.concatenate((y_pretrain1, y_pretrain2), axis=0)

        #Generate (X, y) tuples
        out = self._get_intra_session_train_given_sid_tid(subject_id, trial_id)

        #Post-Condition
        total_length = sum([x[0].shape[0] for x in out])
        total_length += X_pretrain.shape[0]
        if total_length != self.super_matrix.shape[0]:
            logging.error("Some data was lost, or not all rows have a gesture-id associated with it!")
            sys.exit(69)

        logging.debug("<- {} function".format(self.get_intra_session_dataset_given_sid_tid.__name__))
        return X_pretrain, y_pretrain, out

    def _get_intra_session_pretrain_given_sid_tid(self, subject_id, trial_id):
        logging.debug("-> {} function".format(self._get_intra_session_pretrain_given_sid_tid.__name__))
        X = None
        y = None
        found_elements = self.super_matrix[
            (self.super_matrix[:,0] == subject_id) &
            (self.super_matrix[:,2] != trial_id)
        ]

        #Post-Conditions
        if found_elements.size != 0:
            X, y = self._get_X_and_y(found_elements)
        if X is None or y is None:
            logging.error("There are no trials with id {} for subject {} found".format(trial_id, subject_id))
            sys.exit(69)

        logging.debug("<- {} function".format(self._get_intra_session_pretrain_given_sid_tid.__name__))
        return X, y

    def _get_intra_session_train_given_sid_tid(self, subject_id, trial_id):
        logging.debug("-> {} function".format(self._get_cross_session_train_given_subject_id.__name__))
        out = []
        #Take subsection of super_matrix which contains only samples from a specific subject
        all_elements = self.super_matrix[
            (self.super_matrix[:,0] == subject_id) &
            (self.super_matrix[:,2] == trial_id)
        ]

        #Split into cross-validat'able' array
        #Split the array by subjects and extract emgframes and label
        tmp_out = np.split(all_elements, np.where(np.diff(all_elements[:, 1]))[0]+1) #seperating by gestures!
        for ele in tmp_out:
            if ele.size == 0:
                logging.warning("Element has size 0, but will be added to the intra-session training set!")
            tmp = self._get_X_and_y(ele)
            out.append(tmp)

        logging.debug("<- {} function".format(self._get_cross_session_train_given_subject_id.__name__))
        return out


    def get_odd_even_trials(self):
        logging.debug("-> {} function".format(self.get_odd_even_trials.__name__))

        all_samples = np.split(self.super_matrix, np.arange(0, self.super_matrix.shape[0], 1000))
        evens = np.concatenate(all_samples[::2], axis=0)
        odds = np.concatenate(all_samples[1::2], axis=0)

        if evens.shape[0] + odds.shape[0] != self.super_matrix.shape[0]:
            logging.error("Some data was lost, or not all rows have been considered odd or even!")
            sys.exit(69)

        X_evens, y_evens = self._get_X_and_y(evens)
        X_odds, y_odds = self._get_X_and_y(odds)

        logging.debug("<- {} function".format(self.get_odd_even_trials.__name__))
        return X_odds, y_odds, X_evens, y_evens



if __name__ == '__main__':

    dataLoader = DataLoader("Datasets/Preprocessed/DB-a")

    X_pretrain, y_pretrain, X_test, y_test = dataLoader.get_odd_even_trials()
    logging.debug("Pretrain values are X {} and y {}".format(X_pretrain.shape, y_pretrain.shape))
    logging.debug("Train values are X {} and y {}".format(X_test.shape, y_test.shape))







