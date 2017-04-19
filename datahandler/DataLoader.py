from __future__ import print_function
import numpy as np
from Importer import *

#TODO: These functions feel incredibly naive.. Have a look at them later to make them finer / possibly faster.
class DataLoader(object):
    """ This function is here to load any kind of
        All calculations will be based on the the 'super-matrix':
           0  1  2  3  4  5  6  7  ...  n
        -----------------------------------
        |  s  g  t  d  d  d  d  ...  d
        |  s  g  t  d  d  d  d  ...  d
        |  s  g  t  d  d  d  d  ...  d
        |  s  g  t  d  d  d  d  ...  d
        |  s  g  t  d  d  d  d  ...  d
        ...
        Where:
         s is the respective subject ID
         g is the respective gesture ID
         t is the respective trial
         d is the respective frame (16 * 8)
         potentially, we should also have a 'frame_number' id that determines the order of the frame within this gesture and session
    """

    def __init__(self, super_matrix):
        self.sm = super_matrix
        self.no_of_subjects = np.amax(self.sm[:, 0]).astype(int) #there might be inconsistencies in the data, so we must loop from minimum to maximum, and append to the output anything that is fitting.
        self.no_of_gesture = np.amax(self.sm[:, 1]).astype(int)
        self.no_of_trials = np.amax(self.sm[:, 2]).astype(int)
        print("Subjects: ", self.no_of_subjects)
        print("Gestures: ", self.no_of_gesture)
        print("Trials: ", self.no_of_trials)

    #TODO: I think i messed up some indices here, have a look at this. Otherwise, there is an error in the logic/understanding in sematics of each operation
    def get_cross_subject_dataset(self):
        """
        This feels solved.
        For cross-subject, there is no variable that must be kept constant, as we can switch between sessions, gestures, and subjects.
        :return: An array of tuples (X, y). Each array-element represents one subject. X are all the input-frames. y are all the corresponding labels.
        """
        out = [] #The order of the subjects doesn't matter. these could even be anonymous. As such, we don't need i elements for the output!

        print("Entering cross subject function")
        for i in range(self.no_of_subjects):
            found_elements = self.sm[
                                        (self.sm[:, 0] == i)
            ]
            if (found_elements.size != 0):
                tmp = self.get_X_and_y(found_elements, mode="subject")
                out.append(tmp)

        return out

    def get_random_cross_session_dataset(self):
        random_person = np.random.choice(self.sm[:,0])
        out = self.get_cross_session_dataset_given_subject_id(random_person)
        return out #Of course this idea can be expanded to a 2D array incorporating multiple people... currently, we just pick a random person however... Best: this should be done for multiple people.

    #This is to some extent a pseduo-cross-session test in which we pre-train on the given data, and test on the chosen data
    def get_cross_session_pretrain_dataset(self, subject_id):
        """
        Gets the dataset on which we pretrain the cross-session tests. This is not pure cross session, but good enough.
        :param subject_id: The subject id on which we will perform cross-session.
        :return: Return a tuple of the form (X, y), where X is a tensor of the frames and y contains the corresponding gesture labels.
        """
        #first of all, fill all the values with everything except the subject. We will then iterate through all subjects. All the data should be initialliy pre-trained on this.
        found_elements = self.sm[ (self.sm[:, 0] != subject_id) ]
        if (found_elements.size != 0):
            out = self.get_X_and_y(found_elements, mode="trial")

        return out

    def get_cross_session_dataset_given_subject_id(self, subject_id):
        """
        For cross-session, the subject must obviously stay constant. The trial must also stay constant. As such, we can iterate over multiple trials. We must keep the subject constant.
        Maybe an array of arrays is better suitable?
        :return:
        """
        out = [] #The order of the subjects doesn't matter. these could even be anonymous. As such, we don't need i elements for the output!

        print("Entering cross session function")
        for i in range(self.no_of_trials):              #we must iterate over these values, because we don't always have the full range covered by the dataset. We take what we can get
            found_elements = self.sm[
                                        (self.sm[:, 2] == i) &
                                        (self.sm[:, 0] == subject_id)
            ]
            if (found_elements.size != 0):
                tmp = self.get_X_and_y(found_elements, mode="trial")
                out.append(tmp)

        return out

    def get_random_intra_session_dataset(self):
        random_person = np.random.choice(self.sm[:,0])
        person_trial =  self.sm[ (self.sm[:,0] == random_person) ]
        random_trial = np.random.choice( person_trial[:, 2] )
        out = self.get_intra_session_dataset_given_subject_and_trial(
                                        random_person,
                                        random_trial
        )
        return out #Of course this idea can be expanded to a 2D array incorporating multiple people... currently, we just pick a random person however... Best: this should be done for multiple people.


    def get_intra_session_dataset_given_subject_and_trial(self, subject_id, trial_id):
        """
        :param subject_id:
        :param trial_id:
        :return:
        """
        out = [] #The order of the subjects doesn't matter. these could even be anonymous. As such, we don't need i elements for the output!

        print("Entering intra session function")
        for i in range(self.no_of_gesture):
            found_elements = self.sm[
                                        (self.sm[:, 1] == i) &
                                        (self.sm[:, 0] == subject_id) &
                                        (self.sm[:, 2] == trial_id)
            ]
            if (found_elements.size != 0):
                tmp = self.get_X_and_y(found_elements, mode="gesture")
                out.append(tmp)

        return out


    ###################
    # HELPER FUNCTIONS
    def get_X_and_y(self, inp_matrix, mode):
        """
        :return: Returns the entire dataset X, and their respective one-hot labels
        """
        #TODO: change dimension of one-hot vector to adaptable size!
        tmp_y = inp_matrix[:, 1].astype(int) #seems like a really unsafe operation..
        X = inp_matrix[:, 3:]

        #TODO: Is this error-prone enough? Also, this shouldn't be necessarily hardcoded, but it seems like this is a property of the dataset, so not sure what else to do
        range = len(np.unique(self.sm[:, 1]))
        tmp_y[tmp_y == 100] = range-1
        tmp_y[tmp_y == 101] = range

        #Turn into one-hot
        y = np.zeros((tmp_y.shape[0], range))
        y[np.arange(y.shape[0]), np.subtract(tmp_y, 1)] = 1

        for i in xrange(y.shape[0]):
            if np.sum(y[i, :]) != 1:
                print("error!! with one-hot encoding!")

        return X, y




if __name__ == '__main__':

    importerObj = Importer("../Datasets/Preprocessed/DB-a")
    super_matrix = importerObj.get_super_matrix()

    dataLoader = DataLoader(super_matrix)
    cross_subject_dataset = dataLoader.get_cross_subject_dataset()
    cross_session_dataset = dataLoader.get_random_cross_session_dataset()
    intra_session_dataset = dataLoader.get_random_intra_session_dataset()

    print("Data loader: ", dataLoader.sm.shape)
    print("Cross subject elements: ", len(cross_subject_dataset))
    print("Cross session elements: ", len(cross_session_dataset))
    print("Intra session elements: ", len(intra_session_dataset))