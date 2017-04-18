from __future__ import print_function
import numpy as np
import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime

#TODO: in the batch_loader, we cannot guarantee we always have a satisfying batch_number to not cause an index-error. If the batch-size is a divider of 1000, this shouldn't be a problem for now. Beware of empty batches in case there is no gesture for a specific person
#TODO: implement Importer that lets you choose between: Intra-session, Cross-session, Cross-subject
class Importer(object):
    """ All configurations assume that we currently import the CapgMyo dataset. Major work needs to be done if a different type of data is imported.
    """
    #TODO: Potentially, it could be wise to also have an axis for the frame-order - in case we need to
    """ Impoter is responsible for all preprocessing
        Will finally return an array, the 'super-matrix':
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

    def __init__(self, datapath, verbose=True):
        """
        :param datapath: The root path that contains all the training data. The training data is assumed to be nested within another folder (2-level nesting)
        :return: An Importer object
        """
        file_dirs = self.get_all_data_dirs(datapath, verbose=verbose)
        file_list = self.get_data_from_datapath(file_dirs, verbose=verbose)

        #TODO: get rid of any unnecessary self. structures
        #These lists should have the exact same order. They must! I hope I coded everything correctly, and python doesn't optimize away stuff.
        self.data = self.get_dict_property(file_list, "data")
        print("Data: ", np.asarray(self.data).shape)
        self.data = np.reshape(self.data, (-1, 16 * 8))
        print("Data: ", np.asarray(self.data).shape)
        self.gesture = self.get_dict_property(file_list, "gesture")
        #TODO: change this into a hard-coded style, where additional classes are input depending on how many previous classes exist (a function that counts classes, and returns the number of different classes given a certain dataset
        # self.gesture[self.gesture == 100] = 9
        # self.gesture[self.gesture == 101] = 10
        self.subject = self.get_dict_property(file_list, "subject")
        self.trial = self.get_dict_property(file_list, "trial")

        #to create the above given vector, we must repeat the first three values, if we want to treat the individual videos as frames
        self.super_matrix = np.concatenate((self.gesture, self.subject, self.trial), axis=1)
        print("Super Matrix 1: ", self.super_matrix.shape)
        # now we repeat each of these values 1000 times, because we want to flatten out the videos to frames
        # this results in about O(3%) of all data duplicated, which is accebtable given the convenience of the operations that will follow
        self.super_matrix = np.repeat(self.super_matrix, 1000, axis=0)
        print("Super Matrix 2: ", self.super_matrix.shape)
        self.super_matrix = np.reshape(self.super_matrix, (-1, 3))
        print("Super Matrix 3: ", self.super_matrix.shape)
        # now we want to concatenate the data matrix next to this matrix. Before we can do that, we must flatten the video-format to individual frames (before: videos are samples; now: frames are samples)
        self.super_matrix = np.concatenate((self.super_matrix, self.data), axis=1)
        print("Super Matrix 4: ", self.super_matrix.shape)

        #Turn into one-hot
        # self.y = np.zeros((self.X.shape[0], 12))
        # self.y[np.arange(self.X.shape[0]), self.y_tmp] = 1
        #
        # for i in range(self.X.shape[0]):
        #     if np.sum(self.y[i]) != 1:
        #         print("error!!")




    #######################
    # DATA OUTPUT FUNCTIONS
    def get_super_matrix(self):
        return self.super_matrix


    def get_unclassified_dataset(self):
        """
        In this case, we don't care what subject and trial these gestures are from. As such, we can just spit out the entire dataset not sorted or split by any parameters.
        :return: The tensor of emg video-frames, and their respective gesture label
        """
        return self.X, self.y

    def get_dataset_arr(self, mode):
        """
        Always outputs frames! No videos currently
        :param mode:
        :return:
        """
        if not (mode == "random" or mode == "random" or mode == "cross-subject"):
            print("get_dataset_arr in Importer.py: Please select a valid mode! random, cross-session, cross-subject")
            sys.exit(0)
        if mode == "random":
            pass
        if mode == "cross-session":
            pass
        if mode == "cross-subject":
            pass




    def get_intra_session_dataset(self):
        """
        In this case, we must split the dataset by the ses
        :return:
        """
        pass


    def get_cross_session_dataset(self):
        """
        In this case, we care about which emg video-frames was create by which session.
        :return: Return an array of tensors. Each tensor consists of the emg video-frames and the gesture label. Every two array-elements contains data from different sessions.
        """
        pass


    def get_cross_subject_dataset(self):
        """
        In this case, we care about which emg video-frames was created by whom
        :return: Return an array of tensors. Each tensor consists of the emg video-frames and the gesture label. Every two array-elements contains data from different subjects.
        """
        pass







    #######################
    # DIRECTORY FUNCTIONS
    def get_all_data_dirs(self, directory, verbose=False):
        """
        :param directory: The directory in which all the datasets are filed in.
        :param verbose: Whether all encountered folders should be printed or not.
        :return: An array, containing all paths of files that are saved as the default format in the dataset (CapgMyo has .mat for matlab files)
        """
        out = []

        subdirs = [x[0] for x in os.walk(directory)]
        for folder in subdirs:
            if verbose:
                print(folder)
            for filename in os.listdir(folder):
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    path = os.path.join(folder, filename)
                    out.append(path)

        return out


    def get_data_from_datapath(self, datapath, verbose=False):
        """
        :param datapath: An single datapath, or an array of datapaths.
        :param verbose:
        :return: An array of source-data (usually dictionaries including tensor data) extracted from all files from datapath. More specifically, if datapath is an array of .mat files, this function returns an array of data that was contained in all every file of datapath.
        """

        if len(datapath) == 0:
            print("In function get_data: No datapaths given!")
            sys.exit(10)

        out = []

        if type(datapath) == type([]):
            for filepath in datapath:
                file = sio.loadmat(filepath)
                out.append(file)
        else:
            out = [sio.loadmat(datapath)]

        if verbose:
            print("Data dictionary has the keys: ")
            for key, value in out[0].iteritems():
                print(key)

        return out



    #######################
    # HELPER FUNCTIONS
    def get_dict_property(self, data_arr, property):
        """
        :param data_arr: An array of dictionaries.
        :param property: The property of the dictionary one wants to extract for every element.
        :return: An array of elements, which are retrieved by taking this property from the dictionary in every element of the data_arr-array.
        """
        out = []
        for data in data_arr:
            tmp = data[property]
            out.append( tmp )

        return out

    def pop_dict_property(self, data_arr, property):
        out = []
        for data in data_arr:
            tmp = data
            tmp.pop(property, None)
            out.append( tmp )

        return out



    #######################
    # DATA SAMPLE FUNCTIONS
    def sample_data(self):
        """
        Shows you video animation of a 1000 frame data-sample.
        :return: An empty array
        """
        plt.ion()

        sample_int = np.random.randint(0, self.X.shape[0])
        sample_X = self.X[sample_int, :, :, :]
        sample_y = self.y[sample_int]

        self.play_data(sample_X, sample_y, 0.0001)

        return []


    #TODO: There is a bug in which the .pause command accumulates I think. Debug this shit. Currently, we just show the first 20 frames, but it might be cool to show the full frames at some point
    def play_data(self, data, label, framerate, verbose=False):
        """
        :param data: The sample video file to be displayed.
        :param label: The label of the sample video file. Will be displayed in the title of the video.
        :param framerate: The number of frames per second to be display.
        :param verbose: Whether the tensorf of each frame should be displayed or not.
        :return: An empty array.
        """

        video = []

        for raw_frame in data:
            frame = np.reshape(raw_frame, (16, 8))
            video.append( frame )

        fig = plt.gcf()

        fig.canvas.set_window_title("Gesture: " + str(np.argmax(label)))

        itersteps = 0
        for frame in video:
            itersteps += 1
            if itersteps >= 20:
                break

            if verbose:
                start_time = datetime.datetime.now()

            plt.imshow(frame, cmap='gray')
            #time.sleep(framerate)
            plt.pause(1./framerate)
            plt.draw()

            if verbose:
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                print("total_time:")
                print(total_time)


        raw_input("Press Enter to continue...")

        return []



def main():
    importerObj = Importer("Datasets/Preprocessed/DB-a")

    X, y = importerObj.get_trainingset()

if __name__ == "__main__":
    main()