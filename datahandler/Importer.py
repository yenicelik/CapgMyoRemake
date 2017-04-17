import numpy as np
import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime


#TODO: implement Importer that lets you choose between: Intra-session, Cross-session, Cross-subject
class Importer(object):
    """ All configurations assume that we currently import the CapgMyo dataset. Major work needs to be done if a different type of data is imported.
    """
    """ Impoter is responsible for all preprocessing """

    def __init__(self, datapath, verbose=True):
        """
        :param datapath: The root path that contains all the training data. The training data is assumed to be nested within another folder (2-level nesting)
        :return: An Importer object
        """

        self.data_dirs = self.get_all_data_dirs(datapath, verbose=verbose)
        self.data_list = self.get_data(self.data_dirs, verbose=verbose)

        self.X = self.get_dict_property(self.data_list, "data")
        self.y_tmp = self.get_dict_property(self.data_list, "gesture")

        self.y_tmp = np.asarray(self.y_tmp)
        self.X = np.asarray(self.X)

        self.y_tmp = self.y_tmp.flatten()

        #TODO: change this into a hard-coded style, where additional classes are input depending on how many previous classes exist (a function that counts classes, and returns the number of different classes given a certain dataset
        self.y_tmp[self.y_tmp == 100] = 9
        self.y_tmp[self.y_tmp == 101] = 10

        #Turn into one-hot
        self.y = np.zeros((self.X.shape[0], 12))
        self.y[np.arange(self.X.shape[0]), self.y_tmp] = 1

        for i in range(self.X.shape[0]):
            if np.sum(self.y[i]) != 1:
                print("error!!")

        self.X = np.reshape(self.X, (-1, 1000, 16, 8))


    def get_trainingset(self):
        """
        :return: Returns the entire dataset X, and their respective one-hot labels
        """
        #TODO: change dimension of one-hot vector to adaptable size!
        out_y = np.reshape(self.y, (-1, 1, 12))
        out_y = np.repeat(out_y, 1000, axis=1)
        return self.X, out_y


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


    def get_data(self, datapath, verbose=False):
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
    X = importerObj.get_X()
    importerObj.sample_data()

    print(X.shape)
    print(X)

    y = importerObj.get_y()
    print(y.shape)
    print(y)

if __name__ == "__main__":
    main()