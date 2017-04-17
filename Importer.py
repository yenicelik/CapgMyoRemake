import numpy as np
import os
import sys
import scipy.io as sio #use numpy saver and cPickle instead
import matplotlib.pyplot as plt
import time
import datetime


#TODO: implement Importer that lets you choose between: Intra-session, Cross-session, Cross-subject
class Importer(object):
    """ Impoter is responsible for all preprocessing """

    def __init__(self, datapath, sample=False):
        """
        :param datapath: Is the root path that contains all the training data.
        :param sample: Whether one example of how the data looks should be given
        :return:
        """

        self.data_dirs = self.get_all_data_dirs(datapath, verbose=True)
        self.data_list = self.get_data(self.data_dirs, verbose=False)
        self.X = self.get_dict_property(self.data_list, "data")
        self.y_tmp = self.get_dict_property(self.data_list, "gesture")

        self.y_tmp = np.asarray(self.y_tmp)
        self.X = np.asarray(self.X)

        self.y_tmp = self.y_tmp.flatten()

        self.y_tmp[self.y_tmp == 100] = 9
        self.y_tmp[self.y_tmp == 101] = 10

        self.y = np.zeros((self.X.shape[0], 12))
        self.y[np.arange(self.X.shape[0]), self.y_tmp] = 1


        #y is fine!

        for i in range(self.X.shape[0]):
            if np.sum(self.y[i]) != 1:
                print("error!!")

        self.X = np.reshape(self.X, (-1, 1000, 16, 8)) #Assume 1000 is the batch-size now..


    def get_trainingset(self):
        out_y = np.reshape(self.y, (-1, 1, 12))
        out_y = np.repeat(out_y, 1000, axis=1)

        #seems to work aswell!

        return self.X, out_y


    def get_all_data_dirs(self, directory, verbose=False): #maybe create a MACRO CONFIGURATION file
        """
        :param directoy: The directy in which all the datasets are filed in..
        :return:
        """
        out = []

        subdirs = [x[0] for x in os.walk(directory)]
        for folder in subdirs:
            print(folder)
            for filename in os.listdir(folder):
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    path = os.path.join(folder, filename)
                    out.append(path)

        return out


    def get_data(self, datapath, verbose=False):
        """
        :param datapath: single or arraylike
        :return:
        """

        if len(datapath) == 0:
            print("In function get_data: No datapaths given!")
            sys.exit(10)

        out = []

        if type(datapath) == type([]):
            for filepath in datapath:
                file = sio.loadmat(filepath)
                out.append(file) #I assume arrays are not gonna cause a problem here
        else:
            out = [sio.loadmat(datapath)]

        if verbose:
            print("Data dictionary has the keys: ")
            for key, value in out[0].iteritems():
                print(key)


        return out


    def get_dict_property(self, data_arr, property):
        """
        :param data_arr:
        :param property:
        :return:
        """
        out = []
        for data in data_arr:
            tmp = data[property]
            out.append( tmp )

        return out


    def sample_data(self):
        """
        Shows you video animation of the simplest data. This shows you 1000frames, although the model is trained independent of the sequential frames
        :return:
        """

        plt.ion()

        sample_int = np.random.randint(0, self.X.shape[0])
        sample_X = self.X[sample_int, :, :, :]
        sample_y = self.y[sample_int]

        self.play_data(sample_X, sample_y, 0.0001)

        return []


    def play_data(self, data, label, framerate, verbose=False):
        """
        :param data: The video file to be display. This must be a single video data (no array!)
        :param framerate: The framerate in seconds per frame
        :return:
        """
        #TODO: I feel like there is a problem with the matlab.pause operator... it accumulates i think.. find out what is wrong with this

        video = []

        for raw_frame in data:
            frame = np.reshape(raw_frame, (16, 8))
            video.append( frame )

        fig = plt.gcf()

        fig.canvas.set_window_title("Gesture: " + str(np.argmax(label)))


        itersteps = 0 #TODO: This is currently a bug, so only display the first few frames
        for frame in video:
            itersteps += 1
            if itersteps >= 20:
                break

            if verbose:
                start_time = datetime.datetime.now()

            plt.imshow(frame, cmap='gray')
            #time.sleep(framerate)
            plt.pause(.001)
            plt.draw()

            if verbose:
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                print("total_time:")
                print(total_time)


        raw_input("Press Enter to continue...")

        return



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