import numpy as np
import os
import sys
import scipy.io as sio #use numpy saver and cPickle instead
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_all_data_dirs(directory): #maybe create a MACRO CONFIGURATION file
    """
    :return:
    """

    out = []

    for filename in os.listdir(directory):
        if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
            path = os.path.join(directory, filename)
            out.append(path)

    return out

def get_data(datapath, verbose=False):
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


def get_dict_property(data_arr, property):
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


def play_data(data, framerate, verbose=False):
    """
    :param data: The video file to be display. This must be a single video data (no array!)
    :param framerate: The framerate in seconds per frame
    :return:
    """

    video = []

    for raw_frame in data:
        frame = np.reshape(raw_frame, (16, 8))
        video.append( frame )

    for frame in video:
        #plt.plot(frame)
        if verbose:
            print(frame)
        plt.imshow(frame, cmap='gray')
        plt.pause(framerate)

    raw_input("Press Enter to continue...")

    return


def sample_data(directory="Datasets/Preprocessed/dba-preprocessed-001"):
    """
    Shows you video animation of the simplest data
    :return:
    """

    plt.ion()

    data_dirs = get_all_data_dirs(directory)
    sample_data_dir = np.random.choice(data_dirs) #choose random data dir to display

    data_sample_dict = get_data(sample_data_dir, verbose=True)

    data_sample = get_dict_property(data_sample_dict, "data")[0] #because array-like

    play_data(data_sample, 0.0001)
    return []


def main():
    sample_data()


if __name__ == "__main__":
    main()