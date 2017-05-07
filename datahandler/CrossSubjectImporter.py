from __future__ import print_function

import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

class CrossSubjectImporter(object):

    def __init__(self, data_parent_dir="datahandler/Datasets/Preprocessed/DB-a"): #datahandler/
        filepaths = self.get_filepaths_in_directory(data_parent_dir)
        subject_paths = self.get_subject_paths(filepaths)

        data_arr = []
        for sid_paths in subject_paths.values():
            data = self.get_data_from_filepaths(sid_paths)
            X, y = self.get_X_y_from_data(data)
            data_arr.append((X, y))

        self.seperated_subjects = data_arr

    def get_seperated_subjects(self):
        return self.seperated_subjects


    def get_subject_paths(self, filepaths):
        out = {i: [] for i in range(18)}

        for filepath in filepaths:
            sid = int(filepath[-15:-12])
            out[sid % 18].append(filepath)

        return out


    def get_filepaths_in_directory(self, parent_directory):
        out = []
        for path, _, files in os.walk(parent_directory):
            for filename in files:
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    logging.debug("Found file {}".format(filename))
                    filepath = os.path.join(path, filename)
                    out.append(filepath)
        if len(out) == 0:
            print("Error! No files found in given datapath!")
            print(os.getcwd(), parent_directory)
        return out


    def get_data_from_filepaths(self, filepaths):
        if len(filepaths) == 0:
            logging.error("No filepaths are given! Quitting from get_data_from_filepath")
            logging.error(os.getcwd())
            sys.exit(69)
        if type(filepaths) != type([1, 2]):
            logging.error("Filepaths is not an array!")
            sys.exit(69)

        out = []
        for filepath in filepaths:
            filedata = sio.loadmat(filepath)
            out.append(filedata)

        return out


    def get_X_y_from_data(self, data_arr):
        ys = []
        Xs = []

        for ele in data_arr:
            X = ele['data']
            #Turning into one-hot
            gesture = ele['gesture']
            if gesture == 101:
                gesture = 10
            if gesture == 100:
                gesture = 9
            y = np.zeros((X.shape[0], 10)) #10 dimensions in one-hot setting
            y[:,gesture-1] = 1
            #splitting array now again, feeling a little unconfident here, as off-by-one error are critical
            ys.append(y)
            Xs.append(X)

        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

