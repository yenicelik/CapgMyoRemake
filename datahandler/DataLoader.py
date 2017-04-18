from __future__ import print_function
import numpy as np
from Importer import *

class DataLoader(object):
    """ This function is here to load any kind of
    """

    def __init__(self, super_matrix):
        self.super_matrix = super_matrix



if __name__ == '__main__':

    importerObj = Importer("../Datasets/Preprocessed/DB-a")
    super_matrix = importerObj.get_super_matrix()

    dataLoader = DataLoader(super_matrix)
    print(dataLoader.super_matrix.shape)

