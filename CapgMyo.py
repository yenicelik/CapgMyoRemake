import numpy as np
import tensorflow as tf
from Model import init_graph
from Importer import *
from train import train
from AccuracyTester import *


def main():

    #################################
    # Initializing TensorFlow Graph
    #################################

    tf.global_variables_initializer()
    W, b, model_dict = init_graph()

    # model_dict = {
    #                 "X_input": X_input,
    #                 "y_input": y_input,
    #                 "loss": loss,
    #                 "predict": predict,
    #                 "trainer": trainer,
    #                 "updateModel": updateModel
    # }


    ################################
    # Importing all data
    ################################

    #I think we ignore which subjects these are from in the first run..

    importerDBA = Importer("Datasets/Preprocessed/DB-a")
    X, y = importerDBA.get_trainingset()

    print(X.shape)
    print(y.shape)

    #for this specific task, we need to get X and y as frames.... as such, it might be wise to do batches of size 1000, let each have a label of y... otherwise, it is not good to input a homogeneous dataset i thinkg
    X = np.reshape(X, (-1, 16, 8))[:5000, :, :]
    y = np.reshape(y, (-1, 32))[:5000, :]

    ################################
    # Training on data
    ################################
    parameter = {
        'NUM_EPOCHS': 1,
        'BATCH_SIZE': 5000
    }

    train(
                parameter=parameter,
                model_dict=model_dict,
                X=X,
                y=y
    )

    test_accuracy(
                model_dict=model_dict,
                parameter=parameter,
                X=X,
                y=y
    )



if __name__ == '__main__':
    #TODO: Set up the dictionary, or potentially the terminal read-in from here
    main()