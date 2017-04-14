import numpy as np
import tensorflow as tf
from Model import init_graph
from Importer import *


def main():

    #################################
    # Initializing TensorFlow Graph
    #################################

    # tf.reset_default_graph()
    # W, b, model_dict = init_graph()

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

    data_dirs = get_all_data_dirs("Datasets/Preprocessed/dba-preprocessed-001")
    data_dict = get_data(data_dirs, verbose=False)
    X = get_dict_property(data_dict, "data")
    y = get_dict_property(data_dict, "gesture")

    X = np.asarray(X)
    y = np.asarray(y)

    print(X.shape)
    print(y.shape)

    # if is_training:
    #     reward_list, steps_list = train(
    #                                 env=env,
    #                                 parameter=parameter,
    #                                 saver=saver,
    #                                 forward_dict=forward_dict,
    #                                 loss_dict=loss_dict
    #                             )
    # if is_showoff:
    #     sample()


if __name__ == '__main__':
    #TODO: Set up the dictionary, or potentially the terminal read-in from here
    main()