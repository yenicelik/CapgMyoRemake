import numpy as np
import tensorflow as tf
from Model import init_graph


def main():
    tf.reset_default_graph()
    W, b, forward_dict, loss_dict = init_graph()

    # forward_dict = {
    #                   "X_input",
    #                   "y_input",
    #               }

    # loss_dict = {"nextQ":
    #             "loss",
    #             "trainer",
    #             "updateModel"
    #             }

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