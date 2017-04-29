from __future__ import print_function
import tensorflow as tf
import os
import sys
import numpy as np

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

from model.BuildGraph import *

#TODO: How can we individually save and restore weights?
class TFSaver(object):

    def __init__(self, save_dir, model_vars):

        try:
            if model_vars == None:
                self.saver = tf.train.Saver(model_vars)
            else:
                self.saver = tf.train.Saver()
        except Exception as e:
            print("Failureee !")
            print(e)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir


    def get_saver(self):
        return self.saver

    def save_session(self, sess, name="", global_step=None):
        logging.debug("-> {} function".format(self.save_session.__name__))
        checkpoint_path = os.path.join(self.save_dir, name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, 'model.ckpt')
        checkpoint_path = os.path.join("./", checkpoint_path)

        success = False
        try:
            save_path = self.saver.save(sess, checkpoint_path, global_step=global_step)
            success = True
            logging.info("Model should be saved at {}".format(os.path.join(os.getcwd(), save_path)))
            logging.info("Model saved at {}".format(os.path.join(os.getcwd(), save_path)))
        except Exception as e:
            logging.error(e)
            logging.error("Could not save model into {}".format(os.path.join(os.getcwd(), checkpoint_path)))
            sys.exit(69)

        logging.debug("<- {} function".format(self.save_session.__name__))
        return success


    def load_session(self, sess, name=""):
        logging.debug("-> {} function".format(self.load_session.__name__))
        checkpoint_path = os.path.join(self.save_dir, name)
        #checkpoint_path = os.path.join(checkpoint_path, 'model.ckpt')
        checkpoint_path = os.path.join("./", checkpoint_path)
        logging.debug("Absolute checkpoint path: {}".format(os.path.join(os.getcwd(), checkpoint_path)))
        try:
            self.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            logging.info("Model restored from {}".format(checkpoint_path))
        except Exception as e:
            logging.error("Could not load model {} with modelname {}; currently in directory {}. The exception message is: ".format(checkpoint_path, name, os.getcwd()))
            logging.error(e)
            sys.exit(69)

        logging.debug("<- {} function".format(self.load_session.__name__))
        return True



if __name__ == '__main__':
    # dataLoader = DataLoader("Datasets/Preprocessed/DB-a")
    # X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    X_train = np.random.rand(7000, 16, 8)
    y_train = np.random.rand(7000, 10)
    X_test = np.random.rand(7000, 16, 8)
    y_test = np.random.rand(7000, 10)

    parameter = {
        'LEARNING_RATE': 500,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000
    }


    W, b, model_dict, tmpval = init_graph()
    init_op = tf.global_variables_initializer()

    features = W.copy()
    features.update(b)
    features.update({'global_step': model_dict['globalStepTensor']})
    tf_saver = TFSaver("tmpsaver/", features)

    # logging.debug("Running and saving session")
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     loss, predict, _, global_step = sess.run(
    #         # Describe what we want out of the model
    #         [
    #             model_dict['loss'],
    #             model_dict['predict'],
    #             model_dict['updateModel'],
    #             model_dict['globalStepTensor']
    #
    #         ],
    #         # Describe what we input in the model
    #         feed_dict={
    #             model_dict['X_input']: X_train[:2000, :],
    #             model_dict['y_input']: y_train[:2000],
    #             model_dict['keepProb']: 0.5,
    #             model_dict['learningRate']: parameter['LEARNING_RATE'],
    #             model_dict['isTraining']: True
    #         }
    #     )
    #     print("Global step before: {}".format(global_step))
    #     logging.debug("Run on data successful")
    #     tf_saver.save_session(sess, name="sid", global_step=global_step)
    #     logging.debug("Saving done")



    logging.debug("Loading and Running session")
    with tf.Session() as sess:
        logging.debug("Entering session")
        tf_saver.load_session(sess, name="sid")
        logging.debug("Restore done")
        loss, predict, _, global_step = sess.run(
            # Describe what we want out of the model
            [
                model_dict['loss'],
                model_dict['predict'],
                model_dict['updateModel'],
                model_dict['globalStepTensor']

            ],
            # Describe what we input in the model
            feed_dict={
                model_dict['X_input']: X_train[:2000, :],
                model_dict['y_input']: y_train[:2000],
                model_dict['keepProb']: 0.5,
                model_dict['learningRate']: parameter['LEARNING_RATE'],
                model_dict['isTraining']: True
            }
        )
        print("Global step after: {}".format(global_step))
        tf_saver.save_session(sess, name="sid", global_step=global_step)
        logging.debug("Run on data successful")

