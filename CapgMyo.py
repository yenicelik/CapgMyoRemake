from __future__ import print_function

from metrics.AccuracyTester import *
from model.BuildGraph import *
from train import *

import logging
logging = logging.getLogger(__name__)

import os
import json
import logging.config

def setup_logging(
    default_path='logging.json',
    default_level=logging.DEBUG,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


#TODO: Build tensorflow from source (Assembly instruction will make it faster by 3-8 times
def main(restore, parameter, full_train=False, go_train=True):

    ################################
    # Importing all data
    ################################
    logging.debug("Importing datasets")
    importerDBA = Importer("Datasets/Preprocessed/DB-a")
    X, y = importerDBA.get_trainingset()

    #for this specific task, we need to get X and y as frames. Make sure these are not all from the same class
    X = np.reshape(X, (-1, 16, 8))
    y = np.reshape(y, (-1, 10))
    y = np.concatenate((y, np.zeros((y.shape[0], 2))), axis=1)

    logging.debug("X has shape: {}".format(X.shape))
    logging.debug("y has shape: {}".format(y.shape))

    biggest_size = X.shape[0]
    sample_cases = 10000
    if full_train:
        verify_size = X.shape[0] - sample_cases
    else:
        verify_size = 100000


    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    #Very simple test case
    X_verify = X[indices[sample_cases:X.shape[0]], :, :] #should be X_train
    y_verify = y[indices[sample_cases:X.shape[0]], :]

    X_sample = X[indices[:sample_cases], :, :]
    y_sample = y[indices[:sample_cases], :]

    #################################
    # Initializing TensorFlow Graph
    #################################
    tf.global_variables_initializer()
    W, b, model_dict = init_graph()
    saverObj = Saver(parameter['SAVE_DIR'])

    if not restore:
        init = tf.initialize_all_variables()
        

    ################################
    # Training on data
    ################################
    with tf.Session() as sess:

        if restore:
            saverObj.load_session(sess, parameter['SAVE_DIR']) #Do I need anything else? Like to add the global stuff etc.?
            logging.info("Model should be restored now")
        else:
            sess.run(init)
            logging.info("New model should be initialized now")

        if go_train:
            train(
                        sess=sess,
                        parameter=parameter,
                        model_dict=model_dict,
                        X=X_verify,
                        y=y_verify,
                        saverObj=saverObj
            )

        test_accuracy(
                    sess=sess,
                    model_dict=model_dict,
                    parameter=parameter,
                    X=X_sample,
                    y=y_sample,

        )



if __name__ == '__main__':

    # The model_dict we will use
        # model_dict = {
        #                 "X_input": X_input,
        #                 "y_input": y_input,
        #                 "loss": loss,
        #                 "predict": predict,
        #                 "trainer": trainer,
        #                 "updateModel": updateModel
        # }

    setup_logging()

    restore = False #wether we want to use the model saved in 'saves/', or start training a model from scratch.

    paper_parameter = {
            'NUM_EPOCHS': 28, #determine what these values should be. Cross validation can take the same time of training we had for 3h, but applied on every subject / potentially on every session etc.
            'BATCH_SIZE': 1000,
            'SAVE_EVERY': 300, #number of batches after which to save,
            'LEARNING_RATE': 0.1,
            'SAVE_DIR': "save/"
    }


    main(restore, paper_parameter, full_train=True, go_train=True)