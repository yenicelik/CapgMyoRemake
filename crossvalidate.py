from __future__ import print_function

from misc import *
from datahandler.OddEvenImporter import OddEvenImporter
from keras.models import load_model
from config import *
from accuracy import *

import os
import json
import logging
logging.basicConfig(filename=(RUN_NAME + '.log'), level=logging.DEBUG)
logging = logging.getLogger(__name__)

def crossvalidate_intrasession(fn_acc):

    importer = OddEvenImporter()

    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 500,
        'SAVE_DIR': "saves/intrasession/model.ckpt",
        'LOAD_DIR': "" #"saves/intrasession/model.ckpt" #""
    }

    X_train, y_train, X_test, y_test = importer.get_odd_even()
    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format(y_test.shape))
    # tf.reset_default_graph()

    #Prepare saver
    if parameter['SAVE_DIR']:
        if not os.path.exists(parameter['SAVE_DIR']):
            os.makedirs(parameter['SAVE_DIR'])

    if parameter['LOAD_DIR']:
        model = load_model(parameter['LOAD_DIR'])
        model.summary()
        logging.info("Model loaded")
        print("Loaded model")
    else:
        model = init_graph()
        model.summary()
        for e in range(parameter['NUM_EPOCHS']):
            nlr = adapt_lr(e, parameter['LEARNING_RATE'])
            model.optimizer.lr.assign(nlr)
            logging.info("Epoch {} with lr {:.3f}".format(e, nlr))
            print("Epoch {} with lr {:.3f}".format(e, nlr))

            history = model.fit(x=X_train, y=y_train, batch_size=500, epochs=1, shuffle=True, validation_data=(X_test, y_test))
            train_accuracy = history.history['acc'][-1]
            test_accuracy = history.history['val_acc'][-1]

            logging.info("Train-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(train_accuracy))
            print("Train-Accuracy of the current model on intra-sessions is: {:.3f} percent ".format(train_accuracy))

            logging.info("CV-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(test_accuracy))
            print("Accuracy of the current model on intra-sessions is: {:.3f} percent ".format(test_accuracy))

            model.save(os.path.join(parameter['SAVE_DIR'], RUN_NAME + '.h5'))
            logging.debug("Saved model")

    return test_accuracy


if __name__ == '__main__':
    crossvalidate_intrasession(test_model_accuracy)