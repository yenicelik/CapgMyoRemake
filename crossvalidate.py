from __future__ import print_function


from model.train import *
from datahandler.OddEvenImporter import OddEvenImporter

import os
import json
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

def crossvalidate_intrasession():

    importer = OddEvenImporter()

    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000,
        'SAVE_DIR': "saves/"
    }

    X_train, y_train, X_test, y_test = importer.get_odd_even()
    tf.reset_default_graph()
    model = init_graph()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        train(X_train=X_train,
          y_train=y_train,
          X_cv=X_test,
          y_cv=y_test,
          cv_accuracy_function=test_model_accuracy,
          parameter=parameter,
          sess=sess,
          model=model)

        accuracy = test_model_accuracy(
                parameter=parameter,
                X=X_test,
                y=y_test,
                sess=sess,
                model_dict=model
        )

        total_accuracy = accuracy * 100
        logging.warning("CV-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(total_accuracy))
        print("Accuracy of the current model on intra-sessions is: {:.3f} percent ".format(total_accuracy))

    logging.debug("<- {} function".format(crossvalidate_intrasession.__name__))
    return accuracy


if __name__ == '__main__':
    crossvalidate_intrasession()