from __future__ import print_function

import numpy as np
from metrics.AccuracyTester import test_model_accuracy_voting, test_model_accuracy
from datahandler.DataLoader import DataLoader
from model.BuildGraph import *
from model.train import *

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


def crossvalidate_intrasession(parameter, dataLoader, is_voting):
    logging.debug("-> {} function".format(crossvalidate_intrasession.__name__))
    #TODO: DO NOT APPLY is_voting=False here just yet!!

    X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    parameter['SAVE_DIR'] = "saves/intrasession"
    W, b, model_dict= init_graph()

    logging.info("Entering session")
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #Setting up the parameters
        features = W.copy()
        features.update(b)
        features.update({'global_step': model_dict['globalStepTensor']})
        TfSaver = TFSaver(parameter['SAVE_DIR'], features)

        #Training
        train(X_train[:1000], y_train[:1000], parameter, sess=sess, model_dict=model_dict, saverObj=TfSaver)

        if is_voting:
            logging.debug("Voting is used for accuracy testing")
            accuracy = test_model_accuracy_voting(
                    parameter=parameter,
                    X=X_test,
                    y=y_test,
                    voting_window=1000,
                    sess=sess,
                    model_dict=model_dict
            )
        else:
            logging.debug("Not using voting")
            accuracy = test_model_accuracy(
                    X=X_test,
                    y=y_test,
                    parameter=parameter,
                    show_confusion_matrix=False,
                    sess=sess,
                    model_dict=model_dict
            )

        total_accuracy = accuracy * 100
        logging.warning("CV-Accuracy of the current model on cross-subject is: {:.3f}% ".format(total_accuracy))
        print("Accuracy of the current model on cross-subject is: {:.3f} percent ".format(total_accuracy))

    logging.debug("<- {} function".format(crossvalidate_intrasession.__name__))
    return accuracy


def crossvalidate_crosssession(parameter, dataLoader, is_voting):
    logging.debug("-> {} function".format(crossvalidate_crosssession.__name__))

    accuracy_list = []
    #computationally complexity is similar to cross-subject, so it is acceptable to iterate over all subjects
    for sid in dataLoader.subjects:
        #Pretraining
        logging.info("CV crosssession entered for subject number {}".format(sid))

        X_pretrain, y_pretrain, cross_session_dataset = dataLoader.get_cross_session_dataset_given_subject_id(sid)
        print("Entering pre-train function with")
        print("X_pretrain: {}".format(X_pretrain.shape))
        print("y_pretrain: {}".format(y_pretrain.shape))
        print("Leaving pre-train function")

        #TODO: Need to save model

        for i in range(len(cross_session_dataset)):
            logging.debug("CV crosssession entered for subject number {} in trial/session {}".format(sid, i))
            X_cv, y_cv = cross_session_dataset[i]
            train_set = [cross_session_dataset[j] for j in range(len(cross_session_dataset)) if j != i]
            X_train = np.concatenate([x for x, y in train_set], axis=0)
            y_train = np.concatenate([y for x, y in train_set], axis=0)


            print("Entering train function with")
            print("X_train: {}".format(X_train.shape))
            print("y_train: {}".format(y_train.shape))
            print("X_cv: {}".format(X_cv.shape))
            print("y_cv: {}".format(y_cv.shape))
            print("Leaving train function")


            if is_voting:
                logging.debug("Voting is used for accuracy testing")
                accuracy = test_model_accuracy_voting(
                        parameter=parameter,
                        X=X_cv,
                        y=y_cv,
                        voting_window=1000,
                        # sess=sess,
                        # model_dict=model_dict,
                )
            else:
                logging.debug("Not using voting")
                accuracy = test_model_accuracy(
                        X=X_cv,
                        y=y_cv,
                        parameter=parameter,
                        show_confusion_matrix=False,
                        # sess=sess,
                        # model_dict=model_dict
                )

            accuracy_list.append(accuracy)

    total_accuracy = float(sum(accuracy_list))/len(accuracy_list) * 100
    logging.warning("CV-Accuracy of the current model on cross-subject is: {:.3f}% ".format(total_accuracy))
    print("Accuracy of the current model on cross-subject is: {:.3f} percent ".format(total_accuracy))

    logging.debug("<- {} function".format(crossvalidate_crosssession.__name__))
    return accuracy_list



def crossvalidate_subjects(parameter, dataLoader, is_voting):
    logging.debug("-> {} function".format(crossvalidate_subjects.__name__))
    cross_subject_dataset = dataLoader.get_cross_subject_dataset()
    accuracy_list = []

    W, b, model_dict= init_graph()
    old_para = parameter

    for i in range(len(cross_subject_dataset)):
        logging.info("Now crossvalidating by leaving out subject number {}".format(i))
        X_cv, y_cv = cross_subject_dataset[i]
        train_set = [cross_subject_dataset[j] for j in range(len(cross_subject_dataset)) if j != i]
        X_train = np.concatenate([x for x, y in train_set], axis=0)
        y_train = np.concatenate([y for x, y in train_set], axis=0)

        logging.info("Entering session")
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            #Setting up the parameters
            features = W.copy()
            features.update(b)
            features.update({'global_step': model_dict['globalStepTensor']})

            parameter = old_para
            parameter['SAVE_DIR'] = "saves/subject_{}".format(i)
            TfSaver = TFSaver(parameter['SAVE_DIR'], features)

            #Training
            train(X_train, y_train, parameter, sess=sess, model_dict=model_dict, saverObj=TfSaver)

            if is_voting:
                logging.debug("Voting is used for accuracy testing")
                accuracy = test_model_accuracy_voting(
                        parameter=parameter,
                        X=X_cv,
                        y=y_cv,
                        voting_window=1000,
                        sess=sess,
                        model_dict=model_dict,
                )
            else:
                logging.debug("Not using voting")
                accuracy = test_model_accuracy(
                        X=X_cv,
                        y=y_cv,
                        parameter=parameter,
                        show_confusion_matrix=False,
                        sess=sess,
                        model_dict=model_dict
                )

            accuracy_list.append(accuracy)

        total_accuracy = float(sum(accuracy_list))/len(cross_subject_dataset) * 100
        logging.warning("CV-Accuracy of the current model on cross-subject is: {:.3f}% ".format(total_accuracy))
        print("Accuracy of the current model on cross-subject is: {:.3f} percent ".format(total_accuracy))

    logging.debug("<- {} function".format(crossvalidate_subjects.__name__))
    return accuracy_list


if __name__ == '__main__':
    setup_logging()
    dataLoader = DataLoader("datahandler/Datasets/Preprocessed/DB-a")

    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000,
        'SAVE_DIR': "saves/"
    }
    crossvalidate_intrasession(parameter, dataLoader, is_voting=True)