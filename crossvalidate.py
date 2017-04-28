from __future__ import print_function

import numpy as np
from metrics.AccuracyTester import test_model_accuracy_voting, test_model_accuracy
from datahandler.DataLoader import DataLoader

import logging
logging = logging.getLogger(__name__)


def crossvalidate_intrasession(parameter, dataLoader, is_voting):
    logging.debug("-> {} function".format(crossvalidate_intrasession.__name__))

    #TODO: for intra-session, how they did the stuff is that they use half the data as training, the rest as testing
    #We have enough data here, so we don't need to apply leave-one-out CV
    X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    print("Entering train function with")
    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("Leaving train function")

    if is_voting:
        logging.debug("Voting is used for accuracy testing")
        accuracy = test_model_accuracy_voting(
                    parameter=parameter,
                    X=X_test,
                    y=y_test,
                    voting_window=1000,
                    # sess=sess,
                    # model_dict=model_dict,
            )
    else:
        logging.debug("Not using voting")
        accuracy = test_model_accuracy(
                X=X_test,
                y=y_test,
                parameter=parameter,
                show_confusion_matrix=False,
                # sess=sess,
                # model_dict=model_dict
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

    for i in range(len(cross_subject_dataset)):
        logging.info("Now crossvalidating by leaving out subject number {}".format(i))
        X_cv, y_cv = cross_subject_dataset[i]
        train_set = [cross_subject_dataset[j] for j in range(len(cross_subject_dataset)) if j != i]
        X_train = np.concatenate([x for x, y in train_set], axis=0)
        #X_train = np.reshape(X_train, (-1, 16, 8)) #TODO: if we have to do this, consider moving this into the tensorflow model!
        y_train = np.concatenate([y for x, y in train_set], axis=0)

        #TODO: implement saver of the individual subject model (can save time for later re-initilization
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

    total_accuracy = float(sum(accuracy_list))/len(cross_subject_dataset) * 100
    logging.warning("CV-Accuracy of the current model on cross-subject is: {:.3f}% ".format(total_accuracy))
    print("Accuracy of the current model on cross-subject is: {:.3f} percent ".format(total_accuracy))

    logging.debug("<- {} function".format(crossvalidate_subjects.__name__))
    return accuracy_list


if __name__ == '__main__':
    dataLoader = DataLoader("datahandler/Datasets/Preprocessed/DB-a")

    parameter = {
        'BATCH_SIZE': 500
    }
    crossvalidate_intrasession(parameter, dataLoader, is_voting=True)