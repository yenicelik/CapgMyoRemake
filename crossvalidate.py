from __future__ import print_function


from model.train import *
from datahandler.OddEvenImporter import OddEvenImporter
from datahandler.CrossSubjectImporter import CrossSubjectImporter

import os
import json
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)
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
    tf.reset_default_graph()
    model = init_graph()

    #Prepare saver
    if parameter['SAVE_DIR']:
        if not os.path.exists(parameter['SAVE_DIR']):
            os.makedirs(parameter['SAVE_DIR'])
        saver = tf.train.Saver()


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        if parameter['LOAD_DIR']:
            saver.restore(sess, parameter['LOAD_DIR'])
        else:
            train(X_train=X_train,
              y_train=y_train,
              X_cv=X_test,
              y_cv=y_test,
              cv_accuracy_function=fn_acc,
              parameter=parameter,
              sess=sess,
              model=model,
              saver=saver,
              savepath=parameter['SAVE_DIR'])

        accuracy = fn_acc(
                parameter=parameter,
                X=X_test,
                y=y_test,
                sess=sess,
                model=model
        )

        total_accuracy = accuracy * 100
        logging.info("CV-Accuracy of the current model on intra-sessions is: {:.3f}% ".format(total_accuracy))
        print("Accuracy of the current model on intra-sessions is: {:.3f} percent ".format(total_accuracy))

    return accuracy



def crossvalidate_crosssubject(fn_acc):
    importer = CrossSubjectImporter()
    cross_subject_dataset = importer.get_seperated_subjects()

    subj_acc_list = []
    for i in range(len(cross_subject_dataset)):
        X_train = np.concatenate([cross_subject_dataset[j][0] for j in range(len(cross_subject_dataset)) if i != j], axis=0)
        y_train = np.concatenate([cross_subject_dataset[j][1] for j in range(len(cross_subject_dataset)) if i != j], axis=0)
        X_train = np.reshape(X_train, (-1, 16, 8))
        y_train = np.reshape(y_train, (-1, 10))
        X_test = cross_subject_dataset[i][0]
        y_test = cross_subject_dataset[i][1]
        X_test = np.reshape(X_test, (-1, 16, 8))
        y_test = np.reshape(y_test, (-1, 10))

        parameter = {
            'LEARNING_RATE': 0.1,
            'NUM_EPOCHS': 28,
            'BATCH_SIZE': 1000,
            'SAVE_DIR': "saves/crosssubject/{}/model.ckpt".format(i),
            'LOAD_DIR': "" #"saves/intrasession/model.ckpt"#""#"saves/intrasession"
        }

        tf.reset_default_graph()
        model = init_graph()

        #Prepare saver
        if parameter['SAVE_DIR']:
            if not os.path.exists(parameter['SAVE_DIR']):
                os.makedirs(parameter['SAVE_DIR'])
            saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            if parameter['LOAD_DIR']:
                saver.restore(sess, parameter['LOAD_DIR'])
            else:
                train(X_train=X_train,
                  y_train=y_train,
                  X_cv=X_test,
                  y_cv=y_test,
                  cv_accuracy_function=fn_acc,
                  parameter=parameter,
                  sess=sess,
                  model=model,
                  saver=saver,
                  savepath=parameter['SAVE_DIR'])

            accuracy = fn_acc(
                    parameter=parameter,
                    X=X_test,
                    y=y_test,
                    sess=sess,
                    model=model
            )

            total_accuracy = accuracy * 100
            logging.info("Leave-one-out-CV-Accuracy of the current model on subject {} is: {:.3f}% ".format(i, total_accuracy))
            print("Leave-one-out-CV-Accuracy of the current model on subject {} is: {:.3f}% ".format(i, total_accuracy))
            subj_acc_list.append(total_accuracy)

    print("Leave-one-out-CV-Accuracy for cross-subject is: {:.3f}% ".format(np.sum(subj_acc_list)/len(subj_acc_list)))

    return np.sum(subj_acc_list)/len(subj_acc_list)



if __name__ == '__main__':
    crossvalidate_intrasession(test_model_accuracy)