from __future__ import print_function

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)

from misc import *
from datahandler.Importer import Importer
from datahandler.BatchLoader import BatchLoader
from build_graph import *

import logging
logging.basicConfig(filename='naive_full.log',level=logging.DEBUG)
logging = logging.getLogger(__name__)


def main():
    #####################################
    # IMPORT DATA; TRAIN W/ ODD; TEST W/ EVEN
    #####################################
    imp = Importer("datahandler/Datasets/Preprocessed/DB-a")
    X_odd, y_odd, X_even, y_even = imp.get_odd_even()
    OddLoader = BatchLoader(X_odd, y_odd, batch_size=1000, shuffle=True) #Order must not matter
    EvenLoader = BatchLoader(X_even, y_even, batch_size=1000, shuffle=True)

    ##############################
    # CREATE GRAPH
    ##############################
    tf.reset_default_graph()

    #build_graph whatever this returns
    learning_rate, X_input, y_input, logits, loss, updateModel, train_acc, global_step, keep_prob = build_graph()
    model = {
        'lr': learning_rate,
        'X': X_input,
        'y': y_input,
        'logits': logits,
        'loss': loss,
        'updateModel': updateModel,
        'accuracy': train_acc,
        'global_step': global_step,
        'keep_prob': keep_prob
    }


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ##############################
        # TRAIN GRAPH
        ##############################
        for epoch in range(28):
            train_acc_list = []
            loss_list = []
            epoch_done = False

            print("Epoch {}".format(epoch))
            while not epoch_done:
                X_batch, y_batch, epoch_done = OddLoader.load_batch()
                lr = adapt_lr(epoch, 0.1)

                #run graph
                train_acc, _, loss, gs = sess.run([
                    model['accuracy'],
                    model['updateModel'],
                    model['loss'],
                    model['global_step']
                ],
                feed_dict={
                    model['X']: X_batch,
                    model['y']: y_batch,
                    model['lr']: lr,
                    model['keep_prob']: 0.5
                })
                if OddLoader.batch_counter % 100 == 0:
                    print("Global step: {}".format(gs))
                #end of run graph
                train_acc_list.append(train_acc)
                loss_list.append(loss)


                if (OddLoader.batch_counter % 50 == 0): #(epoch < 2 and OddLoader.batch_counter < 10) or

                    print("Loss list is: {}".format(loss_list))
                    print("Train accuracy: {:.3f}".format(float(sum(train_acc_list))/len(train_acc_list)))
                    logging.debug("Train accuracy: {:.3f}".format(float(sum(train_acc_list))/len(train_acc_list)))
                    train_acc_list = []
                    loss_list = []

                ##############################
                # TEST CV ACCURACY
                ##############################
                    cv_acc_list = []
                    cv_epoch_done = False
                    while not cv_epoch_done:
                        X_cv, y_cv, cv_epoch_done = EvenLoader.load_batch()

                        #run graph
                        cv_acc = sess.run([
                            model['accuracy']
                        ],
                        feed_dict={
                            model['X']: X_cv,
                            model['y']: y_cv,
                            model['keep_prob']: 1.
                        })
                        #end of run graph
                        cv_acc_list.append(cv_acc)

                    print("CV accuracy: {:.3f}".format(np.sum(cv_acc_list)/len(cv_acc_list)))
                    logging.debug("CV accuracy: {:.3f}".format(np.sum(cv_acc_list)/len(cv_acc_list)))


    ###################
    # END OF OPERATIONS


if __name__ == '__main__':
    main()