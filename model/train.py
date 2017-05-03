from __future__ import print_function

import tensorflow as tf

from accuracy import *
import logging
logging = logging.getLogger(__name__)

def train(X_train,
          y_train,
          X_cv,
          y_cv,
          cv_accuracy_function,
          parameter,
          sess,
          model):

    trainLoader = BatchLoader(X_train, y_train, parameter['BATCH_SIZE'], shuffle=True)

    for epoch in range(parameter['NUM_EPOCHS']):
        print("Epoch: {}".format(epoch))
        lr = adapt_lr(epoch, parameter['LEARNING_RATE'])

        train_acc_list = []
        epoch_done = False
        while not epoch_done:
            X_batch, y_batch, epoch_done = trainLoader.load_batch()

            accuracy, _ = sess.run(
                [
                    model['accuracy'],
                    model['updateModel'],
                ],
                feed_dict={
                    model['X']: X_batch,
                    model['y']: y_batch,
                    model['keep_prob']: 0.5,
                    model['lr']: lr,
                    model['is_training']: True
                }
            )
            train_acc_list.append(accuracy)

            if (trainLoader.batch_counter % 100 == 0):
                    cv_accuracy = cv_accuracy_function(X_cv, y_cv, parameter, sess=sess, model=model)
                    print("Train accuracy: {:.3f}".format(float(sum(train_acc_list))/len(train_acc_list)))
                    logging.debug("Train accuracy: {:.3f}".format(float(sum(train_acc_list))/len(train_acc_list)))
                    print("CV accuracy: {:.3f}".format(cv_accuracy))
                    logging.debug("CV accuracy: {:.3f}".format(cv_accuracy))
                    train_acc_list = []


def adapt_lr(epoch, para_lr):
    logging.debug("-> {} function".format(adapt_lr.__name__))
    if epoch >= 24:
        learning_rate = para_lr / 100
        logging.debug("Diving learning rate by 100. Before: {}; After: {}".format(para_lr, learning_rate))
    elif epoch >= 16:
        learning_rate = para_lr / 10
        logging.debug("Diving learning rate by 10. Before: {}; After: {}".format(para_lr, learning_rate))
    else:
        learning_rate = para_lr

    logging.debug("<- {} function".format(adapt_lr.__name__))
    return learning_rate
