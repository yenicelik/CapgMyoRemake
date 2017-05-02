from __future__ import print_function
import numpy as np
from datahandler.BatchLoader import BatchLoader


def test_model_accuracy(X, y, parameter, show_confusion_matrix, sess=None, model_dict=None):

    batchLoader = BatchLoader(X, y, 1000, shuffle=False)
    acc_list = []
    epoch_done = False


    while not epoch_done:
        X_batch, y_batch, epoch_done = batchLoader.load_batch()

        loss, logits = sess.run(
                #Describe what we want out of the model
                [
                    model_dict['loss'],
                    model_dict['predict'],
                ],
                #Describe what we input in the model
                feed_dict = {
                    model_dict['X_input']: X_batch,
                    model_dict['y_input']: y_batch,
                    model_dict['keepProb']: 1.,
                    model_dict['learningRate']: parameter['LEARNING_RATE'],
                    model_dict['isTraining']: False
                }
        )


        #Both reduced from one-hot to arg-vector
        predict = np.argmax(logits, axis=1)
        actual = np.argmax(y_batch, axis=1)

        logging.debug("To debug the shit-bug")
        logging.debug("X_batch input is: {}".format(X_batch.shape))
        logging.debug("y_batch input is: {}".format(y_batch.shape))
        logging.debug("Predict list is: {}".format(predict))
        logging.debug("Actual list is: {}".format(actual))

        if len(predict) != len(actual):
            logging.error("Predict length {} doesn't equal actualy length {}".format(len(predict), len(actual)))
            sys.exit(69)
        difference_vector = [1 if pred == act else 0 for pred, act in zip(predict, actual)]
        accuracy = np.sum(difference_vector) / float(len(difference_vector))
        logging.debug("Accuracy so far is: {:.3f}%".format(accuracy*100))
        acc_list.append(accuracy)

    logging.info("Accuracy is: {:.3f}%".format(sum(acc_list)/float(len(acc_list))*100))
    logging.info("Random baseline: {:.3f}%".format(1./10*100))

    #Confusion Matrix
    if show_confusion_matrix:
        cm = confusion_matrix(actual, predict)
        logging.debug("Showing confusion matrix: {}".format(cm))
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
        plt.show(block=True)

    logging.debug("<- {} function".format(test_model_accuracy.__name__))
    return accuracy