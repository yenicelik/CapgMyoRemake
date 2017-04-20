from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from datahandler.BatchLoader import BatchLoader


#TODO: implement majority-voting over time-frame
def test_accuracy(sess, model_dict, parameter, X, y, verbose=False, show_confusion_matrix=False):
    """
    :param sess: The tensorflow session that we are going to use.
    :param model_dict: The model-dictionary that we need to refer to as the tensorflow-graph
    :param parameter: The parameter dictionary, holding different values needed.
    :param X: The test-set.
    :param y: The respective labels of the test-set.
    :param verbose: Whether the sample prediction and actual y-values should be printed.
    :return: Nothing
    """
    plt.interactive(False)
    plt.ion()
    #
    # plt.plot([1, 2, 3])
    # plt.show()

    #TODO: why should the batch_size be specified? I mean, in the end we want the entire X to be tested, right? at least the entire X that was input into this function..
    batchLoader = BatchLoader(X, y, parameter['BATCH_SIZE'], shuffle=False)
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
                            model_dict['keepProb']: 1.
                        }
                    )

    predict = np.argmax(logits, axis=1)
    actual = np.argmax(y_batch, axis=1)

    if verbose:
        print("predict is: ", predict)
        print("actual is: ", actual)

    difference = [1 if pred == act else 0 for pred, act in zip(predict, actual)]
    accuracy = (np.sum(difference) / float(X_batch.shape[0]))

    ##############################
    # Confusion matrix
    ##############################
    print("Accuracy is: {:.3f}%".format(accuracy*100))
    print("Random baseline: {:.3f}%".format(1./32*100))

    if show_confusion_matrix:
        cm = confusion_matrix(actual, predict)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
        plt.show(block=True)

    return accuracy








