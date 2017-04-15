from __future__ import print_function
import numpy as np
import tensorflow as tf
from BatchLoader import BatchLoader


def test_accuracy(model_dict, parameter, X, y):


    #TODO: we must somehow find a way to test the accuracy of this model, without necessarily saving the model... Potentially, we could implement this 'test accuracy' as part of the training process (with a CV option)
    #TODO: create a 'predict' function
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

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
                                model_dict['y_input']: y_batch
                            }
                        )

        predict = np.argmax(logits, axis=1)
        actual = np.argmax(y_batch, axis=1)

        print(predict)
        print(actual)

        difference = [1 if pred == act else 0 for pred, act in zip(predict, actual)]
        accuracy = (np.sum(difference) / float(X_batch.shape[0]))

        print("Accuracy is: {:.3f} ".format(accuracy))
        print("Random baseline: {:.3f}".format(1./32))





