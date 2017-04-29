from datahandler.DataLoader import DataLoader
from model.BuildGraph import *
import tensorflow as tf
from metrics.AccuracyTester import *

def main():
    NUM_TEST_CASES = 500

    dataLoader = DataLoader("datahandler/Datasets/Preprocessed/DB-a")
    X_train, y_train, X_test, y_test = dataLoader.get_odd_even_trials()

    W, b, model_dict, tmpval= init_graph()

    parameter = {
        'LEARNING_RATE': 0.1,
        'NUM_EPOCHS': 28,
        'BATCH_SIZE': 1000
    }

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(NUM_TEST_CASES):
            logging.debug("Entering session")
            sess.run(init_op)
            logging.debug("Initial_op success. Now running session on data")
            loss, predict, _, _ = sess.run(
                # Describe what we want out of the model
                [
                    model_dict['loss'],
                    model_dict['predict'],
                    model_dict['updateModel'],
                    model_dict['globalStepTensor']

                ],
                # Describe what we input in the model
                feed_dict={
                    model_dict['X_input']: X_train[i*1000:(i+1)*1000, :],
                    model_dict['y_input']: y_train[i*1000:(i+1)*1000],
                    model_dict['keepProb']: 0.5,
                    model_dict['learningRate']: parameter['LEARNING_RATE'],
                    model_dict['isTraining']: True
                }
            )

            print("Loss at step {} is: {}".format(i, sum(loss)))

if __name__ == '__main__':
    main()