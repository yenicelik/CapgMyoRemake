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

    acc_list = []
    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(NUM_TEST_CASES):
            print("At step {}".format(i))
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
                    model_dict['X_input']: X_train[i*500:(i+1)*500, :],
                    model_dict['y_input']: y_train[i*500:(i+1)*500],
                    model_dict['keepProb']: 0.5,
                    model_dict['learningRate']: parameter['LEARNING_RATE'],
                    model_dict['isTraining']: True
                }
            )

            if i % 10 == 0:
                acc = test_model_accuracy(X_test[i*500:(i+1)*500], y_test[i*500:(i+1)*500], parameter, show_confusion_matrix=False)
                acc_list.append(acc)
                #print("Accuracy at step {} is: {}".format(i, acc * 100))
                print("Accuracy total is: {}".format(sum(acc_list)/len(acc_list)*100))

if __name__ == '__main__':
    main()