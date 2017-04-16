from __future__ import print_function
import numpy as np
import tensorflow as tf
from Model import *
from Importer import *
from train import train
from AccuracyTester import *
from Saver import *


def main():

    #################################
    # Initializing TensorFlow Graph
    #################################

    restore = True

    parameter = {
            'NUM_EPOCHS': 2,
            'BATCH_SIZE': 100,
            'BATCHES_PASSED': 0,
            'SAVE_DIR': 'saves/',
            'SAVE_EVERY': 25 #number of batches after which to save
    }

    tf.global_variables_initializer()
    W, b, model_dict = init_graph()


    saverObj = None
    if restore:
        saverObj = Saver(parameter['SAVE_DIR'])
    else:
        init = tf.initialize_all_variables()
        saverObj = Saver(parameter['SAVE_DIR'])


    with tf.Session() as sess:

        if restore:
            saverObj.load_session(sess, parameter['SAVE_DIR']) #Do I need anything else? Like to add the globa stuff etc.?
            print("Model should be restored now")
        else:
            sess.run(init)

        # model_dict = {
        #                 "X_input": X_input,
        #                 "y_input": y_input,
        #                 "loss": loss,
        #                 "predict": predict,
        #                 "trainer": trainer,
        #                 "updateModel": updateModel
        # }


        ################################
        # Importing all data
        ################################

        #I think we ignore which subjects these are from in the first run..

        importerDBA = Importer("Datasets/Preprocessed/DB-a")
        X, y = importerDBA.get_trainingset()

        print(X.shape)
        print(y.shape)

        #for this specific task, we need to get X and y as frames.... as such, it might be wise to do batches of size 1000, let each have a label of y... otherwise, it is not good to input a homogeneous dataset i thinkg
        X = np.reshape(X, (-1, 16, 8))
        y = np.reshape(y, (-1, 12))

        verify_size = 100

        X_verify = X[:verify_size, :, :]
        y_verify = y[:verify_size, :]

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X_sample = X[verify_size+indices[:100], :, :]
        y_sample = y[verify_size+indices[:100], :]

        ################################
        # Training on data
        ################################
        # train(
        #             sess=sess,
        #             parameter=parameter,
        #             model_dict=model_dict,
        #             X=X_verify,
        #             y=y_verify,
        #             saverObj=saverObj
        # )

        test_accuracy(
                    sess=sess,
                    model_dict=model_dict,
                    parameter=parameter,
                    X=X_sample,
                    y=y_sample,

        )



if __name__ == '__main__':
    #TODO: Set up the dictionary, or potentially the terminal read-in from here
    main()