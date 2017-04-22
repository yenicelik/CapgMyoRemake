from __future__ import print_function

from datahandler.Importer import *
from datahandler.Saver import *
from metrics.AccuracyTester import *
from model.BuildGraph import *
from train import *


#TODO: Build tensorflow from source (Assembly instruction will make it faster by 3-8 times
def main(restore, parameter, full_train=False, go_train=True):

    ################################
    # Importing all data
    ################################
    importerDBA = Importer("Datasets/Preprocessed/DB-a")
    X, y = importerDBA.get_trainingset()

    #for this specific task, we need to get X and y as frames. Make sure these are not all from the same class
    X = np.reshape(X, (-1, 16, 8))
    y = np.reshape(y, (-1, 12))

    if full_train:
        verify_size = X.shape[0]
    else:
        verify_size = 100000

    X_verify = X[:verify_size, :, :]
    y_verify = y[:verify_size, :]

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_sample = X[indices[:1000], :, :]
    y_sample = y[indices[:1000], :]


    #################################
    # Initializing TensorFlow Graph
    #################################
    tf.global_variables_initializer()
    W, b, model_dict = init_graph()
    saverObj = Saver(parameter['SAVE_DIR'])

    if not restore:
        init = tf.initialize_all_variables()

    ################################
    # Training on data
    ################################
    with tf.Session() as sess:

        if restore:
            saverObj.load_session(sess, parameter['SAVE_DIR']) #Do I need anything else? Like to add the globa stuff etc.?
            print("Model should be restored now")
        else:
            sess.run(init)

        if go_train:
            train(
                        sess=sess,
                        parameter=parameter,
                        model_dict=model_dict,
                        X=X_verify,
                        y=y_verify,
                        saverObj=saverObj
            )

        test_accuracy(
                    sess=sess,
                    model_dict=model_dict,
                    parameter=parameter,
                    X=X_sample,
                    y=y_sample,

        )



if __name__ == '__main__':

    # The model_dict we will use
        # model_dict = {
        #                 "X_input": X_input,
        #                 "y_input": y_input,
        #                 "loss": loss,
        #                 "predict": predict,
        #                 "trainer": trainer,
        #                 "updateModel": updateModel
        # }

    restore = True #wether we want to use the model saved in 'saves/', or start training a model from scratch.

    parameter = {
            'NUM_EPOCHS': 28,
            'BATCH_SIZE': 1000,
            'SAVE_DIR': 'saves/',
            'SAVE_EVERY': 300 #number of batches after which to save
    }

    main(restore, parameter, full_train=False, go_train=True)