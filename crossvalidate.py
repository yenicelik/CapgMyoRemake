from __future__ import print_function

from datahandler.Importer import *
from datahandler.Saver import *
from datahandler.DataLoader import *
from metrics.AccuracyTester import *
from model.BuildGraph import *
from train import *


def main(parameter):
    #Basically, we always perform 'leave-one-out' cross validation

    ################################
    # Importing all data
    ################################
    importerObj = Importer("/Datasets/Preprocessed/DB-a")
    super_matrix = importerObj.get_super_matrix()

    dataLoader = DataLoader(super_matrix)

    cross_subject_dataset = dataLoader.get_cross_subject_dataset()
    cross_session_dataset = dataLoader.get_random_cross_session_dataset()
    intra_session_dataset = dataLoader.get_random_intra_session_dataset()

    print("Data loader: ", dataLoader.sm.shape)
    print("Cross subject elements: ", len(cross_subject_dataset))
    print("Cross session elements: ", len(cross_session_dataset))
    print("Intra session elements: ", len(intra_session_dataset))
    print(intra_session_dataset)


    #################################
    # Initializing TensorFlow Graph
    #################################
    #We don't have an intention to save stuff (at the moment at least, I think so). So no saver object is included. We just want the logs and the capacity of the model trained from scratch
    tf.global_variables_initializer()
    W, b, model_dict = init_graph()
    init = tf.initialize_all_variables()

    ################################
    # Training on data
    ################################



    with tf.Session() as sess:
        sess.run(init)

        for dataset in cross_subject_dataset:
            #TODO: need to work on the 'cross-validatoin' aspect (leaving one out, testing on all the other ones.. I think for that we just need to concatenate all X[not i] (so maybe use an enumeration variable)
            tf.reset_default_graph()
            X, y = dataset

            train(
                        sess=sess,
                        parameter=parameter,
                        model_dict=model_dict,
                        X=X,
                        y=y,
                        saverObj=saverObj #TODO: create an option such that not saving is also possible
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


    #TODO: implement a log-file
    parameter = {
            'NUM_EPOCHS': 1,
            'SAVE_EVERY': 500 #number of batches after which to save
    }

    main(parameter)