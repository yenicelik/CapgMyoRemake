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
    importerObj = Importer("Datasets/Preprocessed/DB-a")
    super_matrix = importerObj.get_super_matrix()

    dataLoader = DataLoader(super_matrix)

    cross_subject_dataset = dataLoader.get_cross_subject_dataset()
    cross_session_dataset = dataLoader.get_random_cross_session_dataset()
    intra_session_dataset = dataLoader.get_random_intra_session_dataset()

    print("Data loader: ", dataLoader.sm.shape)
    print("Cross subject elements: ", len(cross_subject_dataset))
    print("Cross session elements: ", len(cross_session_dataset))
    print("Intra session elements: ", len(intra_session_dataset))


    #################################
    # Initializing TensorFlow Graph
    #################################
    #We don't have an intention to save stuff (at the moment at least, I think so). So no saver object is included. We just want the logs and the capacity of the model trained from scratch
    tf.global_variables_initializer()
    W, b, model_dict = init_graph()


    ################################
    # Cross Validate
    ################################
    with tf.Session() as sess:

        accuracy_list = []


        for i in range(len(cross_subject_dataset)):
            #tf.reset_default_graph() #TODO: Need to reinitialize weights after each run!!
            init = tf.initialize_all_variables()
            sess.run(init)

            X_cv, y_cv = cross_subject_dataset[i]
            X_cv = np.reshape(X_cv, (-1, 16, 8))
            y_cv = np.concatenate((y_cv, np.zeros((y_cv.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!

            train_set = [cross_subject_dataset[j] for j in range(len(cross_subject_dataset)) if i != j] #One could also just flatten the list, and every odd one is X, every even one is y. I just didn't want to introduce futher libraries.

            X_train = np.concatenate([x for x, y in train_set], axis=0)
            X_train = np.reshape(X_train, (-1, 16, 8)) #maybe do this within the model, might be faster if this is done uniformly between all functions anyways
            y_train = np.concatenate([y for x, y in train_set], axis=0)
            y_train = np.concatenate((y_train, np.zeros((y_train.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!

            train(
                        sess=sess,
                        parameter=parameter,
                        model_dict=model_dict,
                        X=X_train,
                        y=y_train,
                        saverObj=None
            )

            accuracy = test_accuracy(
                        sess=sess,
                        model_dict=model_dict,
                        parameter=parameter,
                        X=X_cv,
                        y=y_cv
            )

            accuracy_list.append(accuracy)

        total_accuracy = float(sum(accuracy_list))/len(cross_subject_dataset) * 100
        print("Accuracy of the current model is: {:.3f}%".format(total_accuracy))



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
            'NUM_EPOCHS': 1, #determine what these values should be. Cross validation can take the same time of training we had for 3h, but applied on every subject / potentially on every session etc.
            'BATCH_SIZE': 100,
            'SAVE_EVERY': 500 #number of batches after which to save
    }

    main(parameter)

#TODO: If we want pre-training, we must select all subjects for cross-session; pick all but one for pre-training, and apply the not-selected set as done above with the subjects