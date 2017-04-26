from __future__ import print_function

from datahandler.DataLoader import *
from metrics.AccuracyTester import *
from model.BuildGraph import *
from train import *
import logging
logging = logging.getLogger(__name__)

import os
import json
import logging.config

def setup_logging(
    default_path='logging.json',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

#TODO: implement a logging system
#TODO: go through code for: 1. any sytnax mistakes / improvements 2. any sematic mistakes / improvements 3. and structural mistakes / improvements (encapsulation etc.)
def crossvalidate_intrasession(parameter, dataLoader):
    # Due to the following arguments, this is equivalent to training on the entire training set and validating on a few datasamples that were not seen before.
    # The usual logic implies the following data training:
    #   1. Pre-training on everything except a few frames of.
    #   2. Then use cross-validation on a few left frames by leaving one value out. that are in the same trial.
    # If we do so, all data samples except these few cross-validated sets have been seen.
    # The network may have a higher bias towards the finally trained session (we can easily do so by adaption).
    # This action is equivalent to pretraining on all data but a test-data. Then cross-validating on some randomly encountered data.
    # Because this task is will likely yield similar results to if everything was trained befreo anyways, we just train on everything simultaneously

    pass
    # X_raw, y_raw = dataLoader.get_unfiltered_data()
    #
    # #shuffle that shit
    # indices = np.arange(X_raw.shape[0])
    # np.random.shuffle(indices)
    #
    # test_ratio = 0.2
    # test_size = X_raw.shape[0] * (1-test_ratio) #I very much hope X is divisible by 0.1 with a result that is divisible by 1000 #TODO:somehow make this parameter more 'safe'
    # train_size = X_raw.shape[0] * test_ratio
    # X_train = X_raw[indices[test_size:]]
    # y_train = y_raw[indices[test_size:]]
    # X_cv = X_raw[indices[:test_size]]
    # y_cv = y_cv[indices[:test_size]]
    #
    #
    # ## Initialize TensorFlow Graph
    # tf.global_variables_initializer()
    # W, b, model_dict = init_graph()
    #
    # ## Cross Validate
    # accuracy_list = []
    #
    # # decide if it is useful to filter by subjects
    #
    # with tf.Session() as sess:
    #     init = tf.initialize_all_variables()
    #     sess.run(init)
    #
    #     ## Now train
    #     X_pretrain, y_pretrain = dataLoader.get_cross_session_pretrain_dataset(sid)
    #     X_train = np.reshape(X_train, (-1, 16, 8)) #maybe do this within the model, might be faster if this is done uniformly between all functions anyways
    #     y_train = np.concatenate((y_train, np.zeros((y_pretrain.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!
    #
    #     train(
    #                 sess=sess,
    #                 parameter=parameter,
    #                 model_dict=model_dict,
    #                 X=X_train,
    #                 y=y_train,
    #                 saverObj=None
    #     )
    #
    #     ## Now get accuracy and append to the final list
    #     accuracy = test_accuracy(
    #                 sess=sess,
    #                 model_dict=model_dict,
    #                 parameter=parameter,
    #                 X=X_cv,
    #                 y=y_cv
    #     )
    #
    #     accuracy_list.append(accuracy)
    #
    # total_accuracy = float(sum(accuracy_list))/1 * 100 #Currently, we just do cross-validation over one element. We need to change this later
    # print("Accuracy of the current model on cross-subject is: {:.3f}%".format(total_accuracy))
    #
    #
    # return accuracy_list



def crossvalidate_crosssession(parameter, dataLoader):

    no_of_subjects = dataLoader.no_of_subjects
    #We construct a training set based on all the subjects that are not involved in testing for the individual session.
    #Then we cross-validate over all sessions from within this subject

    #we need to separate by trials!

    ## Initialize TensorFlow Graph
    tf.global_variables_initializer()
    W, b, model_dict = init_graph()

    ## Cross Validate
    accuracy_list = []

    for sid in dataLoader.subjects: #TODO: change any 'range(no_of_subjects)' into this format!

        #Is this effective enough? Couldn't we sample a smaller subset of this or so?

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            ## Pretrain on existing data
            X_pretrain, y_pretrain = dataLoader.get_cross_session_pretrain_dataset(sid)
            X_pretrain = np.reshape(X_pretrain, (-1, 16, 8)) #maybe do this within the model, might be faster if this is done uniformly between all functions anyways
            y_pretrain = np.concatenate((y_pretrain, np.zeros((y_pretrain.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!


            X_pretrain = X_pretrain[:1000, :, :]
            y_pretrain = y_pretrain[:1000, :]

            print("X_pretrain size: ", X_pretrain.shape)
            print("y_pretrain size: ", y_pretrain.shape)

            train(
                        sess=sess,
                        parameter=parameter,
                        model_dict=model_dict,
                        X=X_pretrain,
                        y=y_pretrain,
                        saverObj=None
            )

            ## Save pre-trained weights
            save_path='tmp/cross-session-pretrained.ckpt' #must be ckpt!

            saver = tf.train.Saver()
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            saver.save(sess, save_path, global_step=model_dict['globalStepTensor'])

            ## Now train
            cross_session_dataset = dataLoader.get_cross_session_dataset_given_subject_id(sid)

            print("Cross session dataset: ", len(cross_session_dataset))

            for i in range(len(cross_session_dataset)):

                ## Reinitilize variables to pre-trained state
                saver.restore(sess, save_path)

                X_cv, y_cv = cross_session_dataset[i]
                X_cv = np.reshape(X_cv, (-1, 16, 8))
                y_cv = np.concatenate((y_cv, np.zeros((y_cv.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!

                train_set = [cross_session_dataset[j] for j in range(len(cross_session_dataset)) if i != j] #One could also just flatten the list, and every odd one is X, every even one is y. I just didn't want to introduce futher libraries.

                X_train = np.concatenate([x for x, y in train_set], axis=0)[:2000, :]
                X_train = np.reshape(X_train, (-1, 16, 8)) #maybe do this within the model, might be faster if this is done uniformly between all functions anyways
                y_train = np.concatenate([y for x, y in train_set], axis=0)[:2000]
                y_train = np.concatenate((y_train, np.zeros((y_train.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!

                train(
                            sess=sess,
                            parameter=parameter,
                            model_dict=model_dict,
                            X=X_train,
                            y=y_train,
                            saverObj=None
                )

                ## Now get accuracy and append to the final list
                accuracy = test_accuracy(
                            sess=sess,
                            model_dict=model_dict,
                            parameter=parameter,
                            X=X_cv,
                            y=y_cv
                )

                accuracy_list.append(accuracy)

        total_accuracy = float(sum(accuracy_list))/max(len(cross_session_dataset), 1) * 100
        print("Accuracy of the current model on cross-subject is: {:.3f}%".format(total_accuracy))

        return accuracy_list



def crossvalidate_subjects(parameter, dataLoader, is_voting):

    logging.debug("Entering cross_validate_subjects function")

    cross_subject_dataset = dataLoader.get_cross_subject_dataset()

    logging.debug("Data loader's super_matrix has shape: {}".format(dataLoader.sm.shape))
    logging.debug("The number of different subjects: {}".format(len(cross_subject_dataset)))

    logging.debug("Initializing tensorflow graph")
    ## Initializing TensorFlow Graph
    tf.global_variables_initializer() #TODO: change this to initialize_all_variables! or vice verca
    W, b, model_dict = init_graph()

    logging.debug("tensorflow graph initialized successfully")

    ## Cross Validate
    accuracy_list = []

    for i in range(len(cross_subject_dataset)):
        #tf.reset_default_graph() #TODO: Need to reinitialize weights after each run!! Opening and closing a session each time is a hacky solution!
        logging.info("Cross-validating by leaving subject number {} out".format(i))
        with tf.Session() as sess:

            logging.debug("Re-initializing graph")
            init = tf.initialize_all_variables()
            sess.run(init)
            logging.debug("Graph re-initialized successfully")

            X_cv, y_cv = cross_subject_dataset[i]
            X_cv = np.reshape(X_cv, (-1, 16, 8))
            y_cv = np.concatenate((y_cv, np.zeros((y_cv.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!

            logging.debug("X_cv has shape: {}".format(X_cv.shape))
            logging.debug("y_cv has shape: {}".format(X_cv.shape))

            train_set = [cross_subject_dataset[j] for j in range(len(cross_subject_dataset)) if i != j] #One could also just flatten the list, and every odd one is X, every even one is y. I just didn't want to introduce futher libraries.

            X_train = np.concatenate([x for x, y in train_set], axis=0)
            X_train = np.reshape(X_train, (-1, 16, 8)) #maybe do this within the model, might be faster if this is done uniformly between all functions anyways
            y_train = np.concatenate([y for x, y in train_set], axis=0)
            y_train = np.concatenate((y_train, np.zeros((y_train.shape[0], 2))), axis=1) #TODO: very hacky solution. Must change this value in the model later!!!! Create a variable within the model for this!

            logging.debug("X_train has shape: {}".format(X_train.shape))
            logging.debug("y_train has shape: {}".format(X_train.shape))

            # logging.warning("Using only the first 1000 elements!")
            # X_train = X_train[:1000, :, :]
            # y_train = y_train[:1000, :]

            #TODO: save model, and allow option to retrieve saved model (simple if-statement); we can also just say we want a saver object with a certain path. Might need to change the parameter['SAVE_DIR'] option

            logging.debug("Entering train function")
            train(
                        sess=sess,
                        parameter=parameter,
                        model_dict=model_dict,
                        X=X_train,
                        y=y_train,
                        saverObj=None
            )

            accuracy = 0

            if is_voting:
                logging.debug("Entering voting function")
                accuracy = voting(
                            sess=sess,
                            model_dict=model_dict,
                            parameter=parameter,
                            X=X_cv,
                            y=y_cv,
                            framesize=1000,
                            verbose=False
                )
            else:
                logging.debug("Entering vanilla accuracy tester function")
                accuracy = test_accuracy(
                            sess=sess,
                            model_dict=model_dict,
                            parameter=parameter,
                            X=X_cv,
                            y=y_cv
                )

            logging.warning('Accuracy of leave-out subject is {:.3f}'.format(accuracy*100))  # will not print anything

            accuracy_list.append(accuracy)

    total_accuracy = float(sum(accuracy_list))/len(cross_subject_dataset) * 100
    logging.warning("Accuracy of the current model on cross-subject is: {:.3f} percent ".format(total_accuracy))
    print("Accuracy of the current model on cross-subject is: {:.3f} percent ".format(total_accuracy))

    return accuracy_list


def main(parameter, mode, is_voting, path):
    #Basically, we always perform 'leave-one-out' cross validation

    ################################
    # Importing all data
    ################################
    logging.debug("Using the Importer object on the path: {}".format(path))
    logging.debug("Using the parameter: {}".format(parameter))
    logging.info("Testing on: {}".format(mode))
    logging.debug("Using voting: {}".format(is_voting))

    importerObj = Importer(path)
    super_matrix = importerObj.get_super_matrix()

    dataLoader = DataLoader(super_matrix)

    logging.debug("The imported super_matrix has shape: {}".format(super_matrix.shape))
    logging.debug("dataLoader has type: {}".format(type(dataLoader)))

    #TODO: create a logging style, such that multiple modes at once can be selected aswell
    if mode=="cross-session":
        crossvalidate_crosssession(parameter, dataLoader, is_voting)
    elif mode=="intra-session":
        crossvalidate_intrasession(parameter, dataLoader, is_voting)
    elif mode=="cross-subject":
        crossvalidate_subjects(parameter, dataLoader, is_voting)
    else:
        logging.error("No mode (cross-session, cross-subject, intra-session) was specified!")
        sys.exit(11)


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

    test_parameter = {
        'NUM_EPOCHS': 1, #determine what these values should be. Cross validation can take the same time of training we had for 3h, but applied on every subject / potentially on every session etc.
        'BATCH_SIZE': 100,
        'SAVE_EVERY': 10, #number of batches after which to save,
        'LEARNING_RATE': 0.1
    }

    #TODO: implement a log-file
    paper_parameter = {
            'NUM_EPOCHS': 28, #determine what these values should be. Cross validation can take the same time of training we had for 3h, but applied on every subject / potentially on every session etc.
            'BATCH_SIZE': 1000,
            'SAVE_EVERY': 300, #number of batches after which to save,
            'LEARNING_RATE': 0.1
    }

    setup_logging()


    path = "Datasets/Preprocessed/DB-a"

    main(paper_parameter, mode="cross-subject", is_voting=True, path=path)

#TODO: If we want pre-training, we must select all subjects for cross-session; pick all but one for pre-training, and apply the not-selected set as done above with the subjects