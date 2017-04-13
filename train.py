#External dependencies
import numpy as np
import tensorflow as tf
import random
import sys
import datetime
import time

#Project dependencies
from Model import init_graph
from Importer import *



"""
parameter is a dictionary consisting of
'NUM_EPISODES':

"""

#TODO: Epoch == Full pass through the data
#TODO: the model should be saved on the harddrive frequently!

def train(parameter, forward_dict, X, y):
    """ Trains the network to the given environment. Saves weights to saver
        In: parameter (Dictionary with settings)
        In: saver (tf.saver where weights and bias will be saved to)
        In: forward_dict (Dictionary referencing to the tensorflow model)
        In: loss_dict (Dictionary referencing to the tensorflow model)
        Out: rewards_list (reward for instantenous run)
        Out: steps_list (number of steps 'survived' in given episode)
    """
    loss_list = []

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in xrange(parameter['NUM_EPOCHS']):


            start_time = datetime.datetime.now()

            loss = run_epoch(
                                cur_epoch=epoch,
                                sess=sess,
                                parameter=parameter,
                                forward_dict=forward_dict,
                                loss_dict=loss_dict
            )

            end_time = datetime.datetime.now()
            total_time = end_time - start_time

            loss_list.append(loss) #training loss

            percentage = float(epoch) / parameter['NUM_EPISODES']

            print("Progress: {0:.3f}%%".format(percentage * 100))
            print("EST. time per episode: " + str(total_time))
            print("Episodes left: {0:d}".format(parameter['NUM_EPISODES'] - epoch))
            print("Average loss of current episodes: " + str(sum(loss_list)/parameter['NUM_EPISODES']) + "%")
            print("")

    return loss_list


def run_epoch(sess, cur_episode, parameter, forward_dict, loss_dict):
    """ Run one episode of  environment
        In: cur_episode
        In: parameter
        Out: total_reward (accumulated over episode)
        Out: steps (needed until termination)
    """

    done = False
    steps = 1

    ####### Initializers come here

    #TODO: split up the data and shuffle it, then give one after the other into a step function until the epoch has been worked one.



    size = parameter['BATCH_SIZE'] / data_size



    while steps < parameter['NUM_STEPS']:
        steps += 1
        new_observation, reward, done, p_observation, action, p_new_observation = step_environment(
                            env=env,
                            observation=observation,
                            sess=sess,
                            eps=parameter['EPS']/(cur_episode+1) + 0.06, #consider turning this into a lambda dictionary function
                            gamma=parameter['GAMMA'],
                            forward_dict=forward_dict,
                            loss_dict=loss_dict
                        )
        total_reward += reward

        aptuple = (p_observation, action, reward, p_new_observation)
        replay_memory.append(aptuple)



    while steps < parameter['NUM_STEPS']:
        steps += 1

        #depending on whether we have an algorithm to interwind, we must create a new function step-environment. else, we can just use 'sess.run' a few times




    return total_reward, steps


def step_environment(sess, eps, gamma, forward_dict, loss_dict):
    """ Take one step within the environment """
    """ In: env (OpenAI gym wrapper)
        In: observation (current state of game)
        In: sess (tf graph instance)
        In: eps
        In: gamma
        Out: new_observation
        Out: reward
        Out: done
    """
    sess.run()



    action, all_Qs = sess.run([forward_dict['predict'], forward_dict['Qout']], feed_dict={forward_dict['input']: p_observation}) #takes about 70% of the running time.. which is fine bcs that's the heart of the calculation
    if np.random.rand(1) < eps:
        action[0] = env.action_space.sample()

    new_observation, reward, done, _ = env.step(action[0])

    ##Max Value forward
    p_new_observation = preprocess_image(new_observation)
    Q_next = sess.run([forward_dict['Qout']], feed_dict={forward_dict['input']: p_new_observation})
    maxQ_next = np.max(Q_next)
    targetQ = all_Qs
    targetQ[0, action[0]] = reward + gamma * maxQ_next

    ##Update to more optimal features
    sess.run([loss_dict['updateModel']], feed_dict={forward_dict['input']:p_new_observation, loss_dict['nextQ']: targetQ})

    return new_observation, reward, done, p_observation, action, p_new_observation


