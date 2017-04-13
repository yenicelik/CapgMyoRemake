import numpy as np
import tensorflow as tf

def setup():
    """ Sets up static structures and parameters """

    parameter = {
        "GAMMA": .99,
        "EPS": 0.1, #1. for production
        "NUM_EPISODES": 10, #max sohuld be 10 000 or 100 000 for complex tasks #maybe have multiple NUM_EPISODES IF TESTING OR STH ELSE
        "NUM_STEPS": 100, #should be open ended if wrapper is used
        "SAVE_EVERY": 2,
        "OLD_TF": False,
        "SAVE_FIGS": False,
        "X11": True
    }