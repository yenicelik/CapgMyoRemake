import numpy as np
import tensorflow as tf


def setup_saver(W, b):
    features = W.copy()
    features.update(b)
    saver = tf.train.Saver(features)

    return saver

def to_one_hot(x, dims):
    out = np.zeros(dims)
    out[x] = 1
    return out
