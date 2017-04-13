import numpy as np
import tensorflow as tf


def setup_saver(W, b):
    features = W.copy()
    features.update(b)
    saver = tf.train.Saver(features)

    return saver
