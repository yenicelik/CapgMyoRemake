from __future__ import print_function
import math
import tensorflow as tf
import numpy as np

class LocallyConnected_1x1(object):

    def __init__(self, inputs, layerName, nInputPlane, nFilters, iW, iH, is_training, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0):


        self.oW = math.floor((padW * 2 + iW - kW)) + 1
        self.oH = math.floor((padH * 2 + iH - kH)) + 1
        assert self.oW > 1 and self.oH > 1, "configuration is wrong. output height and width is <= 1!" % id

        self.nInputPlane = nInputPlane
        self.nFilters = nFilters

        self.weight = tf.get_variable("W_" + layerName, shape=[self.oH, self.oW, nFilters, nInputPlane],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01))
        self.bias = tf.get_variable("b_" + layerName, shape=[self.oH, self.oW, nFilters, nInputPlane],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01))


    def build_operation(self, inputs, is_training):
        #inputs must be same size as nInputPlane * iW * iH
        #this shit is incorrect right now, because we need this many pixels for that many shit.

        out = []
        for fid in range(self.nFilters): #fid is filter_id
            tmp_spatial = tf.multiply(inputs, self.weight[:,:, fid,:])
            tmp_spatial = tf.reduce_sum(tmp_spatial, axis=3) + self.bias[:,:, fid,:]
            tmp_spatial = tf.reshape(tmp_spatial, shape=[-1, 16, 8, 1]) #need to for concat later
            out.append(tmp_spatial)
        inputs = tf.concat(out, axis=3) #should be last dimension in which we 'grow'

        #Rest operations
        inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=0.9)
        inputs = tf.nn.relu(inputs)

        return inputs


def test_operation(inputs, weights, bias, nFilters):
    out = []
    for fid in range(nFilters): #fid is filter_id
        tmp_spatial = np.multiply(inputs, weights[:,:, fid,:]) + bias[:,:, fid,:]
        tmp_spatial = np.sum(tmp_spatial, axis=3)
        tmp_spatial = np.reshape(tmp_spatial, shape=[-1, 16, 8, 1]) #need to for concat later
        out.append(tmp_spatial)
    out = np.concat(out, axis=3) #should be last dimension in which we 'grow'
    return out


if __name__ == '__main__':
    inp = [[[1]]]
    w = [[[1]], [[1]]]
    b = []
    inp = [[1, 1],
           [1, 1]]
    w = [[[1, 1],
          [1, 1]],
         [[2, 2],
          [2, 2]],
         [[3, 3],
          [3, 3]]]
    b = []
    test_operation()


