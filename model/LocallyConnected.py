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
        self.bias = tf.get_variable("b_" + layerName, shape=[self.oH, self.oW, nFilters],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01))


    def build_operation(self, inputs, is_training):

        out = []
        for fid in range(self.nFilters): #fid is filter_id
            tmp_spatial = tf.multiply(inputs, tf.reshape(self.weight[:,:, fid,:], tf.shape(inputs)))
            tmp_spatial = tf.reduce_sum(tmp_spatial, axis=3)
            tmp_spatial = tf.reshape(self.bias[:,:, fid], tf.shape(tmp_spatial))
            tmp_spatial = tf.reshape(tmp_spatial, shape=[-1, 16, 8, 1])
            out.append(tmp_spatial)
        inputs = tf.concat(out, axis=3) #should be last dimension in which we 'grow'

        #Follow-up operations
        inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=0.9)
        inputs = tf.nn.relu(inputs)

        return inputs


def test_operation(inputs, weights, bias, nFilters):
    out = []
    for fid in range(nFilters): #fid is filter_id
        #print(weights[np.arange(weights.shape[0]),np.arange(weights.shape[1]),fid,np.arange(weights.shape[3])])
        tmp_spatial = np.multiply(inputs, np.reshape(weights[:,:, fid,:], (inputs.shape)))
        tmp_spatial = np.sum(tmp_spatial, axis=3)
        tmp_spatial += np.reshape(bias[:,:, fid], tmp_spatial.shape)
        tmp_spatial = np.reshape(tmp_spatial, (-1, inputs.shape[0], inputs.shape[1], 1)) #need to for concat later
        out.append(tmp_spatial)
    out = np.concatenate(out, axis=3) #should be last dimension in which we 'grow'
    return out


if __name__ == '__main__':
    #Case 1:
    inp = np.array([[[[1]]]])
    w = np.array([[[[1]]]])
    b = np.array([[[1]]])
    out = np.array([[[2]]])
    predict = test_operation(inp, w, b, 1)
    assert out == predict, "Failure in test case 1"
    print("\n\n")


    #Case 2:
    inp = np.array([[[[1]]]])
    w = np.array([[[[1],
                    [1]],
                   ]])
    b = np.array([[[1, 1]]])
    out = np.array([[[[2, 2]]]])
    predict = test_operation(inp, w, b, 2)
    print("Case 2: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")


    #Case 3:
    inp = np.array([[[[1, 1, 1]]]])
    w = np.array([[[[1, 1, 1],
                    [1, 1, 1]],
                   ]])
    b = np.array([[[2, 1]]])
    out = np.array([[[5, 4]]])
    predict = test_operation(inp, w, b, 2)
    print("Case 3: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")


    #Case 3.1:
    inp = np.array([[[[1, -1, 1]]]])
    w = np.array([[[[1, 2, 1],
                    [1, 1, 4]],
                   ]])
    b = np.array([[[2, 1]]])
    out = np.array([[[2, 5]]])
    predict = test_operation(inp, w, b, 2)
    print("Case 3.1: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")

    #Case 4:
    inp = np.array([
       [[[1, 1, 1]],
        [[1, 1, 1]]],
       [[[1, 1, 1]],
        [[1, 1, 1]]]
    ])
    w = np.array([[
        [[ 1, 1, 1],
         [ 1, 1, 1]],
        [[ 1, 1, 1],
         [ 1, 1, 1]]],

       [[[ 1, 1, 1],
         [ 1, 1, 1]],
        [[ 1, 1, 1],
         [ 1, 1, 1]]]
    ])
    b = np.array(
        [[[ 1, 1],
         [ 1, 1]],

        [[ 1, 1],
         [ 1, 1]]]
    )
    out = np.array([
        [[4, 4],
        [4, 4]],

       [[4, 4],
        [4, 4]]
    ])
    predict = test_operation(inp, w, b, 2)
    print("Case 4: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")


    #Case 4.1:
    inp = np.array([
       [[[1]],
        [[1]]],
       [[[1]],
        [[1]]]])

    w = np.array([
        [[[1],
         [1]],
        [[1],
         [1]]],

       [[[1],
         [1]],
        [[1],
         [1]]]])
    b = np.array([[[1, 1],
                   [1, 1]],
                  [[1, 1],
                   [1, 1]]]
                  )
    out = np.array([[[2, 2],
                     [2, 2]],
                    [[2, 2],
                     [2, 2]]])
    predict = test_operation(inp, w, b, 2)
    print("Case 4.1: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")

    #Case 4.2:
    inp = np.array([
       [[[3]],
        [[1]]],
       [[[4]],
        [[-2]]]])

    w = np.array([
        [[[2], #6
         [0]], #0
        [[3], #3
         [1]]], #-2

       [[[3], #12
         [0]], #0
        [[0], #0
         [-5]]]]) #10
    b = np.array([[[3, 5], #9, 5
                   [-2, -1]], #1, 0,
                  [[0, -3], #12, -3
                   [-4, 2]]] #-4, 12
                  )
    out = np.array([[[9, 5],
                     [1, 0]],
                    [[12, -3],
                     [-4, 12]]])
    predict = test_operation(inp, w, b, 2)
    print("Case 4.2: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")


    #Case 5:
    inp = np.array([
       [[[1, 1, 2]],
        [[1, 4, 1]]],
       [[[1, -2, 1]],
        [[1, 0, 1]]]
    ])
    w = np.array([[

        [[ 1, -2, 1], # [1, 1, 2] # 1 -2 +2 = 1
         [ 0, -1, 4]], # [1, 1, 2] # 0 -1 +8 = 7
        [[ 1, 2, 1], # [1, 4, 1] # 1 + 8 +1 = 10
         [ 1, 0, 3]]], # [1, 4, 1] 1 + 3 = 4

       [[[ 1, 1, 2], # [1, -2, 1] # 1 -2 +2 = 1
         [ 1, -4, 1]], # [1, -2, 1] # 1 + 8  + 1 = 10
        [[ 1, 1, 3], # [1, 0, 1]] # 1 + 3 = 4
         [ 2, 1, 1]]] # [1, 0, 1]] # 2 + 1 = 3
    ])
    b = np.array(
        #bias 0
        [[[ 1, -1], #1 + 1 = 2, -1 + 7 = 6
         [ 4, 1]], #4 + 10 = 14, 1 + 4 = 5
        #bias 1
        [[ 0, -3], #0 + 1 = 1, -3 + 10 = 7
         [ -2, 1]]] #-2 + 4 = 2, 1 + 3 = 4
    )
    out = np.array([
        [[2, 6],
        [14, 5]],

       [[1, 7],
        [2, 4]]
    ])
    predict = test_operation(inp, w, b, 2)
    print("Case 5: \nout: {} ; \npredict: {}".format(out, predict))
    print("\n\n")



