from __future__ import print_function

import tensorflow as tf

from keras.models import Sequential
from config import *
from keras.layers import Activation, Input
from keras.layers.core import Dropout, Reshape, Flatten
from keras import optimizers
from keras.models import Model
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers import LSTM
from keras import regularizers
from keras.layers.local import LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import *

import sys
import logging
logging = logging.getLogger(__name__)


#TODO: apply weight decay with 0.0001
def init_graph():

    ###############################
    # VARIABLES
    ###############################

    inputs = Input(shape=(16,8))
    x = BatchNormalization(momentum=0.9)(inputs)
    x = Reshape((16, 8, 1))(x)

    #1. Conv (64 filters; 3x3 kernel)
    x = Conv2D(64, 3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=RandomUniform(-0.005, 0.005)
               )(x)
    x = BatchNormalization(momentum=0.9)(x) #do i have to specify which axis it is?
    x = Activation('relu')(x)

    #2. Conv (64 filters; 3x3 kernel)
    x = Conv2D(64, 3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=RandomUniform(-0.005, 0.005)
               )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    #3. Local (64 filters: 1x1 kernel)
    x = LocallyConnected2D(64, 1, strides=1,
                           kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer=RandomUniform(-0.005, 0.005)
                           )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    #4. Local (64 filters: 1x1 kernel)
    x = LocallyConnected2D(64, 1, strides=1,
                           kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer=RandomUniform(-0.005, 0.005)
                           )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    #5. Affine (512 units)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=RandomUniform(-0.005, 0.005)
              )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    #6. Affine (512 units)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=RandomUniform(-0.005, 0.005)
              )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    #7. Affine (128 units)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=RandomUniform(-0.005, 0.005)
              )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)


    #8. Affine (NUM_GESTURES units) Output layer
    x = Dense(NUM_GESTURES, kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=RandomUniform(-0.005, 0.005)
              )(x)
    logits = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=logits)
    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #paper seems to use softmax loss (I think this implies reducing crossentropy)

    model.summary()

    return model




def print_layershape(layername, inputs):
    print("{}:\t\t\t {}".format(layername, str(inputs.get_shape()) ))
    logging.info("{} \t\t\t {}".format(layername, str(inputs.get_shape())))

