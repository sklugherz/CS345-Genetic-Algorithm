import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD


class brain:

    NUM_FEATURES = 5 #number of columns in out data
    
    def __init__():
        nn = Sequential()
        nn.add(Dense(NUM_FEATURES * 2, activation='relu', input_shape=(NUM_FEATURES,)))
        nn.add(Dense(1)) #output layer

        opt = SGD(learning_rate=.005) #can play with this
        nn.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mse'])

    