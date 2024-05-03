import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

class brain:
    def __init__():
        nn = tf.keras.Sequential()
        nn.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(5,)))
        # nn.add(tf.keras.layers.Dense(8, activation='relu'))
        # nn.add(tf.keras.layers.Dense(8, activation='relu')) #i think the first layer is all we need
        nn.add(tf.keras.layers.Dense(1)) #output layer
                         #could be 2, acitvation='softmax'

        opt = tf.keras.optimizers.Adam() #can play with this
        nn.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mse'])
