import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

class brain:
    def __init__(self, weights):
        nn = tf.keras.Sequential()
        nn.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(5,)))
        nn.add(tf.keras.layers.Dense(1)) 
        opt = tf.keras.optimizers.Adam() 
        nn.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mse'])
