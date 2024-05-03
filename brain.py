import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

class brain:
    def __init__():
        nn = tf.keras.Sequential()
        nn.add(tf.keras.layers.Dense(80, activation='relu', input_shape=(75,)))
        nn.add(tf.keras.layers.Dense(3))
        opt = tf.keras.optimizers.Adam() 
        nn.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mse'])

    def __init__(self, weights):
        nn = tf.keras.Sequential()
        nn.add(tf.keras.layers.Dense(80, activation='relu', input_shape=(75,)))
        nn.add(tf.keras.layers.Dense(3))
        opt = tf.keras.optimizers.Adam() 
        nn.set_weights(weights)


        nn.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mse'])


    def get_weights(self):
        return [self.nn.layers[0].get_weights(), self.nn.layers[1].get_weights]
    
    def set_weights(self, weights):
        pass

    
    """
    >>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
Arguments

weights: a list of NumPy arrays. The number of arrays and their shape must match number of the dimensions of the weights of the layer (i.e. it should match the output of get_weights).
Raises

ValueError: If the provided weights list does not match the layer's specifications.
"""

    
