# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 2.1 Model implementation
"""

# %% Import
import tensorflow as tf
from tensorflow.keras import layers

# %% Define naive FFNN

class FFNN(layers.Layer):
    """ The naive Model - Task 2.1"""
    def __init__(self, ns=[4, 4], activation='softplus'):
        super().__init__()
        self.ls = [layers.Dense(n, activation) for n in ns]
        # scalar-valued output function
        self.ls += [layers.Dense(9)]
            
    def call(self, F):     
        
        C = tf.transpose(F, perm=[0,2,1]) @ F
        x = tf.reshape(C, (-1, 9))
        x = tf.concat([x[:, 0:3], x[:, 4:6], x[:, 8:9]], axis=1)
        for l in self.ls:
            x = l(x)
            
        P = tf.reshape(x, (-1,3,3))
        return P


# %% main: construction of the NN model

def compile_FFNN(**kwargs):
    Fs = tf.keras.Input(shape=(3,3))
    Ps = FFNN(**kwargs)(Fs)
    model = tf.keras.Model(inputs = Fs, outputs = Ps)
    model.compile('adam', 'mse')
    return model