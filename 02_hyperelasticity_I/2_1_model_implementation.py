# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 2.1 Model implementation
This Code isn't used in the following tasks because I cant import modules 
starting with numbers. 
Instead this implementation can be found in the models_core.py file.

This naive model implements the:
    - objectivity condition     (since it uses C)

"""

# %% Import
import tensorflow as tf
from tensorflow.keras import layers

# %% Define naive FFNN

class FFNN(layers.Layer):
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