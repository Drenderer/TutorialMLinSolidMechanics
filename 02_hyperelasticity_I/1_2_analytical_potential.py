# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 1.2 Analytical potential
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import det, trace, inv
import numpy as np

import data_handler as dh

# %% Calculate invariants

class Invariants(layers.Layer):
    
    def __init__(self, G=None):
        super().__init__()
        
        if G is None:
            G = tf.constant([[4,  0,  0],
                             [0, .5,  0],
                             [0,  0, .5]])
        self.G = G
    
    def call(self, F):
         
        C =  tf.transpose(F, perm=[0,2,1]) @ F
        J = det(F)
        cof_C = det(C)[:,tf.newaxis,tf.newaxis] * inv(C)
        I_1 = trace(C)
        I_4 = trace(C @ self.G)
        I_5 = trace(cof_C @ self.G)
        
        return tf.stack([I_1, J, I_4, I_5], axis=1)


# %% Assert correct implementation

for name, file in dh.files.items():
    
    data = dh.read_file(file)
    invars = Invariants()(data['F'])
    true_invars = np.loadtxt(f'invariants/I_{name}.txt')
    
    assert(tf.reduce_all(tf.equal(invars, true_invars)))