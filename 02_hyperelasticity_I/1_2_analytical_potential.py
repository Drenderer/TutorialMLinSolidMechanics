# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 1.2 Analytical potential
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import det, trace, inv
from tensorflow.math import log, square
import numpy as np

import data_handler_2 as dh

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
         
        C = tf.transpose(F, perm=[0,2,1]) @ F
        J = det(F)
        cof_C = det(C)[:,tf.newaxis,tf.newaxis] * inv(C)
        I_1 = trace(C)
        I_4 = trace(C @ self.G)
        I_5 = trace(cof_C @ self.G)
        
        #return tf.stack([I_1, J, I_4, I_5], axis=1)
        return I_1, J, I_4, I_5


class Potential(layers.Layer):
    
    def __init__(self, G=None):
        super().__init__()
        self.invariants = Invariants(G)
        
    def call(self, F):
        I_1, J, I_4, I_5 = self.invariants(F)
        
        W = 8*I_1 + 10*square(J) - 56*log(J) + 0.2*(square(I_4 )+ square(I_5)) - 44
        
        return W

class Stress(layers.Layer):
    
    def __init__(self, G=None):
        super().__init__()
        self.potential = Potential(G)
        
    def call(self, F):
        
        with tf.GradientTape() as g:
            g.watch(F)
            W = self.potential(F)
        P = g.gradient(W, F)
        return P

# %% Check invariant implementation
def check_invariants():
    print('Invariant test:')
    for name, file in dh.files.items():
        
        data = dh.read_file(file)
        invars = Invariants()(data['F'])
        invars = np.array(invars).T
        true_invars = np.loadtxt(f'invariants/I_{name}.txt')
        
        diff = invars - true_invars
        rel_diff = diff / true_invars
        
        max_abs_error = np.abs(diff.max())
        max_rel_error = np.abs(rel_diff.max())
        print('Maximal error for file {:<10} is: {:2e}. Maximal relative error: {:2e}'.format(name, max_abs_error, max_rel_error))
        
# %% Check Potential implementation
def check_potential():
    print('Potential test:')
    for name, file in dh.files.items():
        
        data = dh.read_file(file)
        W = Potential()(data['F'])
        W = np.array(W)
        
        diff = W - data['W']
        rel_diff = np.divide(diff, data['W'], out=np.zeros_like(diff), where=data['W']!=0)
        
        max_abs_error = np.abs(diff.max())
        max_rel_error = np.abs(rel_diff.max())
        print('Maximal error for file {:<10} is: {:2e}. Maximal relative error: {:2e}'.format(name, max_abs_error, max_rel_error))
        
# %% Check Piola Stress implementation
def check_stress():
    print('Stress test:')
    for name, file in dh.files.items():
        
        data = dh.read_file(file)
        P = Stress()(data['F'])
        P = np.array(P)
        
        diff = P - data['P']
        rel_diff = np.divide(diff, data['P'], out=np.zeros_like(diff), where=data['P']!=0)
        
        max_abs_error = np.abs(diff.max())
        max_rel_error = np.abs(rel_diff.max())
        print('Maximal error for file {:<10} is: {:2e}. Maximal relative error: {:2e}'.format(name, max_abs_error, max_rel_error))
        

# %% Do checks
check_invariants()
check_potential()
check_stress()