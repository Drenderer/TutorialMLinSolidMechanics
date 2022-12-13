# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 3: 1.2 Invariant-based model
"""

# %% Import
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import det, trace, inv
from tensorflow.keras.constraints import non_neg


# %% Define the Invariant based Model

class Invariants(layers.Layer):
    
    def __init__(self, G=None):
        super().__init__()
    
    def call(self, F):
        
        C = tf.transpose(F, perm=[0,2,1]) @ F
        J = det(F)
        cof_C = det(C)[:,tf.newaxis,tf.newaxis] * inv(C)
        
        I_1  = trace(C)
        I_2  = trace(cof_C)
        I_7  = trace(C**2)
        I_11 = trace(cof_C**2)
        
        return tf.stack([I_1, I_2, J, -J, I_7, I_11], axis=1)
    
class InvariantICNN(layers.Layer):
    """
    Basic input convex neural network for hyperelasticity:
        - Uses invariants as inputs (all layers are constraint to have positive weights)
        - Softplus activation for hidden and identity activation for last layer
        - One node in the last layer
    """
    
    def __init__(self, ns=[4]):
        """
        Parameters
        ----------
        ns : List, optional
             List of the numbers of nodes per layer. The default is [4]. Output layer excluded

        """
        
        super().__init__()
        self.ls = [layers.Dense(n, 
                                'softplus', 
                                kernel_constraint=non_neg()) 
                   for n in ns]
        self.ls += [layers.Dense(1,
                                 kernel_constraint=non_neg())]
        
    def call(self, x):
        for l in self.ls:
            x = l(x)
        return x

class InvariantBasedModel(tf.keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.icnn = InvariantICNN(**kwargs)
        self.invariants = Invariants()
        
    def call(self, F):
        with tf.GradientTape() as g:
            g.watch(F)
            i = self.invariants(F)
            W = self.icnn(i)
        P = g.gradient(W, F)
        return P, W
    

# %% Deformation Gradient based model

class PolyconvexArguments(layers.Layer):
    
    def __init__(self, G=None):
        super().__init__()
    
    def call(self, F):
        
        J = det(F)[:,tf.newaxis]    # Newaxis necessary for later concatenating
        cof_F = det(F)[:,tf.newaxis,tf.newaxis] * inv(F)
        
        # Flatten F and cof(F) to batch x 9 array for concatenating
        flat_F = tf.reshape(F, (-1,9))
        flat_cof_F = tf.reshape(cof_F, (-1,9))

        return tf.concat([flat_F, flat_cof_F, J], axis=1)

class DeformationGradientICNN(layers.Layer):
    """
    Basic input convex neural network for hyperelasticity:
        - Uses polyconvexity arguments as inputs (all layers except the first are constraint to have positive weights)
        - Softplus activation for hidden and identity activation for last layer
        - One node in the last layer
    """
    
    def __init__(self, ns=[4]):
        """
        Parameters
        ----------
        ns : List, optional
             List of the numbers of nodes per layer. The default is [4]. Output layer excluded

        """
        
        super().__init__()
        self.ls  = [layers.Dense(ns[0], 
                                 'softplus')]
        self.ls += [layers.Dense(n, 
                                 'softplus', 
                                 kernel_constraint=non_neg()) 
                    for n in ns[1:]]
        self.ls += [layers.Dense(1,
                                 kernel_constraint=non_neg())]
        
    def call(self, x):
        for l in self.ls:
            x = l(x)
        return x

class DeformationGradientBasedModel(tf.keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.icnn = DeformationGradientICNN(**kwargs)
        self.poly_args = PolyconvexArguments()
        
    def call(self, F):

        with tf.GradientTape() as g:
            g.watch(F)
            i = self.poly_args(F)
            W = self.icnn(i)
        P = g.gradient(W, F)
        return P, W


# %% Compile models
def compile_invariant_based_model(loss_weights=[1,0], **kwargs):
    Fs = tf.keras.Input(shape=(3,3))
    Ps, Ws = InvariantBasedModel(**kwargs)(Fs)
    model = tf.keras.Model(inputs = Fs, outputs = [Ps, Ws])
    model.compile('adam', 'mse', loss_weights=loss_weights)
    return model

def compile_deformation_gradient_based_model(loss_weights=[1,0], **kwargs):
    Fs = tf.keras.Input(shape=(3,3))
    Ps, Ws = DeformationGradientBasedModel(**kwargs)(Fs)
    model = tf.keras.Model(inputs = Fs, outputs = [Ps, Ws])
    model.compile('adam', 'mse', loss_weights=loss_weights)
    return model
