# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 2.1 Model implementation
Task 3.1 Model implementation

How does this restriction look like? Note that this also explains why both 
J and −J are used in the invariant vector. Hint: Have a look at the 
polyconvexity condition. Which physical / mathematical conditions are fulfilled
by which part of the model architecture?

All layers need to be restricted to non-neg weights.
Because the invariants are the frist layer.
Exception J: J is part of the polyconvexity condition and therefore must not comply to the extra restriction
"""

# %% Import
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import det, trace, inv
from tensorflow.keras.constraints import non_neg

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

# %% Define the Physics-augmented neural network model

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
        
        return tf.stack([I_1, J, -J, I_4, I_5], axis=1)
    
class ICNN(layers.Layer):
    """
    Basic input convex neural network for hyperelasticity:
        - Uses invariants as inputs (all layers are constraint to have positive weights)
        - Does not have a bias in the last layer
        - Softplus activation for hidden and identity activation for last layer
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

class PhysicsAugmentedNN(tf.keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.icnn = ICNN(**kwargs)
        self.invariants = Invariants()
        
    def call(self, F):
        with tf.GradientTape() as g:
            g.watch(F)
            i = self.invariants(F)
            W = self.icnn(i)
        P = g.gradient(W, F)
        return P, W

# %% Compile models

def compile_FFNN(**kwargs):
    Fs = tf.keras.Input(shape=(3,3))
    Ps = FFNN(**kwargs)(Fs)
    model = tf.keras.Model(inputs = Fs, outputs = Ps)
    model.compile('adam', 'mse')
    return model

def compile_physics_augmented_NN(loss_weights=[1,0], **kwargs):
    Fs = tf.keras.Input(shape=(3,3))
    Ps, Ws = PhysicsAugmentedNN(**kwargs)(Fs)
    model = tf.keras.Model(inputs = Fs, outputs = [Ps, Ws])
    model.compile('adam', 'mse', loss_weights=loss_weights)
    return model


