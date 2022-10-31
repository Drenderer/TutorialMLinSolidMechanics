"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
import datetime
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""

class _NN(layers.Layer):
    def __init__(self, ns=[16, 16], convex=False, activation='softplus'):
        super().__init__()
        constr = {'kernel_constraint': non_neg()} if convex else {}
        # define hidden layers with activation functions
        self.ls = [layers.Dense(ns[0], activation)]
        self.ls += [layers.Dense(n, activation, **constr) for n in ns[1:]]
        # scalar-valued output function
        self.ls += [layers.Dense(1, **constr)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x

class _DNN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.NN = _NN(**kwargs)
            
    def call(self, x):     
        with tf.GradientTape() as g:
            g.watch(x)
            out = self.NN(x)
        grad = g.gradient(out, x)
        return out, grad    #tf.concat([out, grad], axis=1)

# %% Data generating layers

class _f_1(layers.Layer):
    def call(self, x, y):
        return x*x - y*y

class _f_2(layers.Layer):
    def call(self, x, y):
        return x*x + 0.5*y*y
    
class _Df_1(layers.Layer):
    def call(self, x, y):
        dx = 2*x
        dy = -2*y
        return tf.stack([dx, dy], axis=1)

class _Df_2(layers.Layer):
    def call(self, x, y):
        dx = 2*x
        dy = 1*y
        return tf.stack([dx, dy], axis=1)

# %%   
"""
main: construction of the NN model

"""

def main(in_shape=1, model_type='NN', **kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[in_shape])
    # define which (custom) layers the model uses
    if model_type == 'NN':
        ys = _NN(**kwargs)(xs)
    elif model_type == 'DNN':
        ys = _DNN(**kwargs)(xs)
    else:
        raise ValueError(f'Model type "{model_type}" unknown')
    # connect input and output
    model = tf.keras.Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model