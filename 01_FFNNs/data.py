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
import numpy as np
import matplotlib.pyplot as plt

# %%   
"""
Generate data for a bathtub function

"""

def plot_data_3D(X, Y, Z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    plt.title('Training Data')
    plt.show()

def bathtub():

    xs = np.linspace(1,10,450)
    ys = np.concatenate([np.square(xs[0:150]-4)+1, \
                         1+0.1*np.sin(np.linspace(0,3.14,90)), np.ones(60), \
                         np.square(xs[300:450]-7)+1])
    
        
    xs = xs / 10.0
    ys = ys / 10.0

    xs_c = np.concatenate([xs[0:240], xs[330:420]])
    ys_c = np.concatenate([ys[0:240], ys[330:420]])

    xs = tf.expand_dims(xs, axis = 1)
    ys = tf.expand_dims(ys, axis = 1)

    xs_c = tf.expand_dims(xs_c, axis = 1)
    ys_c = tf.expand_dims(ys_c, axis = 1)
    
    return xs, ys, xs_c, ys_c

def data_2D(func, x=np.linspace(-4, 4, 20), y=np.linspace(-4, 4, 20), plot=True):
    X, Y = np.meshgrid(x, y)
    p = np.stack([X.flatten(), Y.flatten()], axis=1)
    f = func(X.flatten(), Y.flatten())
    f = np.array(f)
    Z = f.reshape(X.shape)
    if plot: 
        plot_data_3D(X, Y, Z)
        
    return p, f, X, Y, Z

def gradient_2D(func, x=np.linspace(-4, 4, 20), y=np.linspace(-4, 4, 20)):
    X, Y = np.meshgrid(x, y)
    f = func(X.flatten(), Y.flatten())
    f = np.array(f)
    
    return f
