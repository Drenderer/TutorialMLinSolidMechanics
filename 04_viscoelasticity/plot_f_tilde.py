"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity

Script for plotting the f_tile from the evolution equation
==================
Authors: Fabian Roth
         
01/2023
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf

activations = [tf.nn.softplus, tf.keras.activations.linear]
def plot_f_tilde(weights):
    eps   = np.arange(-5, 5, 0.25)
    gamma = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(eps, gamma)
    
    x = np.reshape(X, (-1))
    y = np.reshape(Y, (-1))
    
    z = np.stack([x, y], axis=1)
    
    num_layers = round(len(weights)/2)
    
    for l in range(num_layers):
        z = np.dot(z, weights[2*l])
        z += weights[2*l + 1]
        z = activations[l==num_layers-1](z)
        
    Z = np.reshape(z, X.shape)
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap=cm.inferno,
                       linewidth=0, antialiased=False)
    # ax.plot_surface(X, Y, 2*np.ones_like(X), cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()