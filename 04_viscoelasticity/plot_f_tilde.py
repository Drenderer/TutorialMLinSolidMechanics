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

def plot_f_tilde(model):
    eps   = np.arange(-5, 5, 0.25)
    gamma = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(eps, gamma)
    
    x = np.reshape(X, (-1))
    y = np.reshape(Y, (-1))
    
    z = np.stack([x, y], axis=1)
    z = model.layers[2].cell.f_tilde(z)
    Z = np.reshape(z, X.shape)
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=400)
    ax.plot_surface(X, Y, Z, cmap=cm.inferno,
                       linewidth=0, antialiased=False)
    
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'$\tilde{f}$')
    
    ax.set_title(r'Function $\tilde{f}$')
    
    plt.tight_layout()
    
    plt.show()
