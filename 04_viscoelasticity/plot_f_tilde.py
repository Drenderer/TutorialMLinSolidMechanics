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
import plot as lp

def indxs2name(indxs):
    name = ''
    for i in indxs:
        name += (f'{chr(65+i)} & ')
    return name[:-3]

def plot_f_tilde(model, eps, gamma):
    
    factor = 1.1
    extent = [factor*np.min(eps) , factor*np.max(eps), factor*np.min(gamma) , factor*np.max(gamma)]

    grid_eps   = np.arange(extent[0], extent[1], 0.1)
    grid_gamma = np.arange(extent[2], extent[3], 0.1)
    X, Y = np.meshgrid(grid_eps, grid_gamma)
    
    x = np.reshape(X, (-1))
    y = np.reshape(Y, (-1))
    
    z = np.stack([x, y], axis=1)
    z = model.layers[2].cell.f_tilde(z)
    Z = np.reshape(z, X.shape)
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=400)
    ax.plot_surface(X, Y, Z, cmap='plasma',
                       linewidth=0, antialiased=False)
    
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'$\tilde{f}$')
    
    ax.set_title(r'Function $\tilde{f}$')
    
    plt.tight_layout()
    
    plt.show()
    
    fig, ax = plt.subplots(dpi=400)
    im = ax.imshow(Z, extent=extent, cmap='plasma', origin = "lower")
    fig.colorbar(im, ax=ax, orientation='vertical')
    # plt.xticks(x)
    # plt.yticks(y)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\gamma$')
    plt.title(r'$\tilde{f}(\epsilon, \gamma)$')
    
    for i in range(eps.shape[0]):
        plt.plot(eps[i,:,0], gamma[i,:,0], label=indxs2name([i]), color=list(lp.colors.values())[9-2*i])
    
    plt.legend()
    plt.show()
