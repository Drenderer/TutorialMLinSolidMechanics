# -*- coding: utf-8 -*-
"""
Import and visualize methods
"""

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from os import path as osp


# %% Custom color Cycler 
# import cycler
# import matplotlib.colors as mcolors

# # sample the colormaps that you want to use. Use 128 from each so we get 256
# # colors in total
# colors1 = plt.cm.inferno(np.linspace(0, 0.8, 128))
# colors2 = plt.cm.viridis(np.linspace(0.8, 0.3, 64))

# # combine them and build a new colormap
# colors = np.vstack((colors1, colors2))
# mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# color = mymap(np.linspace(0, 1, 9))
# custom_cycler = cycler.cycler('color', color)

# %% Constants

files = {'biaxial': 'calibration/biaxial.txt',
         'pure_shear': 'calibration/pure_shear.txt',
         'uniaxial': 'calibration/uniaxial.txt',
         'biax_test': 'test/biax_test.txt',
         'mixed_test': 'test/mixed_test.txt'}
training_files = {'biaxial': 'calibration/biaxial.txt',
                  'pure_shear': 'calibration/pure_shear.txt',
                  'uniaxial': 'calibration/uniaxial.txt'}
test_files = {'biax_test': 'test/biax_test.txt',
              'mixed_test': 'test/mixed_test.txt'}

# %% Module methods

def plot_data(data, **kwargs):
    
    metadata_keys = ['material_type', 'material_law', 'load_case']
    replace_by = {'material_type':  'Material Type',
                  'material_law':   'Material Law',
                  'load_case':      'Load Case',
                  'mlin':           'Material Law Input',
                  'mlout':          'Material Law Output',
                  'lam':            'Load Case Parameter $\lambda$',
                  'F':              'Deformation Gradient $F$',
                  'P':              '$1^{st}$ Piola Kirchhoff Stress $P$',
                  'd':              'Electric Displacement Field $d$',
                  'e':              'Electric Field $e$'}
    # Keyword arguments for first and second ('*') plot
    plot_kw1 = {'linestyle':     '--',
                'marker':        'o',
                'markevery':     20}
    plot_kw2 = {'marker':        None,
                'markevery':     20,
                'alpha':         0.6}
    
    def replace(str):
        return replace_by.get(str) or str
    
    def plot_tensor(ax, tensor, ls=(), y_label='', **kwargs):
        shape = tensor.shape[1:]
        order = len(shape)
        num = tensor.shape[0]
        
        #ax.set_prop_cycle(custom_cycler)
        ax.plot(*ls, tensor.reshape(num,-1), **kwargs)
        if order == 0:   # Scalar
            ax.set_ylabel(y_label)
        elif order == 1: # Vector
            ax.legend(tuple(f'{i}' for i in range(1, shape[0]+1)))
            ax.set_ylabel(f'{y_label}$_i$')
        elif order == 2: # Matrix
            ax.legend(tuple(f'({i}, {j})' for i in range(1, shape[0]+1) for j in range(1, shape[1]+1)))
            ax.set_ylabel(f'{y_label}'+'$_{i \ j}$')
        elif order != 0:
            print(f'Define legend yourself for tensor order {order}')
    
    # Collect Data
    metadata_str = ''   # For every metadata entry this will be extended 
    tensors = []        # For every data entry this will be filled with tuples like: (subplottitle, axis-label, np.ndarray, ax-id, plot_kwargs)
    ls = ()
    name2ax = {}
    def next_ax():
        return len(name2ax)
    dbl_plts = []       # List of data key, value pairs which keys start with '*' and should be plotted in the same axis as the data with the same key but without the '*'
    for key, value in data.items():
        if key in metadata_keys:
            metadata_str += replace(key) + ': ' + value + ', '
        elif key == 'lam' and isinstance(value, np.ndarray) and value.ndim == 1:
            ls = (value,)   # This is necessary for the "argument unpacking hack" that is used in plot_tensor to dynamically switch between no and some x-data
        elif key[0] == '*' and isinstance(value, np.ndarray) and (key[1:] in data.keys()):
            dbl_plts.append((key[1:], value))
        elif isinstance(value, np.ndarray):
            tensors.append((replace(key), key, value, next_ax(), plot_kw1))
            name2ax[key] = next_ax()
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                if isinstance(v, np.ndarray):
                    tensors.append((replace(key) + f' {i+1}', key, v, next_ax(), plot_kw1))
                    name2ax[key] = next_ax()
                    
    for key, value in dbl_plts:     # Loop over all data entries with '*'
        if isinstance(value, np.ndarray):
            tensors.append((replace(key), key, value, name2ax[key], plot_kw2))
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                if isinstance(v, np.ndarray):
                    tensors.append((replace(key) + f' {i+1}', key, v, name2ax[key], plot_kw2))
    
    metadata_str = metadata_str[:-2]    # Remove last comma
    
    # Determine how many subplots are needed
    n = len(name2ax)
    nv = n//2 + (n % 2 > 0)   # Inter division but round up
    nh = 1 if n==1 else 2
    
    fig_title = kwargs.pop('title', metadata_str)
    
    fig, axis = plt.subplots(nv, nh, figsize = (6+2*nh, 4+2*nv), **kwargs)    
    fig.suptitle(fig_title, fontsize=16)
    
    # Cerate the subplots
    for (title, y_label, tensor, ax_id, plot_kw) in tensors:
        ax = axis.flatten()[ax_id] if n>1 else axis
        plot_tensor(ax, tensor, ls, y_label=y_label, **plot_kw)
        ax.set_prop_cycle(None)
        ax.grid(visible=True, which='both')
        ax.set_title(title)
        # x-label
        if ls:
            ax.set_xlabel('Load Step Parameter $\lambda$')
        else:
            ax.set_xlabel('Load Step')
    
    plt.tight_layout()
    plt.show()

def read_file(path, plot=False):
    
    data = np.loadtxt(path)
    
    F, P, W = np.split(data, [9, 18], axis=1)
    F = np.reshape(F, (-1, 3, 3))
    P = np.reshape(P, (-1, 3, 3))
    W = np.squeeze(W)
    
    # Calculate training weigths
    num_data = F.shape[0]
    weight = 1.0 / np.mean(norm(P, ord='fro', axis=(1,2)))
    weight = np.repeat(np.expand_dims(weight, axis=0), num_data, axis=0)
    
    # Get name of file
    filename = osp.basename(path)
    name, _ = osp.splitext(filename)
    
    data = {'load_case': name,
            #'material_type': 'Hyperelasticity',
            'F': F,
            'P': P,
            'W': W,
            'weight': weight}
    
    if plot:
        plot_data(data, dpi=500)
    
    return data

def concatenate_data(data_list):
    d0 = data_list[0]   # First data dict
    # Check if all dicts have the same keys
    if not all(x.keys() == d0.keys() for x in data_list):
        raise ValueError('The data dictionaries dont have all the same keys')
    
    data = {}
    for key, value in d0.items():
        if isinstance(value, np.ndarray):   # concatenate numpy arrays
            data[key] = np.concatenate([d[key] for d in data_list])
        elif all(x[key] == d0[key] for x in data_list): # If all dics have identical key value pair keep it in the final data dict
            data[key] = value
            
    return data

def load_case_data(which='all', concat=False, plot=False):
    if isinstance(which, str):
        file_dict = {'all': files,
                     'train': training_files,
                     'test': test_files}
        f = file_dict.get(which)
    elif isinstance(which, list):
        f = {}
        for lc in which:
            f[lc] = files[lc]
    
    data = []
    for name, file in f.items():
        data.append(read_file(file, plot=(plot and not concat)))
        
    if concat:
        data = concatenate_data(data)
        if plot:
            plot_data(data)
        
    return data
    
