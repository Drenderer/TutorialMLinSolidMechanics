# -*- coding: utf-8 -*-
"""
Import and visualize methods
"""

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from os import path as osp
from os import listdir
import random

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import det, trace, inv
from tensorflow.math import log, square

# %% Custom color Cycler 
import cycler
import matplotlib as mpl

colors = {'b1': '#002f3d',
          'b2': '#004352',
          'b3': '#006e7a',
          'b4': '#00999e',
          'b5': '#51DBDB',
          'o1': '#a12b00',
          'o2': '#d03400',
          'o3': '#f05c00',
          'o4': '#ff9604',
          'o5': '#ffa32b'}

custom_cycler = cycler.cycler('color', [list(colors.values())[i] for i in [1, 9, 3, 6, 2, 7]])
mpl.rcParams['axes.prop_cycle'] = custom_cycler

# %% Constants

files = {'biaxial': 'calibration/biaxial.txt',
         'pure_shear': 'calibration/pure_shear.txt',
         'uniaxial': 'calibration/uniaxial.txt',
         'biax_test': 'test/biax_test.txt',
         'mixed_test': 'test/mixed_test.txt'}
training_files = ['calibration/biaxial.txt', 'calibration/pure_shear.txt', 'calibration/uniaxial.txt']
test_files =     ['test/biax_test.txt', 'test/mixed_test.txt']

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
                  'e':              'Electric Field $e$',
                  'normalized P':   'normalized $1^{st}$ Piola Kirchhoff Stress $P$',
                  'W':              'Strain Energy Density $W$',
                  'normalized W':   'normalized Strain Energy Density $W$',
                  'weight':         'Training Weight'}
    label_replace_by = {'normalized P' : 'P',
                        'normalized W': 'W'}
    
    # Keyword arguments for the plots
    #       master kw takes priority, 1 is for first and 2 for the second plot in each axis
    #       scaler, vector, tensor kw are used to define special arguments for individual tensor components
    #       deault kw are the kw with least priority and like the master kw are used for every plot
    # Keyword arguments for first and second ('*') plot
    master_kw1 = {'linestyle':     '--'}
    master_kw2 = {'marker':        None,
                  'alpha':         0.7}
    scalar_kw = {'marker': 'o'}
    vector_kw = {(0): {'marker': '^'},
                 (1): {'marker': '>'},
                 (2): {'marker': 'v'},
                 'default': {'marker': 'o'},
                 'legend': True}
    tensor_kw = {(0,0): {'marker': 's', 'color': colors['o1']},
                 (1,1): {'marker': 's', 'color': colors['o3']},
                 (2,2): {'marker': 's', 'color': colors['o5']},
                 (0,1): {'marker': 'v', 'color': colors['b2']},
                 (1,0): {'marker': 'v', 'color': colors['b4']},
                 'default': {'marker': 'v', 'color': 'grey'},
                 'legend': True,
                 'legend_args': {'handlelength':1, 'columnspacing': 0.8,
                                 'loc': 'upper left',
                                 'ncol': 3, 'prop': {'size': 8}}}
    default_kw = {'markevery': 20,
                  'markersize': 5,
                  'linewidth': 2}
    
    # Allow the special variables to be overwritten by kwargs when calling the plot_data function
    master_kw1 = master_kw1 | kwargs.pop('master_kw1', {})
    master_kw2 = master_kw2 | kwargs.pop('master_kw2', {})
    scalar_kw  = scalar_kw  | kwargs.pop('scalar_kw', {})
    vector_kw  = vector_kw  | kwargs.pop('vector_kw', {})
    tensor_kw  = tensor_kw  | kwargs.pop('tensor_kw', {})
    default_kw = default_kw | kwargs.pop('default_kw', {})
    
    def replace(str):
        return replace_by.get(str) or str
    
    def plot_tensor(ax, tensor, ls=(), y_label='', first_plot=True):
        
        y_label = label_replace_by.get(y_label, y_label)
        
        shape = tensor.shape[1:]
        order = len(shape)
        
        master_kws = master_kw1 if first_plot else master_kw2
        
        if order == 0:   # Scalar
            kw = default_kw | scalar_kw | master_kws
            ax.plot(*ls, tensor, **kw)
            ax.set_ylabel(y_label)
        elif order == 1: # Vector
            for i in range(shape[0]):
                kw = default_kw | vector_kw.get((i), vector_kw['default']) | master_kws
                ax.plot(*ls, tensor[:, i], **kw)
            if vector_kw.get('legend', True):
                ax.legend(tuple(f'{i}' for i in range(1, shape[0]+1)))
            ax.set_ylabel(f'${y_label}_i$')
        elif order == 2: # Matrix
            legend_elements = []
            for j in range(shape[1]):
                for i in range(shape[0]):
                    kw = default_kw | tensor_kw.get((i,j), tensor_kw['default']) | master_kws
                    ax.plot(*ls, tensor[:, i, j], **kw)
                    
                    kw.update({'linestyle': 'None', 'markersize': '7'})
                    legend_elements.append(Line2D([0], [0], label=f'${y_label}_{{{i+1}{j+1}}}$', **kw))
            if first_plot and tensor_kw.get('legend', True):
                ax.legend(handles=legend_elements, **tensor_kw['legend_args'])
            ax.set_ylabel(f'${y_label}_{{i \ j}}$')
        else:
            ax.text(.4, .47,'I can\'t plot this')
    
    # Collect Data
    dont_plot = kwargs.pop('dont_plot', []) # Keys of the data that shouldn't be plotted
    metadata_str = ''   # For every metadata entry this will be extended 
    tensors = []        # For every data entry this will be filled with tuples like: (subplottitle, axis-label, np.ndarray, ax-id, plot_kwargs)
    ls = ()
    name2ax = {}
    def next_ax():
        return len(name2ax)
    dbl_plts = []       # List of data key, value pairs which keys start with '*' and should be plotted in the same axis as the data with the same key but without the '*'
    for key, value in data.items():
        if key in dont_plot:
            continue
        if key in metadata_keys:
            metadata_str += replace(key) + ': ' + value + ', '
        elif key == 'lam' and isinstance(value, np.ndarray) and value.ndim == 1:
            ls = (value,)   # This is necessary for the "argument unpacking hack" that is used in plot_tensor to dynamically switch between no and some x-data
        elif key[0] == '*' and isinstance(value, np.ndarray) and (key[1:] in data.keys()):
            dbl_plts.append((key[1:], value))
        elif isinstance(value, np.ndarray):
            tensors.append((replace(key), key, value, next_ax(), True))
            name2ax[key] = next_ax()
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                if isinstance(v, np.ndarray):
                    tensors.append((replace(key) + f' {i+1}', key, v, next_ax(), True))
                    name2ax[key] = next_ax()
                    
    for key, value in dbl_plts:     # Loop over all data entries with '*'
        if isinstance(value, np.ndarray):
            tensors.append((replace(key), key, value, name2ax[key], False))
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                if isinstance(v, np.ndarray):
                    tensors.append((replace(key) + f' {i+1}', key, v, name2ax[key], False))
    
    metadata_str = metadata_str[:-2]    # Remove last comma
    
    # Determine how many subplots are needed
    n_cols = kwargs.pop('n_cols', 2)
    n = len(name2ax)
    nv = n//n_cols + (n % n_cols > 0)   # Inter division but round up
    nh = n if n < n_cols else n_cols
    
    fig_title = kwargs.pop('title', metadata_str)
    
    kwargs.setdefault('dpi', 500)
    kwargs.setdefault('figsize', (6+2*nh, 4+2*nv))
    
    fig, axis = plt.subplots(nv, nh, **kwargs)    
    fig.suptitle(fig_title, fontsize=16)
    
    # Cerate the subplots
    for (title, y_label, tensor, ax_id, first_plot) in tensors:
        ax = axis.flatten()[ax_id] if n>1 else axis
        plot_tensor(ax, tensor, ls, y_label=y_label, first_plot=first_plot)
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
    name = osp.splitext(filename)[0]
    
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

def load_case_data(which='all', concat=False, normalize_weights=False, plot=False):
    if isinstance(which, str):
        file_dict = {'all': files.values(),
                     'train': training_files,
                     'test': test_files}
        f = file_dict.get(which)
    elif isinstance(which, list):
        f = [files[lc] for lc in which]
    else:
        raise ValueError(f'keyword argument "which" can\'t be of type {type(which)}')
    
    data = []
    for file in f:
        data.append(read_file(file, plot=(plot and not concat)))
        
    if concat:
        data = concatenate_data(data)

        if normalize_weights:
            factor = np.mean(data['weight'])
            data['weight'] /= factor
            
        if plot:
            plot_data(data, dpi=500)
        
    return data
    
# %% Data imprt & generation for Task 2: 4 Concentric sampled deformation gradients

class StressFromStrain(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.G = tf.constant([[4,  0,  0],
                              [0, .5,  0],
                              [0,  0, .5]])
        
    def call(self, F):

        with tf.GradientTape() as g:
            g.watch(F)
            
            C = tf.transpose(F, perm=[0,2,1]) @ F
            J = det(F)
            cof_C = det(C)[:,tf.newaxis,tf.newaxis] * inv(C)
            I_1 = trace(C)
            I_4 = trace(C @ self.G)
            I_5 = trace(cof_C @ self.G)
            
            W = 8*I_1 + 10*square(J) - 56*log(J) + 0.2*(square(I_4 )+ square(I_5)) - 44
        P = g.gradient(W, F)
        return P, W

def load_concentric(num_train_lp=50, plot=False, normalize_weights=True):
    file_list = listdir("concentric")
    # training_files = random.sample(file_list, num_train_lp)
    # test_files = list(set(file_list) - set(training_files))
    
    stress_from_strain = StressFromStrain()
    data = []
    for filename in file_list:
        path = osp.join('concentric', filename)
        F = np.loadtxt(path)
        F = np.reshape(F, (-1, 3, 3))
        
        # Calculate P and W
        P, W = stress_from_strain(F)
        P = np.array(P)
        W = np.array(W)
        
        # Calculate training weigths
        num_data = F.shape[0]
        weight = 1.0 / np.mean(norm(P, ord='fro', axis=(1,2)))
        weight = np.repeat(np.expand_dims(weight, axis=0), num_data, axis=0)
        
        # Get name of file
        name = osp.splitext(filename)[0]
        
        # Normalize Stress and Strain Energy
        P = P / 10
        W = W / 10
        
        d = {'load_case': name,
             #'material_type': 'Hyperelasticity',
             'F': F,
             'P': P,
             'W': W,
             'weight': weight}
        
        if plot:
            plot_data(d, dpi=300)
        
        data.append(d)
        
    # Split into training and test data
    training_indices = random.sample(range(100), num_train_lp)
    test_indices = list(set(range(100)) - set(training_indices))
    
    training_data = [data[i] for i in training_indices]
    training_data_concat = concatenate_data(training_data)
    
    if normalize_weights:
        factor = np.mean(training_data_concat['weight'])
        training_data_concat['weight'] /= factor
    
    test_data = [data[i] for i in test_indices]
    test_data_concat = concatenate_data(test_data)
    
    return training_data_concat, test_data_concat, training_data, test_data