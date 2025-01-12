"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2023
"""


# %% Import modules

from matplotlib import pyplot as plt
import numpy as np

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
greyed_kwargs = {'color': 'gray', 'alpha': 0.5}

color_list = [list(colors.values())[i] for i in [1, 9, 3, 6, 2, 7]]
custom_cycler = cycler.cycler('color', color_list)
mpl.rcParams['axes.prop_cycle'] = custom_cycler


# %% Methods

def plot_data(eps, eps_dot, sig, omegas, As, title=None):
    
    
    n = len(eps[0])
    ns = np.linspace(0, 2*np.pi, n)
    
    fig, axes = plt.subplots(2, 2, dpi=500, figsize=(9, 7))
    if title is None:
        title = 'Training Data'
    fig.suptitle(title)
    
    ax = axes[0,0]
    for i in range(len(eps)):
                
        ax.plot(ns, sig[i], label = '$\\omega$: %.2f, $A$: %.2f' \
                 %(omegas[i], As[i]), linestyle='--', color=color_list[i])
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylabel('stress $\\sigma$')
        ax.set_xlabel('time $t$')
        ax.legend()
        
    ax = axes[0,1]
    for i in range(len(eps)):
        
        ax.plot(eps[i], sig[i], linestyle='--', color=color_list[i])
        ax.set_xlabel('strain $\\varepsilon$')
        ax.set_ylabel('stress $\\sigma$')
        
    ax = axes[1,0]
    for i in range(len(eps)):
        
        ax.plot(ns, eps[i], linestyle='--', color=color_list[i])
        ax.set_xlim([0, 2*np.pi])
        ax.set_xlabel('time $t$')
        ax.set_ylabel('strain $\\varepsilon$')
    
    ax = axes[1,1]
    for i in range(len(eps)):
        
        ax.plot(ns, eps_dot[i], linestyle='--', color=color_list[i])
        ax.set_xlim([0, 2*np.pi])
        ax.set_xlabel('time $t$')
        ax.set_ylabel(r'strain rate $\.{\varepsilon}$')
        
    plt.tight_layout()
    plt.show()
    
    
def plot_model_pred(eps, sig, sig_m, omegas, As, focus_on=None, title=None, training_idxs=None):
    
    kwarg_list = [{'color': color_list[i]} if (focus_on==None or i in focus_on) else greyed_kwargs for i in range(len(eps))]
    
    if training_idxs is not None:
        kwarg_list = []
        t_col = 0
        v_col = 9
        for i in range(len(eps)):
            if i in training_idxs:
                kwarg_list.append({'color': list(colors.values())[t_col]})
                t_col += 2
            else:
                kwarg_list.append({'color': list(colors.values())[v_col]})
                v_col -= 2
        #kwarg_list = [{'color': colors['b2']} if (i in training_idxs) else {'color': colors['o5']}  for i in range(len(eps))]
    
    n = len(eps[0])
    ns = np.linspace(0, 2*np.pi, n)
    
    fig, ax = plt.subplots(1, 2, dpi=500, figsize=(7,4))
    if title is None:
        title = 'Model Prediction'
    fig.suptitle(title)
        
    for i in range(len(eps)):
                
        ax[0].plot(ns, sig[i],linestyle='--', **kwarg_list[i])
        ax[0].plot(ns, sig_m[i], **kwarg_list[i])
        ax[0].set_xlim([0, 2*np.pi])
        ax[0].set_ylabel('Sstress $\\sigma$')
        ax[0].set_xlabel('Time $t$')
        
        ax[1].plot(eps[i], sig[i], linestyle='--', label = '$\\omega$: %.2f, $A$: %.2f'%(omegas[i], As[i]), **kwarg_list[i])
        ax[1].plot(eps[i], sig_m[i], **kwarg_list[i])
        ax[1].set_xlabel('Strain $\\varepsilon$')
        ax[1].set_ylabel('Stress $\\sigma$')
        ax[1].legend()
        
    plt.tight_layout()
    plt.show()

