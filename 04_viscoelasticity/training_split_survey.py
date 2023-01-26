"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Fabian Roth

Train models per dataset choice and evaluate on test cases
         
01/2023
"""


# %% Import modules
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import datetime
now = datetime.datetime.now

import data as ld
import plot as lp
import models_core as mc


# %% Useful functions

def indxs2name(indxs):
    name = ''
    for i in indxs:
        name += (f'{chr(65+i)} & ')
    return name[:-3]

def model2name(model):
    if model == 'naive_model':
        return 'Naive Model'
    elif model == 'analytic_maxwell':
        return 'Analytic Maxwell Model'
    elif model == 'ffnn_maxwell':
        return 'FFNN Maxwell Model'
    elif model == 'gsm_model':
        return 'GSM Model'
    elif model == 'ffnn_maxwell_extra':
        return 'Extended FFNN Maxwell Model'

# %%  Generate and visualize data 

E_infty = 0.5
E = 2
eta = 1

n = 100
omegas = [1,1,2]
As = [1,2,3]



eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)

model_type='gsm_model' # naive_model analytic_maxwell ffnn_maxwell ffnn_maxwell_extra gsm_model
t_indxs = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
num_models = 3

learning_rate = 0.001
epochs = 6000

results = []
t1 = now()
for t_indx in t_indxs:
    print(f'Training on {t_indx}')
    v_indx = list(set(range(len(As))) - set(t_indx))

    t_eps       = tf.gather(eps, t_indx, axis=0)
    t_eps_dot   = tf.gather(eps_dot, t_indx, axis=0)
    t_sig       = tf.gather(sig, t_indx, axis=0)
    t_dts       = tf.gather(dts, t_indx, axis=0)
    t_omegas    = [omegas[i] for i in t_indx]
    t_As        = [As[i] for i in t_indx]
    
    v_eps       = tf.gather(eps, v_indx, axis=0)
    v_eps_dot   = tf.gather(eps_dot, v_indx, axis=0)
    v_sig       = tf.gather(sig, v_indx, axis=0)
    v_dts       = tf.gather(dts, v_indx, axis=0)
    v_omegas    = [omegas[i] for i in v_indx]
    v_As        = [As[i] for i in v_indx]
    
    instance_results = []
    
    for m in range(num_models):
        print(f'Training Model {m}')
        tf.keras.backend.clear_session()
        model = mc.compile_RNN(model_type=model_type)
            
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
        h = model.fit([t_eps, t_dts], [t_sig],
                      validation_data=([v_eps, v_dts], [v_sig]),
                      epochs = epochs,  verbose = 0)
        
        instance_result = {'model':    model,
                           'loss':     h.history['loss'],
                           'val_loss': h.history['val_loss']}
        
        instance_results.append(instance_result)
        
    results.append(instance_results)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the models')

# %% Plot losses

for i, r in enumerate(results):
    plt.figure(1, dpi=600)
    for ir in r:
        plt.semilogy(ir['loss'], color=lp.colors['b2'], alpha=0.8)
        plt.semilogy(ir['val_loss'], color=lp.colors['o5'], alpha=0.8)
    plt.grid(which='both')
    plt.xlabel('Calibration Epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.title(f'{model2name(model_type)} trained on {indxs2name(t_indxs[i])}')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.show()

# %% Evalueate models on harmonic data

tl_avg = []
tl_std = []
vl_avg = []
vl_std = []
for i, r in enumerate(results):
    tl_a = []
    vl_a = []
    for ir in r:
        tl_a.append(ir['loss'][-1])
        vl_a.append(ir['val_loss'][-1])
      
    tl_std.append(np.std(tl_a))
    tl_avg.append(np.mean(tl_a))
    vl_std.append(np.std(vl_a))
    vl_avg.append(np.mean(vl_a))
    
    
x = np.arange(len(results))
fig, ax = plt.subplots(dpi=600, figsize=(6,4))
ax.bar(x-0.2, tl_avg, yerr=tl_std, label='Trainig Loss',
       align='center', color=lp.colors['b2'], ecolor=lp.colors['b4'], capsize=6, width=0.4)
ax.bar(x+0.2, vl_avg, yerr=vl_std, label='Validation Loss',
       align='center', color=lp.colors['o5'], ecolor=lp.colors['o3'], capsize=6, width=0.4)

# ax.set_ylim([1e-9,2e0])

ax.set_xticks(x, [indxs2name(ti) for ti in t_indxs], zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title(f'Average loss of {model2name(model_type)} per training split')
ax.set_xlabel('Trainig Loadpath')
ax.set_ylabel('log$_{10}$ MSE')
plt.xticks(rotation='horizontal')
plt.yscale('log')
plt.legend(loc='upper right')
plt.show()
    

# %% Evalueate model on relaxation data

r_eps, r_eps_dot, r_sig, r_dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)

vl_avg = []
vl_std = []
for i, r in enumerate(results):
    vl_a = []
    for ir in r:
        vl_a.append(ir['model'].evaluate([r_eps, r_dts], r_sig))
      
    vl_std.append(np.std(vl_a))
    vl_avg.append(np.mean(vl_a))
        
for i in range(len(results)+1):
    x = np.arange(len(results))
    fig, ax = plt.subplots(dpi=600, figsize=(5,4))
    ax.bar(x, vl_avg, yerr=vl_std, label='Validation Loss',
           align='center', color=lp.colors['o5'], ecolor=lp.colors['o3'], capsize=6, width=0.8)
    if i < len(results):
        ax.bar(i, vl_avg[i], yerr=vl_std[i],
               align='center', color=lp.colors['o1'], ecolor=lp.colors['o3'], capsize=6, width=0.8)
    
    #ax.set_ylim([9e-1, 1e0])
    
    ax.set_xticks(x, [indxs2name(ti) for ti in t_indxs], zorder=3)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f'Average loss of {model2name(model_type)} on relaxation data per training split')
    ax.set_xlabel('Trainig Loadpath')
    ax.set_ylabel('log$_{10}$ MSE')
    plt.xticks(rotation='horizontal')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()


# %% Plot relaxation

for i, r in enumerate(results):
    vl_a = []
    # for ir in r:
    ir = r[0]
    r_sig_m = ir['model']([r_eps, r_dts])
    
    title = f'Relaxation prediction of {model2name(model_type)} trained on {indxs2name(t_indxs[i])}'
    lp.plot_model_pred(r_eps, r_sig, r_sig_m, omegas, As, title=title, training_idxs = t_indxs[i])
    
# %% Plot harmonic data

for i, r in enumerate(results):
    vl_a = []
    # for ir in r:
    ir = r[0]
    sig_m = ir['model']([eps, dts])
    
    title = f'Prediction of {model2name(model_type)} trained on {indxs2name(t_indxs[i])}'
    lp.plot_model_pred(eps, sig, sig_m, omegas, As, title=title, training_idxs = t_indxs[i])