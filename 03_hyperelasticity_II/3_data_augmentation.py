# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:48:05 2022

Task 3: 3 Data augmentation
"""

# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime 
now = datetime.now
import pickle

import data_handler_3 as dh
import models_core as mc


# %% Useful function
def get_info_string(loss_weights):
    if loss_weights == [1,1]:
        return 'Trained on both P and W'
    elif loss_weights == [1,0]:
        return 'Trained on P only'
    elif loss_weights == [0,1]:
        return 'Trained on W only'
    else:
        return 'Trained on both P and W, but differently weighted'


# %% Import data
train = dh.load_case_data('train', concat=True, 
                          normalize_weights=True, plot=True)        # Data dict
pre_train = dh.augment_data(train,
                            symmetry_group=dh.cubic_group)
aug_train = dh.augment_data(train,
                            symmetry_group=dh.cubic_group,
                            objectivity_group=8,
                            reduce_to=100_000
                            )

validation = dh.load_case_data('test', concat=True)
validation = dh.augment_data(validation,
                             symmetry_group=dh.cubic_group,
                             objectivity_group=1_000,
                             reduce_to=1_000)

test = dh.load_case_data('all')                                     # List of Loadcase data dicts
aug_test = [dh.augment_data(t,
                            symmetry_group=dh.cubic_group,
                            objectivity_group=1_000,
                            reduce_to=1_000)
            for t in test]                                          # List of augmented data dicts for every load case


# %% Train multiple Models to average the results
model_args = {'ns': [32, 32, 32]}
loss_weights = [1, 1]
precal_epochs = 300    # epochs used to train on unaugmented data before training on augmented data
precal_learning_rate = 0.01
epochs = 750
learning_rate = 0.005
weighted_load_cases = True


results = []

# Model calibration
model = mc.compile_deformation_gradient_based_model(loss_weights=loss_weights,
                                                    **model_args)

info_string = get_info_string(loss_weights)

t1 = now()    

try:
    model.load_weights('./checkpoints/my_checkpoint_2')
    
    with open('./checkpoints/precal_losses_2.pkl', 'rb') as file:
        precal_loss, precal_val_loss = pickle.load(file)
    
except:
    print('creatng pretrained model checkpoint')
    # Pre train
    tf.keras.backend.set_value(model.optimizer.learning_rate, precal_learning_rate)
    weights = pre_train['weight'] if weighted_load_cases else None
    precal_h = model.fit(pre_train['F'], [pre_train['normalized P'], pre_train['normalized W']], 
                         validation_data = [validation['F'], [validation['normalized P'], validation['normalized W']]],
                         sample_weight=weights,
                         epochs = precal_epochs,  verbose = 1,
                         batch_size=512)
    
    precal_loss = precal_h.history['loss']
    precal_val_loss = precal_h.history['val_loss']
    
    with open('./checkpoints/precal_losses_2.pkl', 'wb') as file:
        pickle.dump([precal_loss, precal_val_loss], file)
    
    model.save_weights('./checkpoints/my_checkpoint_2')





tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
weights = aug_train['weight'] if weighted_load_cases else None
h = model.fit(aug_train['F'], [aug_train['normalized P'], aug_train['normalized W']], 
              validation_data = [validation['F'], [validation['normalized P'], validation['normalized W']]],
              sample_weight=weights,
              epochs = epochs,  verbose = 1,
              batch_size=1024)
t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# Evaluate Model
load_case_losses = {}
for t in test:
    l = model.evaluate(t['F'], [t['normalized P'], t['normalized W']])
    load_case_losses[t['load_case']] = l
    
aug_load_case_losses = {}
for t in aug_test:
    l = model.evaluate(t['F'], [t['normalized P'], t['normalized W']])
    aug_load_case_losses[t['load_case']] = l
   

# Write results
model_results = {'model': model,
                 'info_string': info_string,
                 'loss': precal_loss + h.history['loss'],
                 'val_loss': precal_val_loss + h.history['val_loss'],
                 'epochs': epochs,
                 'learning_rate': learning_rate,
                 'load_case_losses': load_case_losses,
                 'aug_load_case_losses': aug_load_case_losses}
results.append(model_results)
    

# %% Plot trainig losses
plt.figure(1, dpi=600, figsize=(6,4))
for r in results:
    plt.semilogy(r['loss'], color=dh.colors['b2'], alpha=0.8)
    plt.semilogy(r['val_loss'], color=dh.colors['o5'], alpha=0.8)
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend(['Training loss', 'Test loss'])

# %% Plot losses for P and W per load case 

load_case_names = [t['load_case'] for t in test]
avg_losses = []         # contains two lists first for loss on P second for loss on W
std_losses = []         # same as above
for i in [1,2]:     # Loop over both loss on P second for loss on W
    avg = []
    std = []
    for lc in load_case_names:
        loss_aggregate = []
        for r in results:
            loss_aggregate.append(r['load_case_losses'][lc][i])
            
        avg_loss = np.mean(loss_aggregate)
        std_loss = np.std(loss_aggregate)
        avg.append(avg_loss)
        std.append(std_loss)
    avg_losses.append(avg)
    std_losses.append(std)

x = np.array([1,2,3,4, 6,7,8,9])
x1 = x[:4]
x2 = x[4:]
fig, ax = plt.subplots(dpi=600, figsize=(4,4))
ax.bar(x1-0.2, avg_losses[0][:4], yerr=std_losses[0][:4], label=r'$P$ training loss',
       align='center', ecolor=dh.colors['b4'], capsize=4, width=0.4,
       color=dh.colors['b2'])
ax.bar(x1+0.2, avg_losses[1][:4], yerr=std_losses[1][:4], label=r'$W$ training loss',
       align='center', ecolor=dh.colors['b4'], capsize=4, width=0.4,
       color=dh.colors['b3'])
ax.bar(x2-0.2, avg_losses[0][4:], yerr=std_losses[0][4:], label=r'$P$ test loss',
       align='center', ecolor=dh.colors['o2'], capsize=4, width=0.4,
       color=dh.colors['o3'])
ax.bar(x2+0.2, avg_losses[1][4:], yerr=std_losses[1][4:], label=r'$W$ test loss',
       align='center', ecolor=dh.colors['o3'], capsize=4, width=0.4,
       color=dh.colors['o5'])

ax.set_ylim([10e-4,2*10e0])

ax.set_xticks(x, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('''Average loss per load case''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()

# %% Plot losses for P per original and fully augmented load case 

load_case_names = [t['load_case'] for t in test]
avg_losses = []         # contains two lists first for loss on P second for loss on W
std_losses = []         # same as above
for i in ['load_case_losses', 'aug_load_case_losses']:     # Loop over both loss on non-augmented and augmented data
    avg = []
    std = []
    for lc in load_case_names:
        loss_aggregate = []
        for r in results:
            loss_aggregate.append(r[i][lc][0])
            
        avg_loss = np.mean(loss_aggregate)
        std_loss = np.std(loss_aggregate)
        avg.append(avg_loss)
        std.append(std_loss)
    avg_losses.append(avg)
    std_losses.append(std)

x = np.array([1,2,3,4, 6,7,8,9])
fig, ax = plt.subplots(dpi=600, figsize=(4,4))
ax.bar(x-0.2, avg_losses[0], yerr=std_losses[0], label=r'Loss on original data', color=dh.colors['b2'], 
       align='center', ecolor=dh.colors['b4'], capsize=4, width=0.4)
ax.bar(x+0.2, avg_losses[1], yerr=std_losses[1], label=r'Loss on augmented data', color=dh.colors['o5'],
       align='center', ecolor=dh.colors['o3'], capsize=4, width=0.4)

ax.set_ylim([10e-2,2*10e-1])

ax.set_xticks(x, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('''Average loss per load case''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()

# %% Examine the stress / energy prediction of the model in the reference configuration F = I.

tF = tf.constant([np.eye(3)])
tP, tW = model(tF)
print(f'For F = I the model predicts: \nW = {tW}, \nP = \n{tP} \n=\n{np.round(tP,2)}')

# %% Plot an example loadcase

#lc = 'mixed_test'
for lc in dh.files.keys():
    lc_test = dh.read_file(dh.files[lc])
    lc_model = results[0]['model']
    lc_test['*normalized P'], lc_test['*normalized W'] = lc_model(lc_test['F'])
    lc_test['*normalized P'] = np.array(lc_test['*normalized P'])
    lc_test['*normalized W'] = np.array(lc_test['*normalized W']).squeeze()
    del lc_test['F'], lc_test['weight'], lc_test['W'], lc_test['P']
    dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5))
    
# %% Check how good the model is at approximatin objectivity


for lc in dh.files.keys():
    lc_model = results[0]['model']
    
    lc_test = dh.read_file(dh.files[lc])
    # augment data
    P_list = []
    for i in range(1000):
        Q = dh.Rotation.random(1).as_matrix().squeeze()
        aug_lc = dh.augment_data(lc_test, objectivity_group=[Q])
        
        aug_lc['*normalized P'], _ = lc_model(aug_lc['F'])
        aug_lc['*normalized P'] = np.array(aug_lc['*normalized P'])
        # unaugment data
        P_list.append(Q.T @ aug_lc['*normalized P'])
    
    # for 
    colors = [dh.colors['o1'], dh.colors['o3'], dh.colors['o5'], dh.colors['b2'], dh.colors['b4']]
    plot_components = [0,4,8] #[0,4,8, 1, 3]
    
    plt.figure(1, dpi=600, figsize=(6,5))
    
    # Plot correct
    p = np.reshape(lc_test['normalized P'], (-1,9))
    p = np.take(p, plot_components, axis=1)
    for i in range(p.shape[1]):
        c = colors[i]
        plt.plot(p[:,i], color=c, marker='s', markevery=20, markersize=5, linewidth=2, linestyle='--')
    
    # Plot outline
    P = np.array(P_list)
    P_max = np.max(P, axis=0)
    P_min = np.min(P, axis=0)        
    P_max = np.reshape(P_max, (-1,9))
    P_max = np.take(P_max, plot_components, axis=1)
    P_min = np.reshape(P_min, (-1,9))
    P_min = np.take(P_min, plot_components, axis=1)
    for i in range(p.shape[1]):
        c = colors[i]
        plt.fill_between(np.arange(P_max.shape[0]), P_max[:,i], P_min[:,i], color=c, alpha=0.2)
    
    
    # For legend
    master_kw1 = {'linestyle':     '--'}
    master_kw2 = {'marker':        None,
                  'alpha':         0.7}
    tensor_kw = {(0,0): {'marker': 's', 'color': dh.colors['o1']},
                 (1,1): {'marker': 's', 'color': dh.colors['o3']},
                 (2,2): {'marker': 's', 'color': dh.colors['o5']},
                 (0,1): {'marker': 'v', 'color': dh.colors['b2']},
                 (1,0): {'marker': 'v', 'color': dh.colors['b4']},
                 'default': {'marker': 'v', 'color': 'grey'},
                 'legend': True,
                 'legend_args': {'handlelength':1, 'columnspacing': 0.8,
                                 'loc': 'upper left',
                                 'ncol': 3, 'prop': {'size': 8}}}
    default_kw = {'markevery': 20,
                  'markersize': 5,
                  'linewidth': 2}
    
    legend_elements = []
    for j in range(3):
        for i in range(3):   
            kw = tensor_kw.get((i,j), tensor_kw['default'])
            kw.update({'linestyle': 'None', 'markersize': '7'})
            legend_elements.append(Line2D([0], [0], label=f'$P_{{{i+1}{j+1}}}$', **kw))
    
    plt.legend(handles=legend_elements, **tensor_kw['legend_args'])
    plt.ylabel(r'$P_{ij}$')
    plt.xlabel('load step')
    plt.title(f'8 Observer, {lc}')
    plt.show()