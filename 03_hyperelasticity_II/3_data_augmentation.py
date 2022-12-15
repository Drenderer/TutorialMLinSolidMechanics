# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:48:05 2022

Task 3: 3 Data augmentation
"""

# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
now = datetime.now

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
                            objectivity_group=100)
test = dh.load_case_data('all')                                     # List of Loadcase data dicts
aug_test = [dh.augment_data(t,
                            symmetry_group=dh.cubic_group,
                            objectivity_group=100)
            for t in test]                                          # List of augmented data dicts for every load case


# %% Train multiple Models to average the results
model_args = {'ns': [32, 32]}
loss_weights = [1, 1]
num_models = 1
precal_epochs = 500    # epochs used to train on unaugmented data before training on augmented data
precal_learning_rate = 0.03
epochs = 50
learning_rate = 0.03
weighted_load_cases = True


results = []
for n_model in range(num_models):
    
    print('Training Model number {:d}'.format(n_model))
    
    # Model calibration
    model = mc.compile_deformation_gradient_based_model(loss_weights=loss_weights,
                                                        **model_args)
    
    info_string = get_info_string(loss_weights)
    
    t1 = now()    
    # Pre train
    tf.keras.backend.set_value(model.optimizer.learning_rate, precal_learning_rate)
    weights = pre_train['weight'] if weighted_load_cases else None
    precal_h = model.fit(pre_train['F'], [pre_train['normalized P'], pre_train['normalized W']], 
                          sample_weight=weights,
                          epochs = precal_epochs,  verbose = 1)
    
    tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    weights = aug_train['weight'] if weighted_load_cases else None
    h = model.fit(aug_train['F'], [aug_train['normalized P'], aug_train['normalized W']], 
                  sample_weight=weights,
                  epochs = epochs,  verbose = 1)
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
    model_results = {'model_number': n_model,
                     'model': model,
                     'info_string': info_string,
                     'loss': precal_h.history['loss'] + h.history['loss'],
                     'epochs': epochs,
                     'learning_rate': learning_rate,
                     'load_case_losses': load_case_losses,
                     'aug_load_case_losses': aug_load_case_losses}
    results.append(model_results)
    

# %% Plot trainig losses
plt.figure(1, dpi=600)#, figsize=(5,4))
for r in results:
    plt.semilogy(r['loss'], label='training loss model {:d}'.format(r['model_number']))
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()

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
fig, ax = plt.subplots(dpi=600, figsize=(4,4))
ax.bar(x-0.2, avg_losses[0], yerr=std_losses[0], label=r'$P$ loss',
       align='center', ecolor='black', capsize=6, width=0.4)
ax.bar(x+0.2, avg_losses[1], yerr=std_losses[1], label=r'$W$ loss',
       align='center', ecolor='black', capsize=6, width=0.4)
ax.set_xticks(x, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('''Average loss per load case, not augmented data''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()

# %% Plot losses for P and W per augmented load case 

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
ax.bar(x-0.2, avg_losses[0], yerr=std_losses[0], label=r'Loss on original data',
       align='center', ecolor='black', capsize=6, width=0.4)
ax.bar(x+0.2, avg_losses[1], yerr=std_losses[1], label=r'Loss on augmented data',
       align='center', ecolor='black', capsize=6, width=0.4)
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
    lc_test['*normalized W'] = np.array(lc_test['*normalized W'])
    del lc_test['F'], lc_test['weight'], lc_test['W'], lc_test['P']
    dh.plot_data(lc_test)