# -*- coding: utf-8 -*-
"""
Task 3.2 Model calibration

Copare Losses when training only on one loadcase
"""

# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
now = datetime.now

import data_handler_2 as dh
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

test = dh.load_case_data('all')         # List of Loadcase data dicts

val_loss_per_lc = {}
val_loss_std_per_lc = {}
for train_on_lc in ['biaxial', 'pure_shear', 'uniaxial', 'biax_test', 'mixed_test']:
    train = dh.load_case_data([train_on_lc], concat=True, normalize_weights=True, plot=False)     # Data dict
    val_lc = list(set(dh.files.keys()) - set([train_on_lc]))
    validation = dh.load_case_data(val_lc, concat=True, normalize_weights=True, plot=False)

    model_args = {'ns': [16, 16]}
    loss_weights = [1, 1]
    num_models = 5
    epochs = 10_000
    learning_rate = 0.003
    weighted_load_cases = True
    
    weights = train['weight'] if weighted_load_cases else None
    final_val_loss_list = []
    plt.figure(1, dpi=600, figsize=(6,4)) # for training losses
    for n_model in range(num_models):
        
        print('Training Model number {:d}'.format(n_model))
        
        # Model calibration
        model = mc.compile_physics_augmented_NN(loss_weights=loss_weights,
                                                **model_args)
        
        info_string = get_info_string(loss_weights)
        
        t1 = now()    
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
        h = model.fit(train['F'], [train['P'], train['W']], 
                      sample_weight=weights,
                      epochs = epochs,  verbose = 2,
                      validation_data=[validation['F'], [validation['P'], validation['W']]])
        t2 = now()
        print('it took', t2 - t1, '(sec) to calibrate the model')
        
        # Evaluate Model
        load_case_losses = {}
        for t in test:
            l = model.evaluate(t['F'], [t['P'], t['W']])
            load_case_losses[t['load_case']] = l
            
        final_val_loss = model.evaluate(validation['F'], [validation['P'], validation['W']])
        final_val_loss_list.append(final_val_loss)
        
        plt.semilogy(h.history['loss'], color=dh.colors['b2'], alpha=0.8)
        plt.semilogy(h.history['val_loss'], color=dh.colors['o5'], alpha=0.8)
    
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend(['Training loss', 'Test loss'])
    plt.title(f'Load Case: {train_on_lc}')
    plt.show()

    
    val_loss_per_lc[train_on_lc] = np.mean(final_val_loss_list)
    val_loss_std_per_lc[train_on_lc] = np.std(final_val_loss_list)


# %% Plot validation losses for each training load case

y_pos = np.arange(len(val_loss_per_lc))

fig, ax = plt.subplots(dpi=600, figsize=(6,4))
ax.barh(y_pos, val_loss_per_lc.values(), xerr=val_loss_std_per_lc.values(), ecolor=dh.colors['o4'], color=dh.colors['b2'], capsize=10)
ax.set_yticks(y_pos, labels=val_loss_per_lc.keys(), zorder=3)
ax.grid('both', zorder=0)
ax.set_axisbelow(True)
plt.xscale('log')
plt.xlabel('log$_{10}$ MSE')
plt.title('Test Loss per training Load Case')
plt.show()