# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 2.2 Model calibration
Task 2.3 Loss weighting strategy (Set weighted_load_cases=True)
"""

# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
now = datetime.now

import data_handler_2 as dh
import models_core as mc


# %% Import data

train_on_lc = ['biaxial', 'pure_shear', 'uniaxial']
train = dh.load_case_data(train_on_lc, concat=True, normalize_weights=True, plot=True)     # Data dict

validation = dh.load_case_data('test', concat=True, normalize_weights=True, plot=False)

test = dh.load_case_data('all')         # List of Loadcase data dicts

# %% Train multiple Models to average the results
model_args = {'ns': [16, 16]}
num_models = 1
epochs = 4000
learning_rate = 0.005
weighted_load_cases = False

weights = train['weight'] if weighted_load_cases else None
results = []
for n_model in range(num_models):
    
    print('Training Model number {:d}'.format(n_model))
    
    # Model calibration
    model = mc.compile_FFNN(**model_args)
    
    t1 = now()    
    tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    h = model.fit(train['F'], train['P'], sample_weight=weights,
                  epochs = epochs,  verbose = 2,
                  validation_data=[validation['F'], validation['P']])
    t2 = now()
    print('it took', t2 - t1, '(sec) to calibrate the model')
    
    # Evaluate Model
    load_case_losses = {}
    for t in test:
        l = model.evaluate(t['F'], t['P'])
        load_case_losses[t['load_case']] = l
   
    
    # Write results
    model_results = {'model_number': n_model,
                     'model': model,
                     'loss': h.history['loss'],
                     'val_loss': h.history['val_loss'],
                     'epochs': epochs,
                     'learning_rate': learning_rate,
                     'load_case_losses': load_case_losses}
    results.append(model_results)
    

# %% Plot trainig losses
plt.figure(1, dpi=600, figsize=(6,4))
for r in results:
    plt.semilogy(r['loss'], color=dh.colors['b2'], alpha=0.8)
    plt.semilogy(r['val_loss'], color=dh.colors['o5'], alpha=0.8)
plt.grid(which='both')
plt.xlabel('Calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend(['Training loss', 'Test loss'])

# %% Plot losses per load case
avg_losses = []
std_losses = []
load_case_names = [t['load_case'] for t in test]
for load_case in load_case_names:      # loop over test data to get load case names
    loss_aggregate = []
    for r in results:
        loss_aggregate.append(r['load_case_losses'][load_case])
        
    avg_loss = np.mean(loss_aggregate)
    std_loss = np.std(loss_aggregate)
    avg_losses.append(avg_loss)
    std_losses.append(std_loss)

x1 = [0, 1, 2]
x2 = [3.5, 4.5]
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.bar(x1, avg_losses[:3], yerr=std_losses[:3], color=dh.colors['b2'],
       align='center', ecolor=dh.colors['b4'], capsize=10,
       label='Training cases')
ax.bar(x2, avg_losses[3:], yerr=std_losses[3:], color=dh.colors['o5'],
       align='center', ecolor=dh.colors['o3'], capsize=10,
       label='Test cases')
ax.set_xticks(x1+x2, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('Average loss per load case')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend()
plt.show()


# %% Examine the stress / energy prediction of the model in the reference configuration F = I.

tF = tf.constant([np.eye(3)])
tP = model(tF)
print(f'For F = I the model predicts: \nP = \n{tP} \n=\n{np.round(tP,2)}')

# %% Plot an example loadcase

#lc = 'mixed_test'
for lc in dh.files.keys():
    lc_test = dh.read_file(dh.files[lc])
    lc_model = results[0]['model']
    lc_test['*P'] = lc_model(lc_test['F'])
    lc_test['*P'] = np.array(lc_test['*P'])
    del lc_test['F'], lc_test['weight'], lc_test['W']
    dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(6,5))