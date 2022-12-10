# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 3.2 Model calibration
"""

# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
now = datetime.now

import data_handler as dh
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
#train_on_lc = ['biaxial', 'pure_shear', 'uniaxial']
train_on_lc = ['biaxial', 'pure_shear']     # mixed_test only is interesting!
train = dh.load_case_data(train_on_lc, concat=True, plot=True)     # Data dict

test = dh.load_case_data('all')         # List of Loadcase data dicts


# %% Train multiple Models to average the results
loss_weights = [1, 1]
num_models = 1
epochs = 10000
learning_rate = 0.001
weighted_load_cases = True

weights = train['weight'] if weighted_load_cases else None
results = []
for n_model in range(num_models):
    
    print('Training Model number {:d}'.format(n_model))
    
    # Model calibration
    model = mc.compile_physics_augmented_NN(loss_weights=loss_weights,
                                            ns=[16, 16])
    
    info_string = get_info_string(loss_weights)
    
    t1 = now()    
    tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    h = model.fit(train['F'], [train['P'], train['W']], 
                  sample_weight=weights,
                  epochs = epochs,  verbose = 2)
    t2 = now()
    print('it took', t2 - t1, '(sec) to calibrate the model')
    
    # Evaluate Model
    load_case_losses = {}
    for t in test:
        l = model.evaluate(t['F'], [t['P'], t['W']])
        load_case_losses[t['load_case']] = l
   
    
    # Write results
    model_results = {'model_number': n_model,
                     'model': model,
                     'info_string': info_string,
                     'loss': h.history['loss'],
                     'epochs': epochs,
                     'learning_rate': learning_rate,
                     'load_case_losses': load_case_losses}
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

x = np.array([0, 1, 2, 3.5, 4.5])
fig, ax = plt.subplots(dpi=600, figsize=(4,4))
ax.bar(x-0.2, avg_losses[0], yerr=std_losses[0], label=r'$P$ loss',
       align='center', ecolor='black', capsize=6, width=0.4)
ax.bar(x+0.2, avg_losses[1], yerr=std_losses[1], label=r'$W$ loss',
       align='center', ecolor='black', capsize=6, width=0.4)
ax.set_xticks(x, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title(f'''Average loss per load case, naive model\n 
                 num_models: {num_models}, learning_rate: {learning_rate},\n
                 epochs: {epochs}, weighted_load_cases: {weighted_load_cases},\n
                 {get_info_string(loss_weights)}''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()

# %% Examine the stress / energy prediction of the model in the reference configuration F = I.

tF = tf.constant([np.eye(3)])
tP, tW = model(tF)
print(f'For F = I the model predicts: \nW = {tW}, \nP = {tP}')

# %% Plot an example loadcase

#lc = 'mixed_test'
for lc in dh.files.keys():
    lc_test = dh.read_file(dh.files[lc])
    lc_model = results[0]['model']
    lc_test['*P'], lc_test['*W'] = lc_model(lc_test['F'])
    lc_test['*P'] = np.array(lc_test['*P'])
    lc_test['*W'] = np.array(lc_test['*W'])
    del lc_test['F'], lc_test['weight']
    dh.plot_data(lc_test)