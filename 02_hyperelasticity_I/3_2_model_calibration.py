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
train = dh.training_data(plot=True)     # Data dict
test = dh.load_case_data('all')         # List of Loadcase data dicts


# %% Train multiple Models to average the results
loss_weights = [1, 0]
num_models = 1
epochs = 4000
learning_rate = 0.005
weighted_load_cases = False

weights = train['weight'] if weighted_load_cases else None
results = []
for n_model in range(num_models):
    
    print('Training Model number {:d}'.format(n_model))
    
    # Model calibration
    model = mc.compile_physics_augmented_NN(loss_weights=loss_weights,
                                            ns=[8, 8])
    
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

# %% Plot losses per load case
avg_losses = {}
std_losses = {}
for t in test:      # loop over test data to get load case names
    load_case_name = t['load_case']
    loss_aggregate = []
    for r in results:
        loss_aggregate.append(r['load_case_losses'][load_case_name][1])
        
    avg_loss = np.mean(loss_aggregate)
    std_loss = np.std(loss_aggregate)
    avg_losses[load_case_name] = avg_loss
    std_losses[load_case_name] = std_loss

x = np.arange(len(avg_losses))
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.bar(x, avg_losses.values(), yerr=std_losses.values(), 
       align='center', ecolor='black', capsize=10)
ax.set_xticks(x, avg_losses.keys(), zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title(f'''Average loss per load case, naive model\n 
                 num_models: {num_models}, learning_rate: {learning_rate},\n
                 epochs: {epochs}, weighted_load_cases: {weighted_load_cases}''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.show()


