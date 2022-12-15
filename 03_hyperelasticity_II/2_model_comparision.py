# -*- coding: utf-8 -*-
"""
Task 3: 2 Deformation-gradient based neural network model
    Comparison between Invariant based Model and Deformation gradient based model on augmented data
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
train = dh.load_case_data('train', concat=True,    #['biaxial', 'shear', 'uniaxial']
                          normalize_weights=True, plot=True)        # Data dict
test = dh.load_case_data('all')                                     # List of Loadcase data dicts
aug_test = [dh.augment_data(t,
                            symmetry_group=dh.cubic_group,
                            objectivity_group=2)
            for t in test]                                          # List of augmented data dicts for every load case
invar_test = dh.load_case_data('test', concat=True)
aug_invar_test = {'original': invar_test,
                  'material symmetry augmented': dh.augment_data(invar_test, symmetry_group=dh.cubic_group),
                  'observer augmented': dh.augment_data(invar_test, objectivity_group=2),
                  'fully augmeted': dh.augment_data(invar_test, symmetry_group=dh.cubic_group, objectivity_group=2)}

# %% Train multiple Models to average the results
dgbm_options = {'model_args': {'ns': [16, 16]},
                'loss_weights': [1, 1],
                'num_models': 1,
                'epochs': 4000,
                'learning_rate': 0.01,
                'weighted_load_cases': True}

ibm_options  = {'model_args': {'ns': [32, 32, 16]},
                'loss_weights': [1, 1],
                'num_models': 1,
                'epochs': 4000,
                'learning_rate': 0.01,
                'weighted_load_cases': True}

arc_options = {'dgbm': dgbm_options,
               'ibm':  ibm_options}



arc_results = {}
for arc_name, arc in arc_options.items():
    results = []
    
    model_args = arc['model_args']
    loss_weights = arc['loss_weights']
    info_string = get_info_string(arc['loss_weights'])
    learning_rate = arc['learning_rate']
    epochs = arc['epochs']
    weights = train['weight'] if arc['weighted_load_cases'] else None
    
    for n_model in range(arc['num_models']):
        
        print('Training Model number {:d}'.format(n_model))
        
        # Model calibration
        if arc_name == 'dgbm':
            model = mc.compile_deformation_gradient_based_model(loss_weights=loss_weights,
                                                                **model_args)
        elif arc_name == 'ibm':
            model = mc.compile_invariant_based_model(loss_weights=loss_weights,
                                                                **model_args)
        else:
            raise ValueError('Unknown model type')
            
        t1 = now()    
        tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
        h = model.fit(train['F'], [train['normalized P'], train['normalized W']], 
                      sample_weight=weights,
                      epochs = epochs,  verbose = 2)
        t2 = now()
        print('it took', t2 - t1, '(sec) to calibrate the model')
        
        # Evaluate Model
        # load_case_losses = {}
        # for t in test:
        #     l = model.evaluate(t['F'], [t['normalized P'], t['normalized W']])
        #     load_case_losses[t['load_case']] = l
            
        # aug_load_case_losses = {}
        # for t in aug_test:
        #     l = model.evaluate(t['F'], [t['normalized P'], t['normalized W']])
        #     aug_load_case_losses[t['load_case']] = l
            
        aug_losses = {}
        for test_case, t in aug_invar_test.items():
            l = model.evaluate(t['F'], [t['normalized P'], t['normalized W']])
            aug_losses[test_case] = l
       
        
        # Write results
        model_results = {'model_number': n_model,
                         'model': model,
                         'info_string': info_string,
                         'loss': h.history['loss'],
                         'epochs': epochs,
                         'learning_rate': learning_rate,
                         # 'load_case_losses': load_case_losses,
                         # 'aug_load_case_losses': aug_load_case_losses,
                         'aug_losses': aug_losses}
        results.append(model_results)
        
    arc_results[arc_name] = results
    

# %% Plot trainig losses
plt.figure(1, dpi=600)#, figsize=(5,4))
colors = {'dgbm': dh.colors['b1'],
          'ibm':  dh.colors['o1']}
for arc_name, arc_r in arc_results.items():
    color = colors[arc_name]
    for r in arc_r:
        plt.semilogy(r['loss'], label='{} training loss model {:d}'.format(arc_name, r['model_number']))
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()

# %% Plot losses for all combinations of augmented test data

test_cases = r['aug_losses'].keys()

arc_avg = {}
arc_std = {}
for arc_name, arc_r in arc_results.items():
    avg = []
    std = []
    for test_case in test_cases:
        loss_aggregate =  []
        for r in arc_r:
            loss_aggregate.append(r['aug_losses'][test_case][0])
        avg_loss = np.mean(loss_aggregate)
        std_loss = np.std(loss_aggregate)
        avg.append(avg_loss)
        std.append(std_loss)
    arc_avg[arc_name] = avg
    arc_std[arc_name] = std

x = np.arange(len(test_cases))
fig, ax = plt.subplots(dpi=600, figsize=(4,4))
ax.bar(x-0.2, arc_avg['dgbm'], yerr=arc_std['dgbm'], label='Deformation Gradient based Model',
       align='center', ecolor=dh.colors['b5'], capsize=6, width=0.4)
ax.bar(x+0.2, arc_avg['ibm'], yerr=arc_std['ibm'], label='Invariant based Model',
       align='center', ecolor=dh.colors['o1'], capsize=6, width=0.4)
ax.set_xticks(x, test_cases, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('''Average loss per test data''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()
