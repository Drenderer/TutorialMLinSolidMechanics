# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:48:05 2022

Task 3: 2 Deformation-gradient based neural network model
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
validation = dh.load_case_data('test', concat=True)
aug_validation = {'original': validation,
                  'material symmetry augmented': dh.augment_data(validation, symmetry_group=dh.cubic_group),
                  'observer augmented': dh.augment_data(validation, objectivity_group=100),
                  'fully augmeted': dh.augment_data(validation, symmetry_group=dh.cubic_group, objectivity_group=100)}

# %% Train multiple Models to average the results
model_args = {'ns': [16, 16]}
loss_weights = [1, 1]
num_models = 3
epochs = 6000
learning_rate = 0.005
weighted_load_cases = True

weights = train['weight'] if weighted_load_cases else None
results = []
for n_model in range(num_models):
    
    print('Training Model number {:d}'.format(n_model))
    
    # Model calibration
    model = mc.compile_deformation_gradient_based_model(loss_weights=loss_weights,
                                                        **model_args)
    
    info_string = get_info_string(loss_weights)
    
    t1 = now()    
    tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    h = model.fit(train['F'], [train['normalized P'], train['normalized W']], 
                  validation_data=[validation['F'], [validation['normalized P'], validation['normalized W']]],
                  sample_weight=weights,
                  epochs = epochs,  verbose = 2)
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
        
    aug_losses = {}
    for test_case, t in aug_validation.items():
        l = model.evaluate(t['F'], [t['normalized P'], t['normalized W']])
        aug_losses[test_case] = l
   
    
    # Write results
    model_results = {'model_number': n_model,
                     'model': model,
                     'info_string': info_string,
                     'loss': h.history['loss'],
                     'val_loss': h.history['val_loss'],
                     'epochs': epochs,
                     'learning_rate': learning_rate,
                     'load_case_losses': load_case_losses,
                     'aug_load_case_losses': aug_load_case_losses,
                     'aug_losses': aug_losses}
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
ax.set_xticks(x, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('''Average loss per load case''')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(loc='upper left')
plt.show()


# %% Plot losses for all combinations of augmented test data

aug_cases = results[0]['aug_losses'].keys()

tc_to_label = {'original': r'$\mathbf{Q}=\mathbb{1}$',
               'material symmetry augmented': r'$\mathbf{Q}_{mat}$',
               'observer augmented': r'$\mathbf{Q}_{obj}$',
               'fully augmeted': r'$\mathbf{Q}_{mat}, \mathbf{Q}_{obj}$'}

avg = []
std = []
for aug_case in aug_cases:
    loss_aggregate =  []
    for r in results:
        loss_aggregate.append(r['aug_losses'][aug_case][0])
        #print(f'Aug case: {aug_case}, Model: {r["model_number"]}', r['aug_losses'][aug_case][0])
    avg_loss = np.mean(loss_aggregate)
    std_loss = np.std(loss_aggregate)
    avg.append(avg_loss)
    std.append(std_loss)


x = np.arange(len(aug_cases))
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.bar(x[0], avg[0], yerr=std[0],
       align='center', ecolor=dh.colors['b4'], capsize=10, color=dh.colors['b2'])
ax.bar(x[1:], avg[1:], yerr=std[1:],
       align='center', ecolor=dh.colors['o3'], capsize=10, color=dh.colors['o5'])

ax.set_ylim([1,10e3])

ax.set_xticks(x, [tc_to_label[tc] for tc in aug_cases], zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('''Average loss per test data''')
plt.xticks(rotation='horizontal')
plt.yscale('log')
plt.legend(['Test Loss', 'Augmented Test Loss'], loc='upper left')
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
       align='center', ecolor=dh.colors['b4'], capsize=6, width=0.4)
ax.bar(x+0.2, avg_losses[1], yerr=std_losses[1], label=r'Loss on augmented data', color=dh.colors['o5'],
       align='center', ecolor=dh.colors['o3'], capsize=6, width=0.4)

ax.set_ylim([10e-4,2*10e3])

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

for lc in dh.files.keys():
    lc_test = dh.read_file(dh.files[lc])
    lc_model = results[0]['model']
    lc_test['*normalized P'], lc_test['*normalized W'] = lc_model(lc_test['F'])
    lc_test['*normalized P'] = np.array(lc_test['*normalized P'])
    lc_test['*normalized W'] = np.array(lc_test['*normalized W']).squeeze()
    del lc_test['F'], lc_test['weight'], lc_test['W'], lc_test['P']
    dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5))
    
# %% Plot an augmented example load Case material symmetry augmented

Q = dh.cubic_group[3]
lc = 'test2'
lc_model = results[1]['model']

lc_test = dh.read_file(dh.files[lc])
# augment data
aug_lc = dh.augment_data(lc_test, symmetry_group=[Q])
aug_lc['*normalized P'], aug_lc['*normalized W'] = lc_model(aug_lc['F'])
aug_lc['*normalized P'] = np.array(aug_lc['*normalized P'])
aug_lc['*normalized W'] = np.array(aug_lc['*normalized W']).squeeze()
# unaugment data
lc_test = dh.augment_data(aug_lc, symmetry_group=[Q.T])

dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5), 
             dont_plot=['F', 'weight', 'W', 'P'],
             title = f'Load Case: {lc}, augmented with ' + r'$Q_{mat}$')

# Compare to unaugmented case
lc_test['*normalized P'], lc_test['*normalized W'] = lc_model(lc_test['F'])
lc_test['*normalized P'] = np.array(lc_test['*normalized P'])
lc_test['*normalized W'] = np.array(lc_test['*normalized W']).squeeze()
dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5), 
             dont_plot=['F', 'weight', 'W', 'P'],
             title = f'Original Load Case: {lc}')

# %% Plot an augmented example load Case objectivity augmented

Q = dh.Rotation.random(1).as_matrix().squeeze()
lc = 'test2'
lc_model = results[1]['model']

lc_test = dh.read_file(dh.files[lc])
# augment data
aug_lc = dh.augment_data(lc_test, objectivity_group=[Q])
aug_lc['*normalized P'], aug_lc['*normalized W'] = lc_model(aug_lc['F'])
aug_lc['*normalized P'] = np.array(aug_lc['*normalized P'])
aug_lc['*normalized W'] = np.array(aug_lc['*normalized W']).squeeze()
# unaugment data
lc_test = dh.augment_data(aug_lc, objectivity_group=[Q.T])

dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5), 
             dont_plot=['F', 'weight', 'W', 'P'],
             title = f'Load Case: {lc}, augmented with ' + r'$Q_{obj}$')

# Compare to unaugmented case
lc_test['*normalized P'], lc_test['*normalized W'] = lc_model(lc_test['F'])
lc_test['*normalized P'] = np.array(lc_test['*normalized P'])
lc_test['*normalized W'] = np.array(lc_test['*normalized W']).squeeze()
dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5), 
             dont_plot=['F', 'weight', 'W', 'P'],
             title = f'Original Load Case: {lc}')