# -*- coding: utf-8 -*-
"""
Task 3.2 Model calibration
This script compares for all training strategies p, w only or both the global results
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
train_on_lc = ['biaxial', 'pure_shear', 'uniaxial']
#train_on_lc = ['biax_test', 'mixed_test']     # mixed_test only is interesting!
train = dh.load_case_data(train_on_lc, concat=True, normalize_weights=True, plot=True)     # Data dict

validation = dh.load_case_data('test', concat=True, normalize_weights=True, plot=False)

test = dh.load_case_data('all', concat=True)         # List of Loadcase data dicts


# %% Train multiple Models to average the results
model_args = {'ns': [16, 16]}
num_models = 2
epochs = 4000
learning_rate = 0.005
weighted_load_cases = True

strategy_results = []

for loss_weights in [[1,0], [0,1], [1,1]]:

    weights = train['weight'] if weighted_load_cases else None
    results = []
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
        total_loss = model.evaluate(test['F'], [test['P'], test['W']])       
        
        # Write results
        model_results = {'model_number': n_model,
                         'model': model,
                         'info_string': info_string,
                         'loss': h.history['loss'],
                         'val_loss': h.history['val_loss'],
                         'epochs': epochs,
                         'learning_rate': learning_rate,
                         'total_loss': total_loss}
        results.append(model_results)
    strategy_results.append(results)

# %% Plot comparison
avg_P_loss = []
std_P_loss = []
avg_W_loss = []
std_W_loss = []
for results in strategy_results:
    P_loss_aggregate = []
    W_loss_aggregate = []
    for r in results:
        P_loss_aggregate.append(r['total_loss'][1])
        W_loss_aggregate.append(r['total_loss'][2])
    avg_P_loss.append(np.mean(P_loss_aggregate))
    std_P_loss.append(np.std(P_loss_aggregate))
    avg_W_loss.append(np.mean(W_loss_aggregate))
    std_W_loss.append(np.std(W_loss_aggregate))
    
x = np.array([0, 1, 2])
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
b1 = ax.bar(x-0.2, avg_P_loss, yerr=std_P_loss, label=r'$P$ loss', color=dh.colors['b1'],
        align='center', ecolor=dh.colors['b4'], capsize=6, width=0.4)
b2 = ax.bar(x+0.2, avg_W_loss, yerr=std_W_loss, label=r'$W$ loss', color=dh.colors['b3'],
        align='center', ecolor=dh.colors['b1'], capsize=6, width=0.4)

ax.set_xticks(x, [r'only $P$', r'only $W$',  r'$P$ and $W$'], zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('Average loss per strategy')
plt.xticks(rotation='horizontal')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 7})

b1[0].set_color(dh.colors['o3'])
b2[0].set_color(dh.colors['o5'])

plt.show()

# %% Plot trainig losses
# plt.figure(1, dpi=600, figsize=(6,4))
# for r in results:
#     plt.semilogy(r['loss'], color=dh.colors['b2'], alpha=0.8)
#     plt.semilogy(r['val_loss'], color=dh.colors['o5'], alpha=0.8)
# plt.grid(which='both')
# plt.xlabel('calibration epoch')
# plt.ylabel('log$_{10}$ MSE')
# plt.legend(['Training loss', 'Test loss'])

# %% Plot losses for P and W per load case 

# load_case_names = [t['load_case'] for t in test]
# avg_losses = []         # contains two lists first for loss on P second for loss on W
# std_losses = []         # same as above
# for i in [1,2]:     # Loop over both loss on P second for loss on W
#     avg = []
#     std = []
#     for lc in load_case_names:
#         loss_aggregate = []
#         for r in results:
#             loss_aggregate.append(r['load_case_losses'][lc][i])
            
#         avg_loss = np.mean(loss_aggregate)
#         std_loss = np.std(loss_aggregate)
#         avg.append(avg_loss)
#         std.append(std_loss)
#     avg_losses.append(avg)
#     std_losses.append(std)

# x = np.array([0, 1, 2, 3.5, 4.5])
# x1 = x[:3]
# x2 = x[3:]
# fig, ax = plt.subplots(dpi=600, figsize=(3,4))
# ax.bar(x1-0.2, avg_losses[0][:3], yerr=std_losses[0][:3], label=r'$P$ training loss', color=dh.colors['b2'],
#        align='center', ecolor=dh.colors['b4'], capsize=6, width=0.4)
# ax.bar(x1+0.2, avg_losses[1][:3], yerr=std_losses[1][:3], label=r'$W$ training loss', color=dh.colors['b3'],
#        align='center', ecolor=dh.colors['b4'], capsize=6, width=0.4)
# ax.bar(x2-0.2, avg_losses[0][3:], yerr=std_losses[0][3:], label=r'$P$ test loss', color=dh.colors['o3'],
#        align='center', ecolor=dh.colors['o2'], capsize=6, width=0.4)
# ax.bar(x2+0.2, avg_losses[1][3:], yerr=std_losses[1][3:], label=r'$W$ test loss', color=dh.colors['o5'],
#        align='center', ecolor=dh.colors['o3'], capsize=6, width=0.4)
# ax.set_xticks(x, load_case_names, zorder=3)
# ax.grid(zorder=0)
# ax.set_axisbelow(True)
# ax.set_title('Average loss per load case')
# plt.xticks(rotation='vertical')
# plt.yscale('log')
# plt.legend(loc='upper left', prop={'size': 7})
# plt.show()

# # %% Plot losses per load case
# avg_losses = []
# std_losses = []
# load_case_names = [lc for lc in dh.files.keys()]
# for load_case in load_case_names:      # loop over test data to get load case names
#     loss_aggregate = []
#     for r in results:
#         loss_aggregate.append(r['load_case_losses'][load_case][0])
        
#     avg_loss = np.mean(loss_aggregate)
#     std_loss = np.std(loss_aggregate)
#     avg_losses.append(avg_loss)
#     std_losses.append(std_loss)

# x1 = [0, 1, 2]
# x2 = [3.5, 4.5]
# fig, ax = plt.subplots(dpi=600, figsize=(3,4))
# ax.bar(x1, avg_losses[:3], yerr=std_losses[:3], color=dh.colors['b2'],
#        align='center', ecolor=dh.colors['b4'], capsize=10,
#        label='Training cases')
# ax.bar(x2, avg_losses[3:], yerr=std_losses[3:], color=dh.colors['o5'],
#        align='center', ecolor=dh.colors['o3'], capsize=10,
#        label='Test cases')
# ax.set_xticks(x1+x2, load_case_names, zorder=3)
# ax.grid(zorder=0)
# ax.set_axisbelow(True)
# ax.set_title('Average loss per load case')
# plt.xticks(rotation='vertical')
# plt.yscale('log')
# plt.legend()
# plt.show()

# # %% Examine the stress / energy prediction of the model in the reference configuration F = I.

# tF = tf.constant([np.eye(3)])
# tP, tW = model(tF)
# print(f'For F = I the model predicts: \nW = {tW}, \nP = \n{tP} \n=\n{np.round(tP,2)}')

# %% Plot an example loadcase

#lc = 'mixed_test'
for lc in dh.files.keys():
    lc_test = dh.read_file(dh.files[lc])
    lc_model = strategy_results[2][0]['model']
    lc_test['*P'], lc_test['*W'] = lc_model(lc_test['F'])
    lc_test['*P'] = np.array(lc_test['*P'])
    lc_test['*W'] = np.array(lc_test['*W']).squeeze()
    del lc_test['weight'], lc_test['F']
    dh.plot_data(lc_test, tensor_kw={'legend': True}, dpi=600, figsize=(8,5), n_cols=3)