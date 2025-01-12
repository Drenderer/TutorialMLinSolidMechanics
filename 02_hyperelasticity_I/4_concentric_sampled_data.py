# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 4 Concentric sampled deformation gradients
"""


# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime 
now = datetime.now
import random

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


# %% Train multiple Models to average the results
ICNN_params = {'loss_weights':        [1, 0],
               'epochs':              300,
               'learning_rate':       0.002,
               'weighted_load_cases': True,
               'model_size':          [8, 8]}

FFNN_params = {'epochs':              6000,
               'learning_rate':       0.002,
               'weighted_load_cases': True,
               'model_size':          [32,32,16]}

t1 = now()

num_models = 5
ICNN_results = []
FFNN_results = []
for n_model in range(num_models):
    
    # Import data
    train, test, train_list, test_list = dh.load_concentric(num_train_lp=80, plot=False, normalize_weights=True)
    ICNN_weight = train['weight'] if ICNN_params['weighted_load_cases'] else None
    FFNN_weight = train['weight'] if FFNN_params['weighted_load_cases'] else None
    
    # Training ICNN Model
    ICNN_model = mc.compile_physics_augmented_NN(loss_weights = ICNN_params['loss_weights'],
                                                 ns = ICNN_params['model_size'])
    ICNN_info_string = get_info_string(ICNN_params['loss_weights'])
    tf.keras.backend.set_value(ICNN_model.optimizer.learning_rate, ICNN_params['learning_rate'])
    ICNN_h = ICNN_model.fit(train['F'], [train['P'], train['W']], 
                            validation_data=[test['F'], [test['P'], test['W']]],
                            sample_weight = ICNN_weight,
                            epochs = ICNN_params['epochs'],  
                            verbose = 2)
    
    FFNN_model = mc.compile_FFNN(ns = FFNN_params['model_size'])   
    tf.keras.backend.set_value(FFNN_model.optimizer.learning_rate, FFNN_params['learning_rate'])
    FFNN_h = FFNN_model.fit(train['F'], train['P'], 
                            validation_data=[test['F'], test['P']],
                            sample_weight = FFNN_weight,
                            epochs = FFNN_params['epochs'],  
                            verbose = 2,
                            batch_size=512)

    # Evaluate Models
    ICNN_test_loss = ICNN_model.evaluate(test['F'], [test['P'], test['W']])
    FFNN_test_loss = FFNN_model.evaluate(test['F'], test['P'])
    
    ICNN_train_loss = ICNN_model.evaluate(train['F'], [train['P'], train['W']])
    FFNN_train_loss = FFNN_model.evaluate(train['F'], train['P'])

    # Write results
    ICNN_model_results = {'model_number': n_model,
                          'model': ICNN_model,
                          'info_string': ICNN_info_string,
                          'loss': ICNN_h.history['loss'],
                          'val_loss': ICNN_h.history['val_loss'],
                          'epochs': ICNN_params['epochs'],
                          'learning_rate': ICNN_params['learning_rate'],
                          'test_loss': ICNN_test_loss,
                          'train_loss': ICNN_train_loss}
    ICNN_results.append(ICNN_model_results)
    
    FFNN_model_results = {'model_number': n_model,
                          'model': FFNN_model,
                          'loss': FFNN_h.history['loss'],
                          'val_loss': FFNN_h.history['val_loss'],
                          'epochs': FFNN_params['epochs'],
                          'learning_rate': FFNN_params['learning_rate'],
                          'test_loss': FFNN_test_loss,
                          'train_loss': FFNN_train_loss}
    FFNN_results.append(FFNN_model_results)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate and evaluate the models')

# %% Plot trainig losses
plt.figure(1, dpi=600, figsize=(6,4))
for n, r in enumerate(ICNN_results):
    b1 = plt.semilogy(r['loss'], color=dh.colors['o3'], alpha=0.8, label='ICNN Training loss')
    b2 = plt.semilogy(r['val_loss'], color=dh.colors['o5'], alpha=0.8, label='ICNN Test loss')
for n, r in enumerate(FFNN_results):
    b3 = plt.semilogy(r['loss'], color=dh.colors['b2'], alpha=0.8, label='FFNN Training loss')
    b4 = plt.semilogy(r['val_loss'], color=dh.colors['b4'], alpha=0.8, label='FFNN Test loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
handles = [Line2D([0], [0], color=dh.colors['o3'], alpha=0.8, label='ICNN Training loss'),
           Line2D([0], [0], color=dh.colors['o5'], alpha=0.8, label='ICNN Test loss'),
           Line2D([0], [0], color=dh.colors['b2'], alpha=0.8, label='FFNN Training loss'),
           Line2D([0], [0], color=dh.colors['b4'], alpha=0.8, label='FFNN Test loss')]
plt.legend(handles, ['ICNN Training loss', 'ICNN Test loss', 'FFNN Training loss', 'FFNN Test loss'])

# %% Compare Training and test losses
# This is one of the ugliest code a human has written, sorry
train_loss_aggregate = []
test_loss_aggregate = []
for r in ICNN_results:
    train_loss_aggregate.append(r['train_loss'][0])
    test_loss_aggregate.append(r['test_loss'][0])
ICNN_train_loss_avg = np.mean(train_loss_aggregate)
ICNN_train_loss_std = np.std(train_loss_aggregate)
ICNN_test_loss_avg = np.mean(test_loss_aggregate)
ICNN_test_loss_std = np.std(test_loss_aggregate)

ICNN_loss_avg = [ICNN_train_loss_avg, ICNN_test_loss_avg]
ICNN_loss_std = [ICNN_train_loss_std, ICNN_test_loss_std]

train_loss_aggregate = []
test_loss_aggregate = []
for r in FFNN_results:
    train_loss_aggregate.append(r['train_loss'])
    test_loss_aggregate.append(r['test_loss'])
FFNN_train_loss_avg = np.mean(train_loss_aggregate)
FFNN_train_loss_std = np.std(train_loss_aggregate)
FFNN_test_loss_avg = np.mean(test_loss_aggregate)
FFNN_test_loss_std = np.std(test_loss_aggregate)
    
FFNN_loss_avg = [FFNN_train_loss_avg, FFNN_test_loss_avg]
FFNN_loss_std = [FFNN_train_loss_std, FFNN_test_loss_std]



x = np.arange(2)
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
b1 = ax.bar(x-0.2, ICNN_loss_avg, yerr=ICNN_loss_std, label='ICNN loss',
       align='center', ecolor=dh.colors['o3'], capsize=6, width=0.4, color=dh.colors['o4'])
b2 = ax.bar(x+0.2, FFNN_loss_avg, yerr=FFNN_loss_std, label='FFNN loss',
       align='center', ecolor=dh.colors['b3'], capsize=6, width=0.4, color=dh.colors['b2'])
ax.set_xticks(x, ['Training', 'Test'], zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
plt.xticks(rotation='horizontal')
plt.yscale('log')
plt.legend(loc='upper left')

# b1[1].set_color(dh.colors['o4'])
# b2[1].set_color(dh.colors['b3'])

plt.show()

# %% Plot an example loadcase

lc_test = random.choice(test_list)

lc_icnn_model = ICNN_results[0]['model']
lc_ffnn_model = FFNN_results[0]['model']

dh.plot_data(lc_test)

ICNN_pred = {'P': lc_test['P']}
ICNN_pred['*P'], _ = lc_icnn_model(lc_test['F'])
ICNN_pred['*P'] = np.array(ICNN_pred['*P'])
dh.plot_data(ICNN_pred, title=f'ICNN prediction for test load case {lc_test["load_case"]}',
             tensor_kw={'legend': True}, dpi=600, figsize=(6,5))

FFNN_pred = {'P': lc_test['P']}
FFNN_pred['*P'] = lc_ffnn_model(lc_test['F'])
FFNN_pred['*P'] = np.array(FFNN_pred['*P'])
dh.plot_data(FFNN_pred, title=f'FFNN prediction for test load case {lc_test["load_case"]}',
             tensor_kw={'legend': True}, dpi=600, figsize=(6,5))


lc_train = random.choice(train_list)

dh.plot_data(lc_train)

lc_model = ICNN_results[0]['model']
ICNN_pred = {'P': lc_train['P']}
ICNN_pred['*P'], _ = lc_icnn_model(lc_train['F'])
ICNN_pred['*P'] = np.array(ICNN_pred['*P'])
dh.plot_data(ICNN_pred, title=f'ICNN prediction for train load case {lc_train["load_case"]}',
             tensor_kw={'legend': True}, dpi=600, figsize=(6,5))

lc_model = FFNN_results[0]['model']
FFNN_pred = {'P': lc_train['P']}
FFNN_pred['*P'] = lc_ffnn_model(lc_train['F'])
FFNN_pred['*P'] = np.array(FFNN_pred['*P'])
dh.plot_data(FFNN_pred, title=f'FFNN prediction for train load case {lc_train["load_case"]}',
             tensor_kw={'legend': True}, dpi=600, figsize=(6,5))