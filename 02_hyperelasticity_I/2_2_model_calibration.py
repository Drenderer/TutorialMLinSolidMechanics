# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:09:59 2022

Task 2.2 Model calibration
"""

# %% Import
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
now = datetime.now

import data_handler as dh
import models_core as mc


# %% Import data and model

model = mc.compile_FFNN(ns=[8, 8])

train = dh.training_data(plot=True)

# %% Model calibration
t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.003)
h = model.fit(train['F'], train['P'], epochs = 4000,  verbose = 2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# %% plot some results
plt.figure(1, dpi=300)#, figsize=(5,4))
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()

# %% Evaluate
test = dh.load_case_data(which='test')
for t in test:
    t['*P'] = np.array(model(t['F']))
    del t['W']
    dh.plot_data(t)
    
# %% Compare load cases
eval_data = dh.load_case_data('all')
losses = []
labels = []
for d in eval_data:
    l = model.evaluate(d['F'], d['P'])
    losses.append(l)
    labels.append(d['load_case'])

x = np.arange(len(losses))
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.bar(x, losses)
ax.set_xticks(x, labels, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.show()
