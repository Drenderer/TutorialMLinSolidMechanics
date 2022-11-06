# -*- coding: utf-8 -*-
"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks
1 Bathtub function
1.1 Hyperparameter sweep - Learning rate
==================

Authors: Dominik K. Klein, Fabian Roth
         
08/2022
"""

# %% Import Modules
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
now = datetime.datetime.now

# Own modules
import data as ld
import models as lm


# %% Load data
xs, ys, xs_c, ys_c = ld.bathtub()


# %% Model calibration

rates = np.linspace(0, -4, 5)
rates = np.power(10, rates)
losses = []
for lr in rates:
    
    model = lm.compile_NN(ns=[8,8], activation='softplus', convex=True)
    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
    h = model.fit(xs_c, ys_c, epochs = 1000,  verbose = 2)
    losses.append(h.history['loss'])
    
# %%

plt.figure(1, dpi=300)#, figsize=(5,4))
for l, r in zip(losses, rates):
    plt.semilogy(l, label='learning rate {:.0e}'.format(r))
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.gca().set_ylim([None, 10])
plt.legend()

# %% Model calibration
# model = lm.compile_NN(ns=[16,16], activation='softplus', convex=False)
# total_epochs = 2000
# epochs = 500
# rates = np.linspace(-1, -4, total_epochs//epochs)
# rates = np.power(10, rates)
# plt.figure(1, dpi=300)#, figsize=(5,4))
# for i, lr in enumerate(rates):
    
#     tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
#     h = model.fit(xs_c, ys_c, epochs = epochs,  verbose = 2)

#     plt.semilogy(np.linspace(i*epochs, (i+1)*epochs, epochs), h.history['loss'], label='learning rate {:.2e}'.format(lr))
    
# plt.grid(which='both')
# plt.xlabel('calibration epoch')
# plt.ylabel('log$_{10}$ MSE')
# plt.legend()
