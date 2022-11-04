# -*- coding: utf-8 -*-
"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks
1.1 Hyperparameter sweep

==================

Authors: Fabian Roth
         
08/2022
"""


# %% Import modules

from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
now = datetime.datetime.now

# Custom modules
import data as ld
import models as lm


# %% Load data

p, f, g, X, Y, Z = ld.gradient_2D(lm._Df_2(), plot=False)

# %% Load model

loss_weights = [1,1]
model = lm.compile_DNN(in_shape=2, loss_weights=loss_weights, ns=[16,16], activation='softplus', convex=False)

info_string = ''
if loss_weights == [1,1]:
    info_string = 'Trained on both output and gradient'
elif loss_weights == [1,0]:
    info_string = 'Trained on output only'
elif loss_weights == [0,1]:
    info_string = 'Trained on gradients only'
else:
    info_string = 'Trained on both output and gradient, but differently weighted'
    
# %% Model calibration

epochs = 250

for e in range(epochs):
    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.0005)
    h = model.fit(p, [f, g], epochs = 1,  verbose = 2)
    
    mZ = model.predict(p)[0]
    mZ = tf.reshape(mZ, Z.shape)

    fig = plt.figure(figsize=(5,5), dpi=500)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='black', alpha=0, label='Training Data')
    ax.plot_surface(X, Y, mZ, rstride=1, cstride=1, cmap='inferno', edgecolor='none', alpha=0.8, label='Model Prediction')
    ax.set(xlabel='x', ylabel='y', zlabel='Output f')
    ax.set_title(f'Training Epoch {e}')
    ax.view_init(30, (2*e)%360)
    plt.show()
    
