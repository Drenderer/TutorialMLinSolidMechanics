# -*- coding: utf-8 -*-
"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks
2.3 Sobolev training

==================

Authors: Fabian Roth
         
08/2022
"""

# %% Import modules
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
now = datetime.datetime.now

# Own modules
import data as ld
import models as lm

# %% Load data
p, f, g, X, Y, Z = ld.gradient_2D(lm._Df_2(), plot=False)


# %% Useful function
def get_info_string(loss_weights):
    if loss_weights == [1,1]:
        return 'Trained on both output and gradient'
    elif loss_weights == [1,0]:
        return 'Trained on output only'
    elif loss_weights == [0,1]:
        return 'Trained on gradients only'
    else:
        return 'Trained on both output and gradient, but differently weighted'

# %% Model calibration
loss_weights = [[1,0], [1,1], [0,1]] # [[1,0], [1,1], [0,1]]
results = [] 
for lw in loss_weights:
    
    tf.keras.backend.clear_session()
    
    model = lm.compile_DNN(in_shape=2, loss_weights=lw, ns=[16,16], activation='softplus', convex=True)
    info_string = get_info_string(lw)
    
    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.001)
    h = model.fit(p, [f, g], epochs = 1500,  verbose = 2)
    
    results.append({'model': model,
                    'loss': h.history['loss'],
                    'info': info_string})

# plot some results
plt.figure(1, dpi=300)
for r in results:
    plt.semilogy(r['loss'], label=r['info'])
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()
plt.title('Losses')
plt.show()

# %% Evaluation

for r in results:
    
    model = r['model']
    mZ = model.predict(p)[0]
    mZ = tf.reshape(mZ, Z.shape)
    
    fig = plt.figure(figsize=(5,5), dpi=500)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='black', alpha=0, label='Training Data')
    ax.plot_surface(X, Y, mZ, rstride=1, cstride=1, cmap='inferno', edgecolor='none', alpha=0.8, label='Model Prediction')
    ax.set(xlabel='x', ylabel='y', zlabel='Output f')
    #ax.set_title(f'Data and Model Prediction, {r['info']}')
    ax.set_title('Loss {:.3e}'.format(r["loss"][-1]))
    
    diff_Z = Z-mZ
    zeros_Z = tf.zeros(shape=diff_Z.shape)
    
    fig = plt.figure(figsize=(5,5), dpi=500)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, zeros_Z, rstride=1, cstride=1, edgecolor='black', alpha=0, label='Training Data')
    ax.plot_surface(X, Y, diff_Z, rstride=1, cstride=1, cmap='inferno', edgecolor='none', alpha=0.8, label='Model Prediction')
    ax.set(xlabel='x', ylabel='y', zlabel='Output f')
    ax.set_title(f'Absolute Error, {info_string}')

