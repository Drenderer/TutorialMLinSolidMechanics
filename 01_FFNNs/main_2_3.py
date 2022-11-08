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
from numpy.linalg import norm
import numpy as np
import datetime
now = datetime.datetime.now

# Own modules
import data as ld
import models as lm

# %% Load data
p, f, g, X, Y, Z = ld.gradient_2D(lm._Df_2(), plot=False)

f_norm = norm(f)
g_norm = norm(g)
both_norm = norm([f_norm, g_norm])
loss_adjsutments = [f_norm, both_norm, g_norm]

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
    h = model.fit(p, [f, g], epochs = 10000,  verbose = 2)
    
    results.append({'model': model,
                    'loss': h.history['loss'],
                    'info': info_string})

# %% Plot training losses
plt.figure(1, dpi=300)
for r in results:
    plt.semilogy(r['loss'], label=r['info'])
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()
plt.title('Losses')
plt.show()

# %% Plot adjusted losses
labels = [r'only $f_2$', r'$f_2$ and $\nabla f_2$', r'only $\nabla f_2$']
plt.figure(1, dpi=600, figsize=(6,4))
for i, r in enumerate(results):
    plt.semilogy(r['loss'] / loss_adjsutments[i], label=labels[i])
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ adjusted MSE')
plt.legend()
#plt.title('Trainig Losses')
plt.show()

# %% Plot evaluation losses
# collect data
data = []
for r in results:
    data.append(r['model'].evaluate(p, [f, g], batch_size=32)[1:])
    
data = np.array(data).T

labels = [r'only $f_2$', r'$f_2$ and $\nabla f_2$', r'only $\nabla f_2$']
width = 0.35  # the width of the bars
x = np.arange(len(results))

fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.bar(x - width/2, data[0], width, color='#F68310', label='Output loss', zorder=3)
ax.bar(x + width/2, data[1], width, color='#791C6C', label='Gradient loss', zorder=3)
ax.grid(zorder=0)
ax.set_xticks(x, labels)
#plt.grid(which='both')
plt.legend()
plt.yscale('log')
#plt.title('Evaluation Losses')
plt.show()


# %% Evaluation

# for r in results:
    
#     model = r['model']
#     mZ = model.predict(p)[0]
#     mZ = tf.reshape(mZ, Z.shape)
    
#     fig = plt.figure(figsize=(5,5), dpi=500)
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='black', alpha=0, label='Training Data')
#     ax.plot_surface(X, Y, mZ, rstride=1, cstride=1, cmap='inferno', edgecolor='none', alpha=0.8, label='Model Prediction')
#     ax.set(xlabel='x', ylabel='y', zlabel='Output f')
#     #ax.set_title(f'Data and Model Prediction, {r['info']}')
#     ax.set_title('Loss {:.3e}'.format(r["loss"][-1]))
    
#     diff_Z = Z-mZ
#     zeros_Z = tf.zeros(shape=diff_Z.shape)
    
#     fig = plt.figure(figsize=(5,5), dpi=500)
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.plot_surface(X, Y, zeros_Z, rstride=1, cstride=1, edgecolor='black', alpha=0, label='Training Data')
#     ax.plot_surface(X, Y, diff_Z, rstride=1, cstride=1, cmap='inferno', edgecolor='none', alpha=0.8, label='Model Prediction')
#     ax.set(xlabel='x', ylabel='y', zlabel='Output f')
#     ax.set_title(f'Absolute Error, {info_string}')

