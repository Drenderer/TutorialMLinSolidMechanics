# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:46:20 2022

@author: fabia
"""

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

import custom_colors as cc



x = np.linspace(-4,4,200)
relu = tf.keras.activations.relu(x)
softplus = tf.keras.activations.softplus(x)
tanh = tf.keras.activations.tanh(x)
sigmoid = tf.keras.activations.sigmoid(x)

activations = [relu, softplus, tanh, sigmoid]
labels = ['relu', 'softplus', 'tanh', 'sigmoid']

for emph in range(len(activations)):
    plt.figure(dpi=600)
    plt.gca().set_prop_cycle(cc.cycler_3)
    for i, a in enumerate(activations):
        if i == emph:
            plt.plot(x, a, color='red', linewidth=3, label=labels[i])
            next(plt.gca()._get_lines.prop_cycler)
        else:
            plt.plot(x, a, alpha=0.3, label=labels[i])
    plt.grid(which='both')
    plt.legend()
    plt.show()
