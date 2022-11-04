# -*- coding: utf-8 -*-
"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks
2.2 Trainable custom layer

==================

Authors: Fabian Roth
         
08/2022
"""

# %%   
"""
Import modules

"""
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
now = datetime.datetime.now

# %% Own modules
import data as ld
import models as lm



# %%   
"""
Load model

"""

model = lm.compile_NN(in_shape=2, ns=[16,16], activation='softplus', convex=True)


# %%   
"""
Load data

"""

p, f, X, Y, Z = ld.data_2D(lm._f_2())

# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit(p, f, epochs = 1000,  verbose = 2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# plot some results
plt.figure(1, dpi=300)
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()


# %%   
"""
Evaluation

"""

mZ = model.predict(p)
mZ = tf.reshape(mZ, Z.shape)

fig = plt.figure(figsize=(5,5), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='black', alpha=0, label='Training Data')
ax.plot_surface(X, Y, mZ, rstride=1, cstride=1, cmap='inferno', edgecolor='none', alpha=0.8, label='Model Prediction')
ax.set(xlabel='x', ylabel='y', zlabel='Output f')
ax.set_title('Data and Model Prediction')

