"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks
1 Bathtub function
1.1 Hyperparameter sweep
1.2 Input convex neural networks
==================

Authors: Dominik K. Klein, Fabian Roth
         
08/2022
"""

# %% Import Modules
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
now = datetime.datetime.now

# Own modules
import data as ld
import models as lm


# %% 
tf.keras.backend.clear_session()

# %% Load model
def activation(x):
    return tf.exp(x)

model = lm.compile_NN(ns=[16,16], activation=activation, convex=False, non_neg_init=False)


# %% Load data
xs, ys, xs_c, ys_c = ld.bathtub()


# %% Model calibration
t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.01)
h = model.fit(xs_c, ys_c, epochs = 4000,  verbose = 2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# %% Print number of zero-weights
zeros = 0
for w in model.weights:
    zeros += (tf.size(w) - tf.math.count_nonzero(w, dtype=tf.int32)).numpy()
print('Model has {:d} zero-weights ({:.2f}%)'.format(zeros, 100*zeros/model.count_params()))


# %% plot some results
plt.figure(1, dpi=300)#, figsize=(5,4))
plt.semilogy(h.history['loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()


# %% Evaluation
plt.figure(2, dpi=600, figsize=(5,4))
plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
plt.plot(xs, ys, c='black', linestyle='--', label='bathtub function')
plt.plot(xs, model.predict(xs), label='model', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# %% Extra

# xe = tf.expand_dims(tf.linspace(-10, 11, 200), axis=1)

# plt.figure(2, dpi=600)
# plt.scatter(xs_c[::10], ys_c[::10], c='green', label = 'calibration data')
# plt.plot(xs, ys, c='black', linestyle='--', label='bathtub function')
# plt.plot(xe, model.predict(xe), label='model', color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

