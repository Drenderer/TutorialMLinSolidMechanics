"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Fabian Roth
         
01/2023
"""


# %% Import modules
import tensorflow as tf
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now

import data as ld
import original_files.plots as lp
import models_core as mc


# %%  Generate and visualize data 

E_infty = 0.5
E = 2
eta = 1

n = 100
omegas = [1,1,2]
As = [1,2,3]

t_indx = [0]
v_indx = list(set(range(len(As))) - set(t_indx))

eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)

t_eps       = tf.gather(eps, t_indx, axis=0)
t_eps_dot   = tf.gather(eps_dot, t_indx, axis=0)
t_sig       = tf.gather(sig, t_indx, axis=0)
t_dts       = tf.gather(dts, t_indx, axis=0)
t_omegas    = [omegas[i] for i in t_indx]
t_As        = [As[i] for i in t_indx]

v_eps       = tf.gather(eps, v_indx, axis=0)
v_eps_dot   = tf.gather(eps_dot, v_indx, axis=0)
v_sig       = tf.gather(sig, v_indx, axis=0)
v_dts       = tf.gather(dts, v_indx, axis=0)
v_omegas    = [omegas[i] for i in v_indx]
v_As        = [As[i] for i in v_indx]

lp.plot_data(t_eps, t_eps_dot, t_sig, t_omegas, t_As)

# %% Load model

tf.keras.backend.clear_session()
model = mc.compile_RNN(model_type='ffnn_maxwell') # naive_model analytic_maxwell ffnn_maxwell ffnn_maxwell_extra gsm_model

# %% Train model

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.003)
h = model.fit([t_eps, t_dts], [t_sig],
              validation_data=([v_eps, v_dts], [v_sig]),
              epochs = 4000,  verbose = 2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# %% Plot loss

plt.figure(1, dpi=600)
plt.semilogy(h.history['loss'], label='training loss')
plt.semilogy(h.history['val_loss'], label='training loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.legend()

# %% Evalueate model

sig_m = model([eps, dts])
lp.plot_data(eps, eps_dot, sig, omegas, As)
lp.plot_model_pred(eps, sig, sig_m, omegas, As)


eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = model([eps, dts])
lp.plot_data(eps, eps_dot, sig, omegas, As)
lp.plot_model_pred(eps, sig, sig_m, omegas, As)
