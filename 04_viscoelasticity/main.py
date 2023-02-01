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
import numpy as np

import data as ld
import plot as lp
import models_core as mc
from plot_f_tilde import plot_f_tilde


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

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.001)
h = model.fit([t_eps, t_dts], [t_sig],
              validation_data=([v_eps, v_dts], [v_sig]),
              epochs = 10,  verbose = 2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

# %% Plot loss

plt.figure(1, dpi=600)
plt.semilogy(h.history['loss'], label='training loss')
plt.semilogy(h.history['val_loss'], label='validation loss')
plt.grid(which='both')
plt.xlabel('calibration epoch')
plt.ylabel('log$_{10}$ MSE')
plt.title('Training Loss')
plt.legend()

# %% Evalueate model on harmonic data

sig_m = model([eps, dts])
lp.plot_data(eps, eps_dot, sig, omegas, As)
lp.plot_model_pred(eps, sig, sig_m, omegas, As, focus_on=t_indx)
lp.plot_model_pred(eps, sig, sig_m, omegas, As, focus_on=v_indx)
lp.plot_model_pred(eps, sig, sig_m, omegas, As, training_idxs=t_indx)

# %% Evalueate model on relaxation data

r_eps, r_eps_dot, r_sig, r_dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
r_sig_m = model([r_eps, r_dts])
lp.plot_data(r_eps, r_eps_dot, r_sig, omegas, As)
lp.plot_model_pred(r_eps, r_sig, r_sig_m, omegas, As, focus_on=t_indx)
lp.plot_model_pred(r_eps, r_sig, r_sig_m, omegas, As, focus_on=v_indx)
lp.plot_model_pred(r_eps, r_sig, r_sig_m, omegas, As, training_idxs=t_indx)

# %% Calculate min and max epsilon and gamma 

if model.name == 'ffnn_maxwell':
    gamma = eps - (1/E) * (sig - E_infty*eps)
    
    plt.figure(dpi=400)
    for i in range(eps.shape[0]):
        
        eps_min = np.min(eps[i,:,:])
        eps_max = np.max(eps[i,:,:])
        gamma_min = np.min(gamma[i,:,:])
        gamma_max = np.max(gamma[i,:,:])
        
        print(eps_min, eps_max, gamma_min, gamma_max)
        plt.plot(eps[i,:,0], gamma[i,:,0])
        
    plt.show()
    
    plot_f_tilde(model, eps, gamma)
