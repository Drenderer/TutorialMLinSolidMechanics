"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity

Models Core
==================
Authors: Fabian Roth
         
01/2023
"""

# %% Import
import tensorflow as tf
from tensorflow.keras import layers


# %% Define useful functions and models

class FFNN(tf.keras.layers.Layer):
    """
    Basic feed forward neural network:
        - Smooth activation (default softplus) for hidden layers
        - identity activation for last layer
    """
    
    def __init__(self, ns=[8, 8], activation='softplus'):
        """
        Parameters
        ----------
        ns : List, optional
            List of the numbers of nodes per layer. The default is [8, 8].
        activation : String, optional
            Name of the activation function in tensorflow. The default is 'softplus'.

        """
        
        super().__init__()
        self.ls = [tf.keras.layers.Dense(n, activation) 
                   for n in ns[:-1]]
        self.ls += [tf.keras.layers.Dense(ns[-1])]
        
    def call(self, x):
        for l in self.ls:
            x = l(x)
        return x

# %% Define RNN Cells

class NaiveRNNCell(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
     
        self.ls = [layers.Dense(32, 'softplus'),
                   layers.Dense(2)]

        
    def call(self, inputs, states):
        
        #   states are the internal variables
        #   n: current time step, N: next time step
        
        eps_n, dts = inputs
        gamma_n = states[0]
        
        x = tf.concat([eps_n, dts, gamma_n], axis=1)
        for l in self.ls:
            x = l(x)
         
        sig_n, gamma_N = tf.split(x, 2, axis=1)
                
        return sig_n, gamma_N
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # define initial values of the internal variables
        return tf.zeros([batch_size, 1])
    

class AnalyticMaxwellRNNCell(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
        
        self.E_inf  = 0.5
        self.E      = 2
        self.eta    = 1
        
    def call(self, inputs, states):
        
        eps_n, dts = inputs
        gamma_n = states[0]
        
        sig_n = self.E_inf * eps_n + self.E * (eps_n - gamma_n)
        gamma_dot_n = (self.E/self.eta) * (eps_n - gamma_n)
        gamma_N = gamma_n + dts * gamma_dot_n
        
        return sig_n, gamma_N
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # define initial values of the internal variables
        return tf.zeros([batch_size, 1])
    
        
class FFNNMaxwellRNNCell(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
        
        self.E_inf  = 0.5
        self.E      = 2
        self.eta    = 1
        
        self.f_tilde = FFNN([8,8,1])
        
    def call(self, inputs, states):
        
        eps_n, dts = inputs
        gamma_n = states[0]
        
        sig_n = self.E_inf * eps_n + self.E * (eps_n - gamma_n)
        
        x = tf.concat([eps_n, gamma_n], axis=1)
        gamma_dot_n = self.f_tilde(x) * (eps_n - gamma_n)
        
        gamma_N = gamma_n + dts * gamma_dot_n
        
        return sig_n, gamma_N
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # define initial values of the internal variables
        return tf.zeros([batch_size, 1])    


class GSMModelRNNCell(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
        
        self.eta    = 1
        
        self.ffnn = FFNN([8,8,1])
        
    def call(self, inputs, states):
        
        eps_n, dts = inputs
        gamma_n = states[0]
        
        x = tf.concat([eps_n, gamma_n], axis=1)
        with tf.GradientTape() as g:
            g.watch(x)
            e = self.ffnn(x)
        x = g.gradient(e, x)
        de_deps, de_dgamma = tf.split(x, 2, axis=1)
        
        gamma_dot_n = -(1/self.eta) * de_dgamma
        sig_n = de_deps

        gamma_N = gamma_n + dts * gamma_dot_n
        
        return sig_n, gamma_N

# %% Compile RNN Models

def compile_RNN(model_type='naive', **kwargs):
    
    eps = tf.keras.Input(shape=(None, 1), name='input_eps')
    dts = tf.keras.Input(shape=(None, 1), name='input_dts')
    
    if   model_type == 'naive':
        cell = NaiveRNNCell()
    elif model_type == 'analytic_maxwell':
        cell = AnalyticMaxwellRNNCell()
    elif model_type == 'ffnn_maxwell':
        cell = FFNNMaxwellRNNCell()
    elif model_type == 'gsm_model':
        cell = GSMModelRNNCell()
    else:
        raise ValueError(f'Model type {model_type} not implemented')
    
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs   = layer1((eps, dts))

    model = tf.keras.Model([eps, dts], sigs, name=model_type)
    model.compile('adam', 'mse')
    return model