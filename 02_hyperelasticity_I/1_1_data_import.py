# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:57:54 2022

Task 1.1 Data Import
"""

import data_handler_2 as dh

# %% Simple plot
for file in dh.files.values():
    dh.read_file(file, plot=True)
    
# %% Fancy plot
for file in dh.files.values():
    lc_data = dh.read_file(file)
    dh.plot_data(lc_data, tensor_kw={'legend': True}, dpi=600, figsize=(15,5), n_cols=3, dont_plot=['weight'], title=None)