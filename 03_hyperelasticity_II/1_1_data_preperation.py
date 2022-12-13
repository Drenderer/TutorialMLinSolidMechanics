# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:48:05 2022

Task 3: 1.1 Data preparation



Batch normalisation (BN) cant be used:
    ?? Since BN shifts and scales the layers outputs, BN would lead to a material model that essentialy behaves the same, independent of the nominal size of F
    Scaling F is not possible since the calculation of the invariants is nonlinear and would thus lead to a scewed model???
Scaling the output:
    unscaled target variables on regression problems can result in exploding gradients causing the learning process to fail. (https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)
"""

import data_handler_3 as dh

dh.load_case_data('all', plot=True)

