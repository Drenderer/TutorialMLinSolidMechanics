# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:57:54 2022

Task 1.1 Data Import
"""

import data_handler as dh

for file in dh.files.values():
    dh.read_file(file, plot=True)