#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:45:38 2018

@author: niry
"""
import numpy as np

# Dictionary class
class dic:
    
#    def __init__(self, dim1, dim2, Tv):
#        self.dim1 = dim1
#        self.dim2 = dim2
#        shD = (self.dim2, self.dim1)
#        self.D = np.zeros(shD)
#        shN = (1, self.dim1)
#        self.normD = np.zeros(shN)
#        self.lut = np.zeros(shD)
#        shV = (Tv, dim2)
#        self.V = np.zeros(shV)
        
    def __init__(self, dict_path, method_name, Nreps, Tv):
        # dictionary data file paths
        dict_real_path = dict_path + method_name + '_real.dat'
        dict_imag_path = dict_path + method_name + '_imag.dat'
        dict_norm_path = dict_path + method_name + '_norm.dat'
        look_up_table_path = dict_path + method_name + '_lut.dat'
        V_real_path = dict_path + method_name + '_real_sVd_compression.dat'
        V_imag_path = dict_path + method_name + '_imag_sVd_compression.dat'
        dim_path = dict_path + method_name + '_dim.dat'
        
        # open the data files
        real_id = open(dict_real_path, 'rb')
        imag_id = open(dict_imag_path, 'rb')
        norm_id = open(dict_norm_path, 'rb')
        lut_id = open(look_up_table_path, 'rb')
        V_real_id = open(V_real_path, 'rb')
        V_imag_id = open(V_imag_path, 'rb')
        dim_id = open(dim_path, 'rb')
        
        # read dictionary files
        dim1 = int(np.fromfile(dim_id, np.float32)[0])
        dim2 = int(np.float32(Nreps))
        self.dim1 = dim1
        self.dim2 = dim2
                
        D_real = np.reshape(np.fromfile(real_id, np.float32), [dim2, dim1])
        D_imag = np.reshape(np.fromfile(imag_id, np.float32), [dim2, dim1])
        self.D = D_real + 1j*D_imag
        self.normD = np.reshape(np.fromfile(norm_id, np.float32), [1, dim1])
        self.lut = np.reshape(np.fromfile(lut_id, np.float32), [6, dim1])
        V_real = np.reshape(np.fromfile(V_real_id, np.float32), [Tv, dim2])
        V_imag = np.reshape(np.fromfile(V_imag_id, np.float32), [Tv, dim2])
        self.V = V_real + 1j*V_imag
        
        #return self
        
    def setD(self, D):
        self.D = D
        
    def setNormD(self, normD):
        self.normD = normD
    
    def setLUT(self, lut):
        self.lut = lut
        
    def setV(self, V):
        self.V = V
        
   