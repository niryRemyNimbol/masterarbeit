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


def load_dict(dict_path, method_name, Nreps, Tv):
    # dictionary data file paths
    dict_real_path = dict_path + method_name + '_real.dat'
    dict_imag_path = dict_path + method_name + '_imag.dat'
    look_up_table_path = dict_path + method_name + '_lut.dat'
    dim_path = dict_path + method_name + '_dim.dat'

    # open the data files
    real_id = open(dict_real_path, 'rb')
    imag_id = open(dict_imag_path, 'rb')
    lut_id = open(look_up_table_path, 'rb')
    dim_id = open(dim_path, 'rb')

    # read dictionary files
    dim1 = int(np.fromfile(dim_id, np.float32)[0])
    dim2 = int(np.float32(Nreps))

    D_real = np.reshape(np.fromfile(real_id, np.float32), [dim2, dim1])
    D_imag = np.reshape(np.fromfile(imag_id, np.float32), [dim2, dim1])
    D = D_real + 1j*D_imag
    lut = np.reshape(np.fromfile(lut_id, np.float32), [6, dim1])

    return D, lut

def format_data(D, lut, timesteps, num_in, noise_level=2):
    D = D[:, lut[0, :] >= lut[1, :]]

    train_size = int(np.floor(D.shape[1]*0.8))
    val_size = D.shape[1]-train_size

    permutation = np.random.permutation(D.shape[1])
    series_mag = np.abs(D.T[permutation] + 0.01 * noise_level * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.01 * noise_level * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape)).T
    series_mag /= np.linalg.norm(series_mag, axis=0)
    series_mag = series_mag.T

    train_set = series_mag[:train_size+val_size].reshape((train_size, timesteps, num_in))
    val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_in))

    relaxation_times = lut[:, lut[0, :] >= lut[1, :]][0:2].T[permutation]
    times_max = np.max(relaxation_times, axis=0)
    relaxation_times /= times_max

    train_times = relaxation_times[:train_size]
    val_times = relaxation_times[train_size:train_size+val_size]

    return train_set, train_times, val_set, val_times, times_max

def build_batches(data, target, batch_size):
    train_size, timesteps, num_in = data.shape
    batches_per_epoch  = int(np.floor(train_size / batch_size))

    train_set = [data[batch_size*step:batch_size*(step+1)].reshape((batch_size, timesteps, num_in)) for step in range(batches_per_epoch)]
    train_set.append(data[batch_size*batches_per_epoch:train_size].reshape((train_size - batch_size*batches_per_epoch, timesteps, num_in)))

    train_times = [target[batch_size*step:batch_size*(step+1)] for step in range(batches_per_epoch)]
    train_times.append(target[batch_size*batches_per_epoch:train_size])

    return train_set, train_times

def shuffle(data, target):
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation]
    target = target[permutation]
    return data, target
