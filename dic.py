#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:45:38 2018

@author: niry
"""

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

    train_set = series_mag[:train_size].reshape((train_size, timesteps, num_in))
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

def read_mrf_data(data_path, Nreps, dim):
    data_mag_path = data_path + '_mag.dat'
    # open the data files
    data_id = open(data_mag_path, 'rb')
    data_mag = np.reshape(np.fromfile(data_id, np.float32), [Nreps, dim, dim])
    data_mag = data_mag.reshape((Nreps, dim*dim))
    data_mag /= np.linalg.norm(data_mag, axis=0)
    return data_mag

def load_relaxation_times_map(data_path, dim, method=0):
    id = open(data_path, 'rb')
    data = np.fromfile(id, np.float32)
    if method == 2:
        return np.reshape(data, [dim, dim])
    else:
        return np.reshape(data, [dim, dim, 2], order='F')
