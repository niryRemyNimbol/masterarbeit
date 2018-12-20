# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import dic

## Parallelism configurations
#config = tf.ConfigProto()
#config.intra_op_parallelism_threads = 4
#config.inter_op_parallelism_threads = 4


# Training Parameters
learning_rate = 1.0e1
training_steps = 5000
train_size = 2000
val_size = 200
test_size = 200
display_step = 100
batch_size = 600

# Network Parameters
num_input = 100 
timesteps = 10 # timesteps
num_hidden = 40 # hidden layer num of features
num_output = 2 # number of output parameters

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])

# Define weights
#weights = {
#    'out': [tf.Variable(tf.random_normal([num_hidden, num_output])/np.sqrt(num_hidden)) for timestep in range(timesteps)]
#}

#biases = {
#    'out': [tf.Variable(tf.random_normal([num_output])) for timestep in range(timesteps)]
#}

# Time series and corresponding T1 and T2
#dictionary = dic.dic('recon_q_examples/dict/', 'qti', 260, 10)
dictionary = dic.dic('recon_q_examples/dict/', 'fisp_mrf_bis', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :] >= dictionary.lut[1, :]] 
D /= np.linalg.norm(D, axis=0)
permutation = np.random.permutation(D.shape[1])

series_real = np.real(D.T[permutation])
series_imag = np.imag(D.T[permutation])
series_mag = np.abs(D.T[permutation] + np.random.normal(0, 0.01*np.mean(np.max(np.real(D))), D.T.shape) + 1j * np.random.normal(0, 0.01*np.mean(np.max(np.imag(D))), D.T.shape))
series_phase = np.angle(D.T[permutation])
series = np.concatenate([series_mag.T, series_phase.T])
series = series.T

test_set = series_mag[train_size:train_size+val_size].reshape((test_size, timesteps, num_input), order='C') #+ np.random.normal(0, 0.5, size=(test_size, timesteps, num_input))
#val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_input), order='C')

relaxation_times = dictionary.lut[0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

#val_times = np.repeat(relaxation_times[train_size:train_size+val_size], timesteps, axis=1)
#val_times = val_times.reshape((timesteps, val_size, num_output), order='F')
test_times = np.repeat(relaxation_times[train_size:train_size+val_size], timesteps, axis=1)
test_times = test_times.reshape((test_size, timesteps, num_output), order='F')

from rnn_functions import LSTM

model = LSTM(num_input, timesteps, num_hidden, num_output)#, weights['out'], biases['out'])

model.compile(optimizer='sgd', loss='mse')

batch_x = series_mag[0:train_size]
batch_x = batch_x.reshape((train_size, timesteps, num_input), order='C')
batch_y = np.repeat(relaxation_times[0:train_size], timesteps, axis=1)
batch_y = batch_y.reshape((train_size, timesteps, num_output), order='F')
    

model.fit(batch_x, batch_y, epochs=training_steps, batch_size=batch_size, verbose=1, validation_split=0.1)

square_error = model.evaluate(test_set, test_times)
times = model.predict(test_set)

error = 0
square_error = 0
for i in range(len(times)):
    error += np.abs(times[i]-relaxation_times[train_size+i]*times_max)
    square_error += (times[i]-relaxation_times[train_size+i]*times_max)**2
error /= len(times)
square_error /= len(times)

