import dic
import rnn_functions
import tensorflow as tf
import numpy as np

epochs = 1000
batch_size = 500
learning_rate = 5e-1
save_step = 10

timesteps = 10
num_input = 100
num_hidden = 8
num_output = 2
num_fc = 64

dictionary, relaxation_times = dic.load_dict('recon_q_examples/dict/', 'fisp_mrf_const_tr', 1000, 10)
train_set, train_times, val_set, val_times, times_max = dic.format_data(dictionary, relaxation_times, timesteps, num_input, noise_level=1)

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])

net = rnn_functions.LSTM(X, timesteps, num_hidden, num_output, fc=True, num_input=num_fc)
rnn_functions.train_lstm(X, Y, net, epochs, batch_size, save_step, tf.losses.mean_squared_error, learning_rate, train_set, train_times, val_set, val_times)
