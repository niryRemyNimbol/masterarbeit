import dic
import rnn_functions
import tensorflow as tf
import numpy as np

epochs = 2000
batch_size = 500
learning_rate = 2e-1
save_step = 10

timesteps = 1000
num_input = 1
num_hidden = 1
num_output = 2
#num_fc = 64

dictionary, relaxation_times = dic.load_dict('recon_q_examples/dict/', 'fisp_mrf_const_tr', 1000, 10)
train_set, train_times, val_set, val_times, times_max = dic.format_data(dictionary, relaxation_times, timesteps, num_input, noise_level=5)

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])

net = rnn_functions.LSTM(X, timesteps, num_hidden, num_output, activation=tf.sigmoid, fc=True, tr=True, out_step=100, num_input=num_fc)
rnn_functions.train_lstm(X, Y, net, epochs, batch_size, save_step, rnn_functions.mape_loss, learning_rate, train_set, train_times, val_set, val_times, tr=True)
