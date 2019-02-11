# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import dic
import os
import matplotlib.pyplot as plt

## Parallelism configurations
#config = tf.ConfigProto()
#config.intra_op_parallelism_threads = 4
#config.inter_op_parallelism_threads = 4


# Training Parameters
# Training Parameters
epochs = 1000
learning_rate = 4.0e-1
display_step = 50
early_stop_step = 10
batch_size = 500
noise_levels = [1, 2, 5, 10]

# Network Parameters
num_input = 64 
timesteps = 10 # timesteps
num_hidden = 8
num_output = 2 # number of output parameters

# Fully Connected Layer Parameters
num_in_fc = 100

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_in_fc])
Y = tf.placeholder("float", [None, num_output])

## Define weights
#weights = {
#    'out': tf.Variable(tf.random_normal([num_hidden, num_output])/np.sqrt(num_hidden))
#}
#biases = {
#    'out': tf.Variable(tf.random_normal([num_output]))
#}

# Time series and corresponding T1 and T2
#dictionary = dic.dic('recon_q_examples/dict/', 'qti', 260, 10)
#dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_const_tr', 1000, 10)
dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_const_tr_test', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :]>=dictionary.lut[1, :]]
#D /= np.linalg.norm(D, axis=0)
#dictionary_val = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_val_var_tr', 1000, 10)
#D_val = dictionary_val.D[:, dictionary_val.lut[0, :]>=dictionary_val.lut[1, :]]
#D_val /= np.linalg.norm(D_val, axis=0)

#train_size = D.shape[1]
#val_size = D_val.shape[1]
train_size = int(np.floor(D.shape[1]*0.8))
val_size = D.shape[1]-train_size
batches_per_epoch  = int(np.floor(train_size / batch_size))

permutation = np.random.permutation(D.shape[1])

relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

#val_times = dictionary_val.lut[:, dictionary_val.lut[0, :] >= dictionary_val.lut[1, :]][0:2].T
#val_times_max = np.max(val_times, axis=0)
#val_times /= val_times_max

from rnn_functions import RNN_with_fc

times = {1:{}, 2:{}, 5:{}, 10:{}}
errors_t1 = {1:{}, 2:{}, 5:{}, 10:{}}
errors_t2 = {1:{}, 2:{}, 5:{}, 10:{}}

# Restoration directory
ckpt_dir = '../rnn_model_noise/'
ckpt_epoch = [930, 990, 990, 950]
ckpt_list = [ckpt_dir + 'model_fc_noise{}_checkpoint{}.ckpt'.format(noise_levels[m], ckpt_epoch[m]) for m in range(len(noise_levels))]

# Start training
for m in range(len(ckpt_list)):
    with tf.variable_scope('noise{}'.format(noise_levels[m])):
        logits = RNN_with_fc(X, num_input, timesteps, num_hidden, num_output)

# Define loss and optimizer
        loss_op = tf.losses.mean_squared_error(Y, logits)
#       loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Y, logits), Y))) # mean averaged percentage error
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
        mse_t1 = tf.losses.mean_squared_error(labels=times_max[0]*Y[:, 0], predictions=times_max[0]*logits[:, 0])
        mse_t2 = tf.losses.mean_squared_error(labels=times_max[1]*Y[:, 1], predictions=times_max[1]*logits[:, 1])
        out = times_max * logits
#mse_t1 = tf.losses.mean_squared_error(labels=Y[:, 0], predictions=logits[:, 0])
#mse_t2 = tf.losses.mean_squared_error(labels=Y[:, 1], predictions=logits[:, 1])
#out = logits

# Saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, ckpt_list[m])
            for n in range(len(noise_levels)):
                series_test = np.abs(D.T[permutation] + 0.01 * noise_levels[n] * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.01 * noise_levels[n] * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape)).T
                series_test /= np.linalg.norm(series_test, axis=0)
                series_test = series_test.T
#        train_set = [series_mag[batch_size*step:batch_size*(step+1)].reshape((batch_size, timesteps, num_in_fc)) for step in range(batches_per_epoch)]
#        train_set.append(series_mag[batch_size*batches_per_epoch:train_size].reshape((train_size - batch_size*batches_per_epoch, timesteps, num_in_fc)))
#        val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_in_fc))

#        train_times = [relaxation_times[batch_size*step:batch_size*(step+1)] for step in range(batches_per_epoch)]
#        train_times.append(relaxation_times[batch_size*batches_per_epoch:train_size])
#        val_times = relaxation_times[train_size:train_size+val_size]

                times[noise_levels[m]][noise_levels[n]], \
                errors_t1[noise_levels[m]][noise_levels[n]], \
                errors_t2[noise_levels[m]][noise_levels[n]] = sess.run([out, mse_t1, mse_t2],
                                                                       feed_dict={X:series_test.reshape(D.shape[1], timesteps, num_in_fc), Y:relaxation_times})

sum_dir = ['../tensorboard_noise/' + d for d in os.listdir('../tensorboard_noise')]
sum_dir.sort(reverse=True)
s1 = sum_dir.pop()
s10 = sum_dir.pop()
sum_dir.append(s1)
sum_dir.sort()
sum_dir.append(s10)

v_loss_noise = []
best_val_loss = []

for path in sum_dir:
    file_list = os.listdir(path)
    v_loss = []
    for e in tf.train.summary_iterator(path + '/' + file_list[0]):
        for v in e.summary.value:
            if v.tag.find('validation_loss') >= 0:
                v_loss.append(v.simple_value)
    best_val_loss.append(min(v_loss))
    v_loss_noise.append(v_loss)

v_loss_len = np.array(v_loss_noise)

plt.rc('text', usetex=True)
x2 = [n for n in range(1, 1001)]
fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
axs4.plot(x2, v_loss_len.T)
axs4.set_xlabel(r'Epoch')
axs4.set_ylabel(r'Validation loss')
axs4.legend(('1\%', '2\%', '5\%', '10\%'))
fig4.show()
fig5, axs5 = plt.subplots(1, 2, figsize=(10, 5))

x = [err for err in errors_t1[1]]
fig5, axs5 = plt.subplots(1, 2, figsize=(10, 5))
axs5[0].plot(x, [np.sqrt(errors_t1[2][err]) * 1e3 for err in errors_t1[2]], '.')
axs5[0].set_title(r'\textbf{T1 RMSE vs series length}', weight='bold')
axs5[0].set_xlabel(r'Number time steps')
axs5[0].set_ylabel(r'RMSE (ms)')
axs5[1].plot(x, [np.sqrt(errors_t2[2][err]) * 1e3 for err in errors_t2[2]], '.')
axs5[1].set_title(r'\textbf{T2 RMSE vs series length}', weight='bold')
axs5[1].set_xlabel(r'Number time steps')
axs5[1].set_ylabel(r'RMSE (ms)')
fig5.show()

def scatter_plot_noise(level):
    fig10, axs10 = plt.subplots(4, 2, figsize=(10, 20))
    for k in range(4):
        axs10[k, 0].scatter(times_max[0]*relaxation_times[:, 0]*1e3, times[level][noise_levels[k]][:, 0]*1e3, c='b', marker='.', alpha=0.1)
        axs10[k, 0].plot(times_max[0]*relaxation_times[:, 0]*1e3, times_max[0]*relaxation_times[:, 0]*1e3, 'g--')
        axs10[k, 0].set_title(r'\textbf{T1, }' + '{}'.format(noise_levels[k]) + r'\textbf{\% noise}', weight='bold')
        axs10[k, 0].set_ylabel(r'Predictions (ms)')
        axs10[k, 0].set_xlabel(r'Ground truth (ms)')
        axs10[k, 1].scatter(times_max[1]*relaxation_times[:, 1]*1e3, times[level][noise_levels[k]][:, 1]*1e3, c='r', marker='.', alpha=0.1)
        axs10[k, 1].plot(times_max[1]*relaxation_times[:, 1]*1e3, times_max[1]*relaxation_times[:, 1]*1e3, 'g--')
        axs10[k, 1].set_title(r'\textbf{T2, }' + '{}'.format(noise_levels[k]) + r'\textbf{\% noise}', weight='bold')
        axs10[k, 1].set_ylabel(r'Predictions (ms)')
        axs10[k, 1].set_xlabel(r'Ground truth (ms)')
    fig10.show()
    return fig10

figs = []
for l in noise_levels:
    figs.append(scatter_plot_noise(l))