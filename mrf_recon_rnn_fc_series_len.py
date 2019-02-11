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
epochs = 10000
learning_rate = 1.5e-2
display_step = 500
early_stop_step = 5
batch_size = 500

# Network Parameters
num_input = 64 
timesteps = 10 # timesteps
num_hidden = 8
num_output = 2 # number of output parameters

# Fully Connected Layer Parameters
num_in_fc = 100

# tf Graph input
#X = tf.placeholder("float", [None, timesteps, num_in_fc])
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
#dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf', 1000, 10)
dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_const_tr_test', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :]>=dictionary.lut[1, :]]
#D /= np.linalg.norm(D, axis=0)
#dictionary_val = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_val', 1000, 10)
#dictionary_val = dic.dic('recon_q_examples/dict/', 'fisp_mrf_val', 1000, 10)
#D_val = dictionary_val.D[:, dictionary_val.lut[0, :]>=dictionary_val.lut[1, :]]
#D_val /= np.linalg.norm(D_val, axis=0)
permutation = np.random.permutation(D.shape[1])

train_size = int(np.floor(D.shape[1]*0.8))
val_size = D.shape[1]-train_size
#train_size = D.shape[1]
#val_size = D_val.shape[1]
batches_per_epoch  = int(np.floor(train_size / batch_size))

#series_real = np.real(D.T[permutation])
#series_imag = np.imag(D.T[permutation])
#series_mag = np.abs(D.T[permutation])
#Ten percent gaussian noise data
#series_mag = np.abs(D.T[permutation] + 0.01 * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.01 * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape))
series_mag = np.abs(D.T[permutation] + 0.02 * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.02 * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape)).T
series_mag /= np.linalg.norm(series_mag, axis=0)
series_mag = series_mag.T
#series_mag_val = np.abs(D_val.T + 0.1 * np.max(np.real(D_val)) * np.random.normal(0.0, 1.0, D_val.T.shape) + 1j * 0.1 * np.max(np.imag(D_val)) * np.random.normal(0.0, 1.0, D_val.T.shape))
#series_phase = np.angle(D.T[permutation])
#series = np.concatenate([series_mag.T, series_phase.T])
#series = series.T

train_set = [series_mag[batch_size*step:batch_size*(step+1)].reshape((batch_size, timesteps, num_in_fc)) for step in range(batches_per_epoch)]
train_set.append(series_mag[batch_size*batches_per_epoch:train_size].reshape((train_size - batch_size*batches_per_epoch, timesteps, num_in_fc)))
val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_in_fc))
#val_set = series_mag_val.reshape((val_size, timesteps, num_in_fc), order='F')

#relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T[permutation]
relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

train_times = [relaxation_times[batch_size*step:batch_size*(step+1)] for step in range(batches_per_epoch)]
train_times.append(relaxation_times[batch_size*batches_per_epoch:train_size])
val_times = relaxation_times[train_size:train_size+val_size]
#val_times = dictionary_val.lut[:, dictionary_val.lut[0, :] >= dictionary_val.lut[1, :]][0:2].T
#val_times_max = np.max(val_times, axis=0)
#val_times /= val_times_max

from rnn_functions import RNN_with_fc

#val_losses = []
#best_val_losses = []
times = []
errors_t1 = []
errors_t2 = []
p_errors1 = []
p_errors2 = []
for timestep in range(1, timesteps+1):
    with tf.variable_scope('len{}'.format(timestep)):
        X = tf.placeholder("float", [None, timestep, num_in_fc])
        logits = RNN_with_fc(X, num_input, timestep, num_hidden, num_output)

    # Define loss and optimizer
        loss_op = tf.losses.mean_squared_error(Y, logits)
        #loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Y, logits), Y))) # mean averaged percentage error
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
    
    # Evaluate model (with test logits, for dropout to be disabled)
        mse_t1 = tf.losses.mean_squared_error(labels=times_max[0]*Y[:, 0], predictions=times_max[0]*logits[:, 0])
        mse_t2 = tf.losses.mean_squared_error(labels=times_max[1]*Y[:, 1], predictions=times_max[1]*logits[:, 1])
        p_err_t1 = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(times_max[0]*Y[:, 0], times_max[0]*logits[:, 0]), times_max[0]*Y[:, 0])))
        p_err_t2= tf.reduce_mean(tf.abs(tf.divide(tf.subtract(times_max[1]*Y[:, 1], times_max[1]*logits[:, 1]), times_max[1]*Y[:, 1])))
        out = times_max * logits
    
    # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
    
    # Saver
        saver = tf.train.Saver()
    
    # Restoration directory
        ckpt_dir = '../rnn_model_len/rnn_model_len{}/'.format(timestep)
        ckpt_epochs = [990, 940, 1000, 940, 1000, 950, 990, 1000, 950, 980]#[970, 970, 980, 990, 920, 990, 980, 900, 970, 990]
    
    # Start training
        with tf.Session() as sess:
    
    # Run the initializer
    #    sess.run(init)
            ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, ckpt_epochs[timestep-1])
            saver.restore(sess, ckpt_file)


            time, squared_error_t1, squared_error_t2, p_error1, p_error2 = sess.run([out, mse_t1, mse_t2, p_err_t1, p_err_t2],
                                                             feed_dict={X: series_mag[:, :timestep*num_in_fc].reshape((D.shape[1], timestep, num_in_fc)),
                                                                        Y: relaxation_times})
            error_t1 = np.sqrt(squared_error_t1)
            error_t2 = np.sqrt(squared_error_t2)
            times.append(time)
            p_errors1.append(p_error1)
            p_errors2.append(p_error2)
            errors_t1.append(error_t1)
            errors_t2.append(error_t2)

p_errors1 = np.array(p_errors1)
p_errors2 = np.array(p_errors2)
errors_t1 = np.array(errors_t1)
errors_t2 = np.array(errors_t2)

v_loss_len = []
best_v = []
sum_dir = ['../tensorboard_len/' + dir for dir in os.listdir('../tensorboard_len')]
sum_dir.sort(reverse=True)
s100 = sum_dir.pop()
s1000 = sum_dir.pop()
sum_dir.append(s100)
sum_dir.sort()
sum_dir.append(s1000)

for path in sum_dir:
    file_list = os.listdir(path)
    v_loss = []
    for e in tf.train.summary_iterator(path + '/' + file_list[0]):
        for v in e.summary.value:
            if v.tag.find('validation_loss') >= 0:
                v_loss.append(v.simple_value)
    v_loss_len.append(v_loss)
    best_v.append(min(v_loss))
v_loss_len = np.array(v_loss_len)

plt.rc('text', usetex=True)
fig, axs = plt.subplots(5, 4, figsize=(20, 25))
for  k in range(5):
    if k == 0:
        axs[k, 0].set_title(r'\textbf{T1, }'+'{}'.format(k+1)+r'\textbf{ time step}', weight='bold')
    else:
        axs[k, 0].set_title(r'\textbf{T1, }'+'{}'.format(k+1)+r'\textbf{ time steps}', weight='bold')
    axs[k, 0].scatter(times_max[0]*relaxation_times[:, 0]*1e3, times[k][:, 0]*1e3, c='b', marker='.', alpha=0.1)
    axs[k, 0].plot(times_max[0]*relaxation_times[:, 0]*1e3, times_max[0]*relaxation_times[:, 0]*1e3, 'g--')
    axs[k, 0].set_ylabel(r'Predictions (ms)')
    axs[k, 0].set_xlabel(r'Ground truth (ms)')
    if k == 0:
        axs[k, 1].set_title(r'\textbf{T2, }'+'{}'.format(k+1)+r'\textbf{ time step}', weight='bold')
    else:
        axs[k, 1].set_title(r'\textbf{T2, }'+'{}'.format(k+1)+r'\textbf{ time steps}', weight='bold')
    axs[k, 1].scatter(times_max[1]*relaxation_times[:, 1]*1e3, times[k][:, 1]*1e3, c='r', marker='.', alpha=0.1)
    axs[k, 1].plot(times_max[1]*relaxation_times[:, 1]*1e3, times_max[1]*relaxation_times[:, 1]*1e3, 'g--')
    axs[k, 1].set_ylabel(r'Predictions (ms)')
    axs[k, 1].set_xlabel(r'Ground truth (ms)')
    axs[k, 2].scatter(times_max[0]*relaxation_times[:, 0]*1e3, times[k+5][:, 0]*1e3, c='b', marker='.', alpha=0.1)
    axs[k, 2].plot(times_max[0]*relaxation_times[:, 0]*1e3, times_max[0]*relaxation_times[:, 0]*1e3, 'g--')
    axs[k, 2].set_title(r'\textbf{T1, }'+'{}'.format(k+6)+r'\textbf{ time steps}', weight='bold')
    axs[k, 2].set_ylabel(r'Predictions (ms)')
    axs[k, 2].set_xlabel(r'Ground truth (ms)')
    axs[k, 3].scatter(times_max[1]*relaxation_times[:, 1]*1e3, times[k+5][:, 1]*1e3, c='r', marker='.', alpha=0.1)
    axs[k, 3].plot(times_max[1]*relaxation_times[:, 1]*1e3, times_max[1]*relaxation_times[:, 1]*1e3, 'g--')
    axs[k, 3].set_title(r'\textbf{T2, }'+'{}'.format(k+6)+r'\textbf{ time steps}', weight='bold')
    axs[k, 3].set_ylabel(r'Predictions (ms)')
    axs[k, 3].set_xlabel(r'Ground truth (ms)')
fig.show()

# plot validation loss as a function of the series length
x = [n for n in range(1, 11)]
fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
axs2.plot(x, best_v, '.')
axs2.set_title(r'\textbf{Validatiton loss vs series length}', weight='bold')
axs2.set_xlabel(r'Number time steps')
axs2.set_ylabel(r'Best validation loss')
fig2.show()

fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))
axs3[0].plot(x, p_errors1 * 1e2, '.')
axs3[0].set_title(r'\textbf{T1 percentage error vs series length}', weight='bold')
axs3[0].set_xlabel(r'Number time steps')
axs3[0].set_ylabel(r'Percentage error')
axs3[1].plot(x, p_errors2 * 1e2, '.')
axs3[1].set_title(r'\textbf{T2 percentage error vs series length}', weight='bold')
axs3[1].set_xlabel(r'Number time steps')
axs3[1].set_ylabel(r'Percentage error')
fig3.show()
fig5, axs5 = plt.subplots(1, 2, figsize=(10, 5))
axs5[0].plot(x, errors_t1 * 1e3, '.')
axs5[0].set_title(r'\textbf{T1 RMSE vs series length}', weight='bold')
axs5[0].set_xlabel(r'Number time steps')
axs5[0].set_ylabel(r'RMSE (ms)')
axs5[1].plot(x, errors_t2 * 1e3, '.')
axs5[1].set_title(r'\textbf{T2 RMSE vs series length}', weight='bold')
axs5[1].set_xlabel(r'Number time steps')
axs5[1].set_ylabel(r'RMSE (ms)')
fig5.show()

x2 = [n for n in range(1, 1001)]
fig4, axs4 = plt.subplots(1, 1, figsize=(5, 5))
axs4.plot(x2, v_loss_len.T)
axs4.set_xlabel(r'Epoch')
axs4.set_ylabel(r'Validation loss')
fig4.show()

fig10, axs10 = plt.subplots(1, 2, figsize=(10, 5))
axs10[0].scatter(times_max[0]*relaxation_times[:, 0]*1e3, times[9][:, 0]*1e3, c='b', marker='.', alpha=0.1)
axs10[0].plot(times_max[0]*relaxation_times[:, 0]*1e3, times_max[0]*relaxation_times[:, 0]*1e3, 'g--')
axs10[0].set_title(r'\textbf{T1}', weight='bold')
axs10[0].set_ylabel(r'Predictions (ms)')
axs10[0].set_xlabel(r'Ground truth (ms)')
axs10[1].scatter(times_max[1]*relaxation_times[:, 1]*1e3, times[9][:, 1]*1e3, c='r', marker='.', alpha=0.1)
axs10[1].plot(times_max[1]*relaxation_times[:, 1]*1e3, times_max[1]*relaxation_times[:, 1]*1e3, 'g--')
axs10[1].set_title(r'\textbf{T2}', weight='bold')
axs10[1].set_ylabel(r'Predictions (ms)')
axs10[1].set_xlabel(r'Ground truth (ms)')
fig10.show()
