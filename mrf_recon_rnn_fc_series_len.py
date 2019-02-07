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
series_mag = np.abs(D.T[permutation] + 0.1 * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.1 * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape)).T
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
for timestep in range(1, timesteps+1):
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
    out = times_max * logits
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Summaries to view in tensorboard
    #            train_loss_summary = tf.summary.scalar('training_loss', loss_op)
#    val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
    #merged = tf.summary.merge_all()
    
    # Saver
    saver = tf.train.Saver()
    
    # Restoration directory
    ckpt_dir = '../rnn_model_len_new/rnn_model_len{}/'.format(timestep)
    ckpt_epochs = [970, 970, 980, 990, 920, 990, 980, 900, 970, 990]
    
    # Start training
    with tf.Session() as sess:
    
    # Run the initializer
    #    sess.run(init)
        ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, ckpt_epochs[timestep-1])
        saver.restore(sess, ckpt_file)


        time, squared_error_t1, squared_error_t2 = sess.run([out, mse_t1, mse_t2],
                                                         feed_dict={X: series_mag[:, :timestep*num_in_fc].reshape((D.shape[1], timestep, num_in_fc)),
                                                                    Y: relaxation_times})
        error_t1 = np.sqrt(squared_error_t1)
        error_t2 = np.sqrt(squared_error_t2)
        times.append(time)
        errors_t1.append(error_t1)
        errors_t2.append(error_t2)


v_loss_len = []
best_v = []
sum_dir = ['../tensorboard_len_new/' + dir for dir in os.listdir('../tensorboard_len_new')]
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

fig, axs = plt.subplots(2, 10, figsize=(50, 10))
for  k in range(len(times)):
    axs[0, k].plot(times_max[0]*relaxation_times[:, 0]*1e3, times[k][:, 0]*1e3, 'b.')
    axs[0, k].plot(times_max[0]*relaxation_times[:, 0]*1e3, times_max[0]*relaxation_times[:, 0]*1e3, 'g--')
    axs[0, k].set_title('T1, timestep {}'.format(k+1), weight='bold')
    axs[0, k].set_ylabel('Predictions (ms)')
    axs[0, k].set_xlabel('Ground truth (ms)')
    axs[1, k].plot(times_max[1]*relaxation_times[:, 1]*1e3, times[k][:, 1]*1e3, 'r.')
    axs[1, k].plot(times_max[1]*relaxation_times[:, 1]*1e3, times_max[1]*relaxation_times[:, 1]*1e3, 'g--')
    axs[1, k].set_title('T2, timestep {}'.format(k+1), weight='bold')
    axs[1, k].set_ylabel('Predictions (ms)')
    axs[1, k].set_xlabel('Ground truth (ms)')
fig.show()

# plot validation loss as a function of the series length
x = [n for n in range(1, 11)]
x2 = [n for n in range(1, 1001)]
fig2, axs2 = plt.subplots(4, 1, figsize=(5, 20))
axs2[0].plot(x, best_v, '.')
axs2[0].set_title('Validatiton loss vs series length', weight='bold')
axs2[0].set_xlabel('# time steps')
axs2[0].set_ylabel('Best validation loss')
axs2[1].plot(x, errors_t1 * 1e3, '.')
axs2[1].set_title('T1 RMSE vs series length', weight='bold')
axs2[1].set_xlabel('# time step')
axs2[1].set_ylabel('RMSE (ms)')
axs2[2].plot(x, errors_t2 * 1e3, '.')
axs2[2].set_title('T2 RMSE vs series length', weight='bold')
axs2[2].set_xlabel('# time step')
axs2[2].set_ylabel('RMSE (ms)')
axs2[3].plot(x2, v_loss_len.T)
axs2[3].set_xlabel('Epoch')
axs2[3].set_ylabel('Validation loss')
fig2.show()


#     Calculate MSE for test time series
#    times, squared_error_t1, squared_error_t2 = sess.run([out, mse_t1, mse_t2], 
#                                         feed_dict={X: test_set,
#                                                    Y: relaxation_times[train_size+val_size:]})
#    error_t1 = np.sqrt(squared_error_t1)
#    error_t2 = np.sqrt(squared_error_t2)

#    W1 = weights['h1'].eval(sess)
#    W2 = weights['h2'].eval(sess)
#    Wout = weights['out'].eval(sess)
#    
#    b1 = biases['b1'].eval(sess)
#    b2 = biases['b2'].eval(sess)
#    bout = biases['out'].eval(sess)
    
