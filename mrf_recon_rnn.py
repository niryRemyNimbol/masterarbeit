# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import os
import dic

## Parallelism configurations
#config = tf.ConfigProto()
#config.intra_op_parallelism_threads = 4
#config.inter_op_parallelism_threads = 4


# Training Parameters
learning_rate = 8.0e-3
training_steps = 100000


# Network Parameters
num_input = 100 
timesteps = 10 # timesteps
num_hidden = 40 # hidden layer num of features
num_output = 2 # number of output parameters

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])

## Define weights
#weights = {
#    'out': tf.Variable(tf.random_normal([num_hidden, num_output])/np.sqrt(num_hidden))
#}
#biases = {
#    'out': tf.Variable(tf.random_normal([num_output]))
#}

# Time series and corresponding T1 and T2dictionary = dic.dic('recon_q_examples/dict/', 'qti', 260, 10)
dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_test', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :]>=dictionary.lut[1, :]] / np.linalg.norm(dictionary.D, axis=0)
permutation = np.random.permutation(D.shape[1])
series_real = np.real(D.T[permutation])
series_imag = np.imag(D.T[permutation])
series_mag = np.abs(dictionary.D.T[permutation])
series_phase = np.angle(dictionary.D.T[permutation])
series = np.concatenate([series_mag.T, series_phase.T])
series = series.T

relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max


#def RNN(x):
#
#    # Prepare data shape to match `rnn` function requirements
#    # Current data input shape: (batch_size, timesteps, n_input)
#    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
#
#    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
#    x = tf.unstack(x, timesteps, 1)
#
#    # Define a lstm cell with tensorflow
#    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, 
#                                  reuse=tf.AUTO_REUSE)
#
#    # Get lstm cell output
#    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#    # Linear activation, using rnn inner loop last output
#    return tf.layers.dense(outputs[-1], num_output, activation=tf.sigmoid, kernel_regularizer=tf.norm)
#sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])

from rnn_functions import RNN

logits = RNN(X, timesteps, num_hidden, num_output)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(Y, logits)
#loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Y, logits), Y)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
mse_t1 = tf.losses.mean_squared_error(labels=times_max[0]*Y[:, 0], predictions=times_max[0]*logits[:, 0])
mse_t2 = tf.losses.mean_squared_error(labels=times_max[1]*Y[:, 1], predictions=times_max[1]*logits[:, 1])
out = times_max * logits

# Saver
saver = tf.train.Saver()

# Restoration directory
ckpt_dir = '../rnn_model/'

# Start training
with tf.Session() as sess:

    # Restore trained network
#    ckpt_file = ckpt_dir + 'model_0_checkpoint10000_mse.ckpt'
    ckpt_file = ckpt_dir + 'model_2_checkpoint10000.ckpt'
    saver.restore(sess, ckpt_file)
#    os.remove(ckpt_file + '.index')
#    os.remove(ckpt_file + '.meta')
#    os.remove(ckpt_file + '.data-00000-of-00001')
#    os.remove('rnn_model/checkpoint')

#    for step in range(1, training_steps+1):
#        batch_x = series_mag[0:train_size]
#        batch_x = batch_x.reshape((train_size, timesteps, num_input), order='F')
#        batch_y = relaxation_times[0:train_size]
#        # Run optimization op (backprop)
#        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#        if step % display_step == 0 or step == 1:
#            # Calculate batch loss and accuracy
#            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
#            print("Step " + str(step) + ", Minibatch Loss= " + \
#                  "{:.8f}".format(loss))    
#        if step == training_steps:
#            # Save trained network
##            ckpt_file = ckpt_dir + 'model_0_checkpoint{}_mse.ckpt'.format(step)
#            ckpt_file = ckpt_dir + 'model_2_checkpoint{}.ckpt'.format(step)
#            saver.save(sess, ckpt_file)
#    print("Optimization Finished!")

#     Calculate MSE for test time series
    times, squared_error_t1, squared_error_t2 = sess.run([out, mse_t1, mse_t2], 
                                                         feed_dict={X: series_mag.reshape((D.shape[1], timesteps, num_input), order='F'),
                                                                    Y: relaxation_times})
    error_t1 = np.sqrt(squared_error_t1)
    error_t2 = np.sqrt(squared_error_t2)

error = 0
square_error = 0
for i in range(len(times)):
    error += np.abs(times[i]-relaxation_times[train_size+i]*times_max)
    square_error += (times[i]-relaxation_times[train_size+i]*times_max)**2
error /= len(times)
square_error /= len(times)
rmserror = np.sqrt(square_error)
#    W1 = weights['h1'].eval(sess)
#    W2 = weights['h2'].eval(sess)
#    Wout = weights['out'].eval(sess)
#    
#    b1 = biases['b1'].eval(sess)
#    b2 = biases['b2'].eval(sess)
#    bout = biases['out'].eval(sess)
    
