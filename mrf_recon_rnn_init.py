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
learning_rate = 5.0e-2
training_steps = 50000
display_step = 100

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

# Time series and corresponding T1 and T2
#dictionary = dic.dic('recon_q_examples/dict/', 'qti', 260, 10)
dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_train', 1000, 10)
D = dictionary.D / np.linalg.norm(dictionary.D, axis=0)
permutation = np.random.permutation(D.shape[1])

train_size = int(np.floor(D.shape[1]*0.8))
val_size = D.shape[1]-train_size

series_real = np.real(D.T[permutation])
series_imag = np.imag(D.T[permutation])
series_mag = np.abs(dictionary.D.T[permutation])
series_phase = np.angle(dictionary.D.T[permutation])
series = np.concatenate([series_mag.T, series_phase.T])
series = series.T

#test_set = series_mag[train_size+val_size:].reshape((test_size, timesteps, num_input), order='F')
val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_input), order='F')

relaxation_times = dictionary.lut[0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

from rnn_functions import RNN

logits = RNN(X, timesteps, num_hidden, num_output)

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
train_loss_summary = tf.summary.scalar('training_loss', loss_op)
val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
#merged = tf.summary.merge_all()

# Saver
saver = tf.train.Saver()

# Restoration directory
ckpt_dir = '../rnn_model/'

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
        
    train_loss_writer = tf.summary.FileWriter('../tensorboard/training_loss/', sess.graph)
    val_loss_writer = tf.summary.FileWriter('../tensorboard/validation_loss/', sess.graph)
    
    for step in range(1, training_steps+1):
        batch_x = series_mag[0:train_size]
        batch_x = batch_x.reshape((train_size, timesteps, num_input), order='F')
        batch_y = relaxation_times[0:train_size]
        
        # Training, validation and loss computation
        training, loss, summary = sess.run([train_op, loss_op, train_loss_summary], feed_dict={X: batch_x, Y: batch_y})
        train_loss_writer.add_summary(summary, step)
        
#        # Validation
#        val_loss, val_loss_sum = sess.run([loss_op, val_loss_summary], feed_dict={X:batch_x, Y:batch_y})
#        val_loss_writer.add_summary(val_loss_sum, step)
        val_loss, val_summary = sess.run([loss_op, val_loss_summary], feed_dict={X: val_set, Y: relaxation_times[train_size:train_size+val_size]})
        val_loss_writer.add_summary(val_summary, step)
        
        if step % display_step == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.10f}".format(loss))
        
        if step == training_steps:
            # Save trained network
            ckpt_file = ckpt_dir + 'model_2_checkpoint{}.ckpt'.format(step)
            saver.save(sess, ckpt_file)
    
    print("Optimization Finished!")

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
    
