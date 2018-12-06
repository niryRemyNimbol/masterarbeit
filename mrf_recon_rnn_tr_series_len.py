# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import dic
import matplotlib.pyplot as plt

## Parallelism configurations
#config = tf.ConfigProto()
#config.intra_op_parallelism_threads = 4
#config.inter_op_parallelism_threads = 4


# Training Parameters
epochs = 1000
learning_rate = 5.5e-1
display_step = 20
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
dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :]>=dictionary.lut[1, :]]
D /= np.linalg.norm(D, axis=0)
dictionary_val = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_val', 1000, 10)
D_val = dictionary_val.D[:, dictionary_val.lut[0, :]>=dictionary_val.lut[1, :]]
D_val /= np.linalg.norm(D_val, axis=0)
permutation = np.random.permutation(D.shape[1])

#train_size = int(np.floor(D.shape[1]*0.8))
#val_size = D.shape[1]-train_size
train_size = D.shape[1]
val_size = D_val.shape[1]
batches_per_epoch  = int(np.floor(train_size / batch_size))

#series_real = np.real(D.T[permutation])
#series_imag = np.imag(D.T[permutation])
#series_mag = np.abs(D.T[permutation])
#Ten percent gaussian noise data
#series_mag = np.abs(D.T[permutation] + 0.01 * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.01 * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape))
series_mag = np.abs(D.T[permutation] + 0.01 * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.01 * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape))
series_mag_val = np.abs(D_val.T + 0.01 * np.max(np.real(D_val)) * np.random.normal(0.0, 1.0, D_val.T.shape) + 1j * 0.01 * np.max(np.imag(D_val)) * np.random.normal(0.0, 1.0, D_val.T.shape))
#series_phase = np.angle(D.T[permutation])
#series = np.concatenate([series_mag.T, series_phase.T])
#series = series.T

train_set = [series_mag[batch_size*step:batch_size*(step+1)].reshape((batch_size, timesteps, num_in_fc), order='F') for step in range(batches_per_epoch)]
train_set.append(series_mag[batch_size*batches_per_epoch:train_size].reshape((train_size - batch_size*batches_per_epoch, timesteps, num_in_fc), order='F'))
#val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_in_fc), order='F')
val_set = series_mag_val.reshape((val_size, timesteps, num_in_fc), order='F')

#relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T[permutation]
relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

train_times = [relaxation_times[batch_size*step:batch_size*(step+1)] for step in range(batches_per_epoch)]
train_times.append(relaxation_times[batch_size*batches_per_epoch:train_size])
#val_times = relaxation_times[train_size:train_size+val_size]
val_times = dictionary_val.lut[:, dictionary_val.lut[0, :] >= dictionary_val.lut[1, :]][0:2].T
val_times_max = np.max(val_times, axis=0)
val_times /= val_times_max

from rnn_functions import RNN_with_tr

val_losses = []
best_val_losses = []
t1_err = {}
t2_err = {}
for timestep in range(1, timesteps+1):
    X = tf.placeholder("float", [None, timestep, num_in_fc])
    logits = RNN_with_tr(X, num_input, timestep, num_hidden, num_output)

    # Define loss and optimizer
    loss_ops = [tf.losses.mean_squared_error(Y, logit) for logit in logits]
    #loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Y, logits), Y))) # mean averaged percentage error
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_ops = [optimizer.minimize(loss_op) for loss_op in loss_ops]
    
    # Evaluate model (with test logits, for dropout to be disabled)
    mse_t1 = [tf.losses.mean_squared_error(labels=times_max[0]*Y[:, 0], predictions=times_max[0]*logit[:, 0]) for logit in logits]
    mse_t2 = [tf.losses.mean_squared_error(labels=times_max[1]*Y[:, 1], predictions=times_max[1]*logit[:, 1]) for logit in logits]
    out = [times_max * logit for logit in logits]
    
    t1_err[timestep] = []
    t2_err[timestep] = []
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Summaries to view in tensorboard
    #            train_loss_summary = tf.summary.scalar('training_loss', loss_op)
#    val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
    #merged = tf.summary.merge_all()
    
    # Saver
    saver = tf.train.Saver()
    
    # Restoration directory
    ckpt_dir = '../rnn_model/'
    
    # Start training
    with tf.Session() as sess:
    
    # Run the initializer
        sess.run(init)
    
    #                train_loss_writer = tf.summary.FileWriter('tensorboard/training_loss/', sess.graph)
#        val_loss_writer = tf.summary.FileWriter('tensorboard/validation_loss_len{}/'.format(num_in_fc*timestep), sess.graph)
        
        for epoch in range(1, epochs+1):
    #                    batch_x = series_mag[(step-1)%32 * batch_size:min(((step-1)%32+1) * batch_size, series_mag.shape[0])]
    #                    batch_x = batch_x.reshape((batch_x.shape[0], timesteps, num_input), order='F')
    #                    batch_y = relaxation_times[(step-1)%32 * batch_size:min(((step-1)%32+1) * batch_size, series_mag.shape[0])]
            total_loss = 0
            for k in range(len(train_set)):
                batch_x = train_set[k][:,:timestep,:]
                batch_y = train_times[k]
                
                training, batch_loss = sess.run([train_ops, loss_ops], feed_dict={X: batch_x, Y:batch_y})
                total_loss += batch_loss[-1]
    # Training, validation and loss computation
    #                    training, loss, summary = sess.run([train_op, loss_op, train_loss_summary], feed_dict={X: batch_x, Y: batch_y})
    #                    train_loss_writer.add_summary(summary, step)
    
    #        # Validation
    #        val_loss, val_loss_sum = sess.run([loss_op, val_loss_summary], feed_dict={X:batch_x, Y:batch_y})
    #        val_loss_writer.add_summary(val_loss_sum, step)
#            val_loss, val_summary = sess.run([loss_op, val_loss_summary], feed_dict={X: val_set[:, :timestep, :], Y: val_times})
#            val_loss_writer.add_summary(val_summary, epoch)
            total_loss /= len(train_set) 
            val_loss, mse1, mse2 = sess.run([loss_ops, mse_t1, mse_t2], feed_dict={X: val_set[:, :timestep, :], Y: val_times})
            val_losses.append(val_loss[-1])
            t1_err[timestep].append(mse1)
            t2_err[timestep].append(mse2)
            
            # Reshuffling the train set
            permutation = np.random.permutation(D.shape[1])
            series_mag = series_mag[permutation]
            train_set = [series_mag[batch_size*step:batch_size*(step+1)].reshape((batch_size, timesteps, num_in_fc), order='F') for step in range(batches_per_epoch)]
            train_set.append(series_mag[batch_size*batches_per_epoch:train_size].reshape((train_size - batch_size*batches_per_epoch, timesteps, num_in_fc), order='F'))
            relaxation_times = relaxation_times[permutation]            
            train_times = [relaxation_times[batch_size*step:batch_size*(step+1)] for step in range(batches_per_epoch)]
            train_times.append(relaxation_times[batch_size*batches_per_epoch:train_size])
    
            if epoch % display_step == 1:
                print("Epoch " + str(epoch) + ", average training loss= " + "{:.10f}".format(total_loss))
                print("Epoch " + str(epoch) + ", validation loss= " + "{:.10f}".format(val_loss[-1]))
        
            if epoch == epochs:
    # Save trained network
#                ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, epoch)
#                saver.save(sess, ckpt_file)
                best_val_losses.append(min(val_losses))
    
    print("Optimization Finished!")

# plot validation loss as a function of the series length    
fig = plt.figure()
fig.add_axes([0.2,0.2,0.6,0.6])
fig.axes[0].plot(best_val_losses)
fig.text(0.5,0.9,"Validation loss vs series length", weight='bold', verticalalignment='top', horizontalalignment='center', size=14)
#fig.savefig('series_length_ter.jpg')

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
    
