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
epochs = 1000
learning_rate = 5.0e-1
display_step = 200
early_stop_step = 10
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
dictionary = dic.dic('recon_q_examples/dict/', 'fisp_mrf_const_tr', 1000, 10)
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
        out = times_max * logits
    
    # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
    
    # Summaries to view in tensorboard
#    train_loss_summary = tf.summary.scalar('training_loss', loss_op)
        val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
    #merged = tf.summary.merge_all()
    
    # Saver
        saver = tf.train.Saver()
    
    # Restoration directory
        ckpt_dir = 'rnn_model/rnn_model_len{}/'.format(timestep)
    
    # Start training
        with tf.Session() as sess:
    
    # Run the initializer
            sess.run(init)
    
#        train_loss_writer = tf.summary.FileWriter('tensorboard/training_loss/', sess.graph)
            val_loss_writer = tf.summary.FileWriter('tensorboard/validation_loss_len{}/'.format(num_in_fc*timestep), sess.graph)
            counter = 0
            for epoch in range(1, epochs+1):
    #                    batch_x = series_mag[(step-1)%32 * batch_size:min(((step-1)%32+1) * batch_size, series_mag.shape[0])]
    #                    batch_x = batch_x.reshape((batch_x.shape[0], timesteps, num_input), order='F')
    #                    batch_y = relaxation_times[(step-1)%32 * batch_size:min(((step-1)%32+1) * batch_size, series_mag.shape[0])]
                total_loss = 0
                for k in range(len(train_set)):
                    batch_x = train_set[k][:,:timestep,:]
                    batch_y = train_times[k]

     #           training, batch_loss, summary = sess.run([train_op, loss_op, train_loss_summary], feed_dict={X: batch_x, Y: batch_y})
     #           train_loss_writer.add_summary(summary, step)
                    training, batch_loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y:batch_y})
                    total_loss += batch_loss
    # Training, validation and loss computation

    #        # Validation
#            val_loss, val_loss_sum = sess.run([loss_op, val_loss_summary], feed_dict={X:batch_x, Y:batch_y})
#            val_loss, val_summary = sess.run([loss_op, val_loss_summary], feed_dict={X: val_set[:, :timestep, :], Y: val_times})
#            val_loss_writer.add_summary(val_summary, epoch)
                total_loss /= len(train_set)
                val_loss, val_loss_sum = sess.run([loss_op, val_loss_summary], feed_dict={X: val_set[:, :timestep, :], Y: val_times})
                val_loss_writer.add_summary(val_loss_sum, epoch)
#            val_losses.append(val_loss)
            # Reshuffling the train set
                permutation = np.random.permutation(D.shape[1])
                series_mag = series_mag[permutation]
                train_set = [series_mag[batch_size*step:batch_size*(step+1)].reshape((batch_size, timesteps, num_in_fc)) for step in range(batches_per_epoch)]
                train_set.append(series_mag[batch_size*batches_per_epoch:train_size].reshape((train_size - batch_size*batches_per_epoch, timesteps, num_in_fc)))
                relaxation_times = relaxation_times[permutation]
                train_times = [relaxation_times[batch_size*step:batch_size*(step+1)] for step in range(batches_per_epoch)]
                train_times.append(relaxation_times[batch_size*batches_per_epoch:train_size])


                if epoch % display_step == 1:
                    print("Epoch " + str(epoch) + ", average training loss= " + "{:.10f}".format(total_loss))
                    print("Epoch " + str(epoch) + ", validation loss= " + "{:.10f}".format(val_loss))

                if epoch == 1:
                    ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, 0)
                    saver.save(sess, ckpt_file)
                    best_loss = val_loss
                elif epoch % early_stop_step == 0:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        prev_ckpt = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, epoch-10*(counter+1))
                        ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, epoch)
                        saver.save(sess, ckpt_file)
                        os.remove(prev_ckpt + '.index')
                        os.remove(prev_ckpt + '.meta')
                        os.remove(prev_ckpt + '.data-00000-of-00001')
                        counter = 0
                    else:
                        counter += 1

        print("Optimization Finished! Best loss: {}".format(best_loss))

