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
learning_rate = 1.0e-1
training_steps = 50000
train_size = 0
batch_size = 600
val_size = 0
test_size = 2218
display_step = 100

# Network Parameters
num_input = 100 
timesteps = 10 # timesteps
num_hidden = 40 # hidden layer num of features
num_output = 2 # number of output parameters

# tf Graph input


# Define weights
#weights = {
#    'out': [tf.Variable(tf.random_normal([num_hidden, num_output])/np.sqrt(num_hidden)) for timestep in range(timesteps)]
#}

#biases = {
#    'out': [tf.Variable(tf.random_normal([num_output])) for timestep in range(timesteps)]
#}

# Time series and corresponding T1 and T2
dictionary = dic.dic('recon_q_examples/dict/', 'qti', 260, 10)
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

train_set =  series_mag[:train_size].reshape((train_size, timesteps, num_input), order='C')
test_set = series_mag[train_size+val_size:train_size+val_size+test_size].reshape((test_size, timesteps, num_input), order='C') #+ np.random.normal(0, 0.5, size=(test_size, timesteps, num_input))
val_set = series_mag[train_size:train_size+val_size].reshape((val_size, timesteps, num_input), order='C')

relaxation_times = dictionary.lut[0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

train_times = relaxation_times[:train_size]

times = np.zeros((timesteps, test_size, num_output))
squared_error_t1 = np.zeros((timesteps, 1))
squared_error_t2 = np.zeros((timesteps, 1))

from rnn_functions import RNN

for timestep in range(1,timesteps+1):
    train_set =  series_mag[:train_size,:timestep*num_input].reshape((train_size, timestep, num_input), order='C')
    test_set = series_mag[train_size+val_size:train_size+val_size+test_size,:timestep*num_input].reshape((test_size, timestep, num_input), order='C') #+ np.random.normal(0, 0.5, size=(test_size, timesteps, num_input))
    val_set = series_mag[train_size:train_size+val_size,:timestep*num_input].reshape((val_size, timestep, num_input), order='C')
    
    X = tf.placeholder("float", [None, timestep, num_input])
    Y = tf.placeholder("float", [None, num_output])
    
    logits = RNN(X, timestep, num_hidden, num_output)#, weights['out'], biases['out'])
    
    # Define loss and optimizer
#    loss_op = tf.losses.mean_squared_error(Y, logits)
#    #loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Y, logits), Y)))
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#    train_op = optimizer.minimize(loss_op)
#
#    # Evaluate model (with test logits, for dropout to be disabled)
    mse_t1 = tf.losses.mean_squared_error(labels=times_max[0]*Y[0], predictions=times_max[0]*logits[0])
    mse_t2 = tf.losses.mean_squared_error(labels=times_max[1]*Y[1], predictions=times_max[1]*logits[1])
    out = times_max * logits
#
#    # Initialize the variables (i.e. assign their default value)
#    init = tf.global_variables_initializer()
#
#    # Summaries to view in tensorboard
#    train_loss_summary = tf.summary.scalar('training_loss'.format(timestep), loss_op)
#    val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
#    #merged = tf.summary.merge_all()

    # Saver
    saver = tf.train.Saver()

    # Restoration directory
    ckpt_dir = 'rnn_model/'

    # Start training
    with tf.Session() as sess:

            # Run the initializer
#        sess.run(init)
        ckpt_file = ckpt_dir + 'model_{}_checkpoint{}.ckpt'.format(timestep, 50000)
        saver.restore(sess, ckpt_file)        
#        
#        train_loss_writer = tf.summary.FileWriter('tensorboard/training_loss{}/'.format(timestep), sess.graph)
#        val_loss_writer = tf.summary.FileWriter('tensorboard/validation_loss{}/'.format(timestep), sess.graph)
        
#        for step in range(training_steps):
#            batch_x = train_set[(step%3)*batch_size:(step%3+1)*batch_size]
#            batch_x = batch_x.reshape((batch_size, timestep, num_input), order='C')
#            batch_y = train_times[(step%3)*batch_size:(step%3+1)*batch_size]
#        
#            # Training, validation and loss computation
##            training, loss = sess.run([train_op, loss_op], feed_dict={X:batch_x, Y:batch_y})
#            training, loss, summary = sess.run([train_op, loss_op, train_loss_summary], feed_dict={X: batch_x, Y: batch_y})
#            train_loss_writer.add_summary(summary, step)
#            
##           # Validation
##            val_loss = sess.run(loss_op, feed_dict={X: val_set, Y: relaxation_times[train_size:train_size+val_size]})
##           val_loss, val_loss_sum = sess.run([loss_op, val_loss_summary], feed_dict={X:batch_x, Y:batch_y})
##           val_loss_writer.add_summary(val_loss_sum, step)
#            val_loss, val_summary = sess.run([loss_op, val_loss_summary], feed_dict={X: val_set, Y: relaxation_times[train_size:train_size+val_size]})
#            val_loss_writer.add_summary(val_summary, step)
#            
#            if step % display_step == 0 or step == 1:
#                print("Step " + str(step) + ", Minibatch Loss= " + \
#                      "{:.10f}".format(loss))
#                
#            if step == training_steps-1:
#                    # Save trained network
#                    ckpt_file = ckpt_dir + 'model_{}_checkpoint{}.ckpt'.format(timestep, step+1)
#                    saver.save(sess, ckpt_file)
#                    
#                    print("Optimization Finished!")

    #     Calculate MSE for test time series
        times[timestep-1], squared_error_t1[timestep-1], squared_error_t2[timestep-1] = sess.run([out, mse_t1, mse_t2], 
                                         feed_dict={X: test_set,
                                                    Y: relaxation_times[train_size+val_size:]})

error_t1 = np.sqrt(squared_error_t1)
error_t2 = np.sqrt(squared_error_t2)

#    W1 = weights['h1'].eval(sess)
#    W2 = weights['h2'].eval(sess)
#    Wout = weights['out'].eval(sess)
#    
#    b1 = biases['b1'].eval(sess)
#    b2 = biases['b2'].eval(sess)
#    bout = biases['out'].eval(sess)
    
