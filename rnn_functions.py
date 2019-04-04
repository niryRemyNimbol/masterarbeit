#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:58:29 2018

@author: niry
"""
import tensorflow as tf
import dic
import os

def RNN(x, timesteps, num_hidden, num_output):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, 
                                  reuse=tf.AUTO_REUSE)

    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.layers.dense(outputs[-1], num_output, activation=tf.sigmoid, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
#sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    
def RNN_with_fc(x, num_input, timesteps, num_hidden, num_output):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    
    # Applying fully connected layer
    x = [tf.layers.dense(x_in, num_input, activation=tf.tanh, kernel_regularizer=tf.norm, name='fc{}'.format(x.index(x_in)), reuse=tf.AUTO_REUSE) for x_in in x]
#    seq_length = tf.constant([x_in.shape[1].value for x_in in x])

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, 
                                  reuse=tf.AUTO_REUSE)

    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
#    outputs, states = tf.nn.static_rnn(lstm_cell, x, sequence_length=seq_length, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.layers.dense(outputs[-1], num_output, activation=tf.sigmoid, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
#    return tf.layers.dense(outputs[-1], num_output, activation=tf.nn.relu, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
#sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    
def RNN_MAPE(x, num_input, timesteps, num_hidden, num_output):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    
    # Applying fully connected layer
    x = [tf.layers.dense(x_in, num_input, activation=tf.tanh, kernel_regularizer=tf.norm, name='fc{}'.format(x.index(x_in)), reuse=tf.AUTO_REUSE) for x_in in x]
    #    seq_length = tf.constant([x_in.shape[1].value for x_in in x])
    
    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, 
                                  reuse=tf.AUTO_REUSE)
    
    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #    outputs, states = tf.nn.static_rnn(lstm_cell, x, sequence_length=seq_length, dtype=tf.float32)
    
    # Linear activation, using rnn inner loop last output
    #    return tf.layers.dense(outputs[-1], num_output, activation=tf.sigmoid, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
    return tf.layers.dense(outputs[-1], num_output, activation=tf.nn.relu, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
    #sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    
def RNN_with_tr(x, num_input, timesteps, num_hidden, num_output):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    
    # Applying fully connected layer
    x = [tf.layers.dense(x_in, num_input, activation=tf.tanh, kernel_regularizer=tf.norm, name='fc{}'.format(x.index(x_in)), reuse=tf.AUTO_REUSE) for x_in in x]
#    seq_length = tf.constant([x_in.shape[1].value for x_in in x])

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, 
                                  reuse=tf.AUTO_REUSE)

    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
#    outputs, states = tf.nn.static_rnn(lstm_cell, x, sequence_length=seq_length, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return [tf.layers.dense(output, num_output, activation=tf.sigmoid, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE) for output in outputs]
#    return tf.layers.dense(outputs[-1], num_output, activation=tf.nn.relu, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
#sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])

def LSTM(x, timesteps, num_hidden, num_out, activation=tf.sigmoid, fc=False, tr=False, num_input=64):

    x = tf.unstack(x, timesteps, 1)

    if fc:
        x = [tf.layers.dense(x_in, num_input, activation=tf.tanh, kernel_regularizer=tf.norm, name='fc{}'.format(x.index(x_in)), reuse=tf.AUTO_REUSE) for x_in in x]

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, reuse=tf.AUTO_REUSE)

    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    if tr:
        return [tf.layers.dense(output, num_out, activation=activation, kernel_regularizer=tf.norm, reuse=tf.AUTO_REUSE) for output in outputs]
    else:
        return tf.layers.dense(outputs[-1], num_out, activation=activation, kernel_regularizer=tf.norm, reuse=tf.AUTO_REUSE)

def train_lstm_batch(X, Y, session, train_op, loss_op, batch_data, batch_target):
    train, loss = session.run([train_op, loss_op], feed_dict={X: batch_data, Y:batch_target})
    return loss

def train_lstm(X, Y, net, epochs, batch_size, save_step, loss_function, learning_rate, data, target, val_data, val_target, tr=False):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if tr:
        loss_op = [loss_function(Y, net_) for net_ in net]
        train_op = [optimizer.minimize(op) for op in loss_op]
        val_loss_summary = [tf.summary.scalar('validation_loss', op) for op in loss_op]
    else:
        loss_op = loss_function(Y, net)
        train_op = optimizer.minimize(loss_op)
        val_loss_summary = tf.summary.scalar('validation_loss', loss_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(init)
        total_loss = 0
        data_batches, target_batches = dic.build_batches(data, target, batch_size)
        num_batches = len(data_batches)

        if tr:
            val_loss_writer = [tf.summary.FileWriter('tensorboard/validation_loss_cell{}'.format(n+1), session.graph) for n in range(len(loss_op))]
        else:
            val_loss_writer = tf.summary.FileWriter('tensorboard/validation_loss', session.graph)

        for epoch in range(1, epochs + 1):
            for k in range(num_batches):
                batch_data = data_batches[k]
                batch_target = target_batches[k]

                batch_loss = train_lstm_batch(X, Y, session, train_op, loss_op, batch_data, batch_target)
                if tr:
                    total_loss += batch_loss[-1]
                else:
                    total_loss += batch_loss

            total_loss /= num_batches

            val_loss, val_loss_summ = session.run([loss_op, val_loss_summary], feed_dict={X:val_data, Y:val_target})
            if tr:
                val_loss = val_loss[-1]
                for n in range(len(val_loss_writer)):
                    val_loss_writer[n].add_summary(val_loss_summ, epoch)
            else:
                val_loss_writer.add_summary(val_loss_summ, epoch)

            data, target = dic.shuffle(data, target)
            data_batches, target_batches = dic.build_batches(data, target, batch_size)

            if epoch == 1:
                best_loss, counter = save_lstm(saver, session, val_loss, epoch)
                display_loss(total_loss, val_loss, epoch)
            elif epoch % save_step == 0:
                best_loss, counter = save_lstm(saver, session, val_loss, epoch, best_loss=best_loss, counter=counter)
                display_loss(total_loss, val_loss, epoch)
    print('Optimisation finished! Best validation loss: {}, epoch: {}'.format(best_loss, epoch - 10 * (counter + 1)))

def save_lstm(saver, session, val_loss, epoch, best_loss=None, counter=0):
    ckpt_dir = 'rnn_model/'
    ckpt_file = ckpt_dir + 'lstm_model_checkpoint{}.ckpt'.format(epoch)
    if epoch == 1:
        saver.save(session, ckpt_file)
        best_loss = val_loss
    elif val_loss < best_loss:
        saver.save(session, ckpt_file)
        prev_save = epoch - 10 * (counter + 1)
        prev_save += epoch==1
        prev_ckpt = ckpt_dir + 'lstm_model_checkpoint{}.ckpt'.format(prev_save)
        os.remove(prev_ckpt + '.index')
        os.remove(prev_ckpt + '.meta')
        os.remove(prev_ckpt + '.data-00000-of-00001')
        counter = 0
        best_loss = val_loss
    else:
        counter += 1

    return best_loss, counter

def display_loss(total_loss, val_loss, epoch):
    print("Epoch " + str(epoch) + ", Training Loss= " + "{:.10f}".format(total_loss))
    print("Epoch " + str(epoch) + ", Validation Loss= " + "{:.10f}".format(val_loss))


