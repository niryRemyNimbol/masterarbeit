#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:58:29 2018

@author: niry
"""
import tensorflow as tf

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
    x = [tf.layers.dense(x_in, num_input, kernel_regularizer=tf.norm, name='fc{}'.format(x.index(x_in)), reuse=tf.AUTO_REUSE) for x_in in x]

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.tanh, 
                                  reuse=tf.AUTO_REUSE)

    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.layers.dense(outputs[-1], num_output, activation=tf.sigmoid, kernel_regularizer=tf.norm, name='out', reuse=tf.AUTO_REUSE)
#sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])