# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
import numpy as np
import dic

# Parameters
learning_rate = 0.006
num_steps = 5000
batch_size = 250
display_step = 1000

# Network Parameters
n_hidden_1 = 300
n_hidden_2 = 300 
num_input = 520
num_output = 2

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_output]))
}

# Time series and corresponding T1 and T2
dictionary = dic.dic('recon_q_examples/dict/', 'qti', 260, 10)
permutation = np.random.permutation(640)

series = np.concatenate([np.real(dictionary.D.T[permutation].T), 
                                 np.imag(dictionary.D.T[permutation].T)])
series = series.T
relaxation_times = dictionary.lut[0:2].T[permutation]
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

train_series = series[0:501]
train_times = relaxation_times[0:501]
validation_series = series[501:601]
validation_times = relaxation_times[501:601]
test_series = series[601:641]
test_times = relaxation_times[601:641]


# Create model
def neural_net(x):
    # Hidden ful;tanly connected layer with 256 neurons
    layer_1 = tf.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.sigmoid(tf.matmul(layer_2, weights['out']) + biases['out'])
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(Y, logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
acc_op = tf.losses.mean_squared_error(labels=Y, predictions=logits)
out = times_max * logits

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(0, num_steps):
        train_batch_x = train_series[(step * batch_size)%640: ((step + 1) * batch_size + 1)%640]
        train_batch_y = train_times[(step * batch_size)%640: ((step + 1) * batch_size + 1)%640]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: train_batch_x, Y: train_batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss
            loss = sess.run(loss_op, feed_dict={X: train_batch_x,
                                                                 Y: train_batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))

    print("Optimization Finished!")

    # Calculate MSE for test time series
    W1 = weights['h1'].eval(sess)
    W2 = weights['h2'].eval(sess)
    Wout = weights['out'].eval(sess)
    
    b1 = biases['b1'].eval(sess)
    b2 = biases['b2'].eval(sess)
    bout = biases['out'].eval(sess)
    
    print("Testing Trained Model:", \
        sess.run([out, acc_op], feed_dict={X: test_series,
                                      Y: test_times}))