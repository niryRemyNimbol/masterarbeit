# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
import numpy as np
import dic

# Parameters
learning_rate = 5.0e-3
training_steps = 10000
train_size = 1800
batch_size = 400
val_size = 400
test_size = 200
batch_size = 600

# Network Parameters
n_hidden_1 = 600
n_hidden_2 = 600
#n_hidden_3 = 200 
num_input = 1000 
num_output = 2

# Time series and corresponding T1 and T2
dictionary = dic.dic('recon_q_examples/dict/', 'fisp_mrf_bis', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :] >= dictionary.lut[1, :]] 
D /= np.linalg.norm(D, axis=0)
permutation = np.random.permutation(D.shape[1])

series_real = np.real(D.T[permutation])
series_imag = np.imag(D.T[permutation])
series_mag = np.abs(D.T[permutation] + np.random.normal(0, 0.01*np.mean(np.max(np.real(D))), D.T.shape) + 1j * np.random.normal(0, 0.01*np.mean(np.max(np.imag(D))), D.T.shape))
#series_mag = np.abs(D.T[permutation])
series_phase = np.angle(D.T[permutation])
series = np.concatenate([series_mag.T, series_phase.T])
series = series.T

train_set =  series_mag[:train_size]
val_set = series_mag[train_size:train_size+val_size]
test_set = series_mag[train_size+val_size:train_size+val_size+test_size]

relaxation_times = np.float64(dictionary.lut[0:2].T[permutation])
times_max = np.max(relaxation_times, axis=0)
relaxation_times /= times_max

train_times = relaxation_times[:train_size]
val_times = relaxation_times[train_size:train_size+val_size]
test_times = relaxation_times[train_size+val_size:train_size+val_size+test_size]

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'series': train_set}, y=train_times, 
        batch_size=batch_size, num_epochs=None, shuffle=False)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['series']
    # Hidden fully connected layer with 100 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.tanh, 
#                              kernel_initializer=tf.random_uniform_initializer, 
                              kernel_regularizer=tf.norm)
    # Hidden fully connected layer with 100 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, 
#                              kernel_initializer=tf.random_uniform_initializer, 
                              activation=tf.tanh, kernel_regularizer=tf.norm)
    # Hidden fully connected layer with 100 neurons
    #layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation=tf.tanh)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_output, 
#                                kernel_initializer=tf.random_uniform_initializer, 
                                activation=tf.sigmoid, kernel_regularizer=tf.norm)
    return out_layer

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    
    # Build the neural network
    logits = neural_net(features)
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=times_max * logits) 
        
    # Define loss and optimizer
    loss_op = tf.losses.mean_squared_error(labels, logits)
#    loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(labels, logits), labels))) # mean averaged percentage error
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # Evaluate the accuracy of the model
    acc_op =  tf.metrics.mean_squared_error(labels=times_max*labels, predictions=times_max*logits)
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=logits,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'mse': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Train the Model
model.train(input_fn, steps=training_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'series': val_set}, y=val_times,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
model.evaluate(input_fn)

# Get series from test set
#test_series = series_mag[600:640]
#test_images = np.zeros_like(series[601:641])
# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'series': test_set}, batch_size=batch_size, shuffle=False)
# Use the model to predict the images class
times = list(model.predict(input_fn))

error = 0
square_error = 0
for i in range(len(times)):
    error += np.abs(times[i]-relaxation_times[600+i]*times_max)
    square_error += (times[i]-relaxation_times[600+i]*times_max)**2
error /= len(times)
square_error /= len(times)
rmserror = np.sqrt(square_error)


