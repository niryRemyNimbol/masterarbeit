# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:35:53 2018

@author: andriama
"""

import numpy as np
import tensorflow as tf
from rnn_functions import RNN_with_fc
from rnn_functions import  RNN_with_tr
import matplotlib.pyplot as plt



def read_mrf_data(data_path, Nreps, dim):
    # dictionary data file pat
    data_mag_path = data_path + '_mag.dat'

    # open the data files
    data_id = open(data_mag_path, 'rb')

    data_mag = np.reshape(np.fromfile(data_id, np.float32), [Nreps, dim, dim])

    return data_mag

data_path = '../recon_q_examples/data/Exam52006/Series5/recon_data'
#data_path = '../recon_q_examples/data/recon_data'
mrf = read_mrf_data(data_path, 1000, 256)
series = mrf.reshape((1000, 256**2)).T
series /= np.linalg.norm(series, axis=0)
#series /= np.amax(series, axis=0)
series = series.T
times_max = np.array([4., .6])

# Network Parameters
num_input = 64
timesteps = 10 # timesteps
num_hidden = 8
num_output = 2 # number of output parameters

# Fully Connected Layer Parameters
num_in_fc = 100

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_in_fc])

#logits = RNN_with_fc(X, num_input, timesteps, num_hidden, num_output)
logits = RNN_with_tr(X, num_input, timesteps, num_hidden, num_output)

#out = times_max * logits
out = [times_max*logit for logit in logits]

# Saver
saver = tf.train.Saver()

# Restoration directory
ckpt_dir = '../rnn_model/'

with tf.Session() as sess:
#    ckpt_file = ckpt_dir + 'model_fc_len1000_checkpoint855.ckpt'
#    ckpt_file = ckpt_dir + 'model_var_tr_norm_10_checkpoint300.ckpt'
    ckpt_file = ckpt_dir + 'model_var_tr_norm_checkpoint975.ckpt'
#    ckpt_file = ckpt_dir + 'model_var_tr_norm_checkpoint1000.ckpt'
    saver.restore(sess, ckpt_file)

    times = sess.run(out, feed_dict={X: series.T.reshape((series.shape[1], timesteps, num_in_fc), order='F')})

imgs = [time.reshape((256,256,2), order='C') for time in times]
#imgs = times.reshape((256,256,2), order='C')

fig, axs = plt.subplots(2, 10, figsize=(50,10))
for k in range(len(imgs)):
    t1 = axs[0, k].imshow(imgs[k][:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
    fig.colorbar(t1, ax=axs[0, k])
    axs[0, k].set_title('T1, timestep {}'.format(k+1), weight='bold')
    t2 = axs[1, k].imshow(imgs[k][:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
    fig.colorbar(t2, ax=axs[1, k])
#fig, axs = plt.subplots(1, 2, figsize=(40, 20))
#t1 = axs[0].imshow(imgs[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
#fig.colorbar(t1, ax=axs[0])
#t2 = axs[1].imshow(imgs[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
#fig.colorbar(t2, ax=axs[1])
fig.show()
