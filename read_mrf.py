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
mask_path = '../recon_q_examples/data/Exam52006/Series5/mask.dat'
map_path = '../recon_q_examples/data/Exam52006/Series5/qmaps.dat'
dl_path = '../recon_q_examples/data/Exam52006/Series5/dl_qmaps.dat'
#data_path = '../recon_q_examples/data/recon_data'
mrf = read_mrf_data(data_path, 1000, 256)
series = mrf.reshape((1000, 256**2))
series /= np.linalg.norm(series, axis=0)
mask_id = open(mask_path, 'rb')
map_id = open(map_path, 'rb')
dl_id = open(dl_path, 'rb')
mask = np.reshape(np.fromfile(mask_id, np.float32), [256,256])
map = np.reshape(np.fromfile(map_id, np.float32), [256,256,2],order='F')
dl_map = np.reshape(np.fromfile(dl_id, np.float32), [256,256,2],order='F')
#series /= np.amax(series, axis=0)
#series = series.T
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
ckpt_dir = '../rnn_model_tr/'

with tf.Session() as sess:
#    ckpt_file = ckpt_dir + 'model_fc_len1000_checkpoint855.ckpt'
#    ckpt_file = ckpt_dir + 'model_var_tr_norm_10_checkpoint300.ckpt'
    ckpt_file = ckpt_dir + 'model_tr_checkpoint940.ckpt'
#    ckpt_file = ckpt_dir + 'model_var_tr_norm_checkpoint1000.ckpt'
    saver.restore(sess, ckpt_file)

    times = sess.run(out, feed_dict={X: series.T.reshape((series.shape[1], timesteps, num_in_fc))})

imgs = [time.reshape((256,256,2), order='C') for time in times]
#imgs = times.reshape((256,256,2), order='C')

#fig, axs = plt.subplots(2, 10, figsize=(50,10))
#for k in range(len(imgs)):
#    t1 = axs[0, k].imshow(imgs[k][:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
#    fig.colorbar(t1, ax=axs[0, k])
#    axs[0, k].set_title('T1, timestep {} (ms)'.format(k+1), weight='bold')
#    t2 = axs[1, k].imshow(imgs[k][:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
#    fig.colorbar(t2, ax=axs[1, k])
#    axs[1, k].set_title('T2, timestep {} (ms)'.format(k+1), weight='bold')
#fig, axs = plt.subplots(1, 2, figsize=(40, 20))
#t1 = axs[0].imshow(imgs[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
#fig.colorbar(t1, ax=axs[0])
#t2 = axs[1].imshow(imgs[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
#fig.colorbar(t2, ax=axs[1])
plt.rc('text', usetex=True)

label = 1
aprev = 10
true_t1 = np.array([604, 596,1448, 1262, 444, 754, 903, 1276, 1034, 745, 1160, 966])
#true_t1_std = 0.03 * true_t1
true_t2 = np.array([95, 136, 390, 184, 154, 116, 137, 204, 167, 157, 214, 224])
#true_t2_std = 0.03 * true_t2
#data_t1 = []
#data_t1_std = []
#data_t2 = []
#data_t2_std = []
img_gt = np.zeros_like(map)
angles = [(k,l) for k in range(0,221,2) for l in range(0,221,2)]
for k, l in angles:
    if mask[k:k + 36, l].sum() == 0 and mask[k:k + 36, l + 35].sum() == 0 and mask[k, l:l + 36].sum() == 0 and mask[k + 35, l:l + 36].sum() == 0 and mask[k:k + 36, l:l + 36].sum() > 0:
        a = np.mean(mask[k:k + 36, l:l + 36] * imgs[9][k:k + 36, l:l + 36, 1], axis=(0, 1))
        if np.abs(a - aprev) > 1e-4:
            #            axs[m, n].imshow(mask[k:k + 36, l:l + 36] * imgs[k:k + 36, l:l + 36, 0], cmap='hot', origin='lower', vmin=0,
            #                             vmax=3.0)
#            y_t1_std = []
            y_t1_mean = []
#            y_t2_std = []
            y_t2_mean = []
            for n in range(len(imgs)):
                #                axs[0, n].annotate('{}'.format(label), (l, k), color='y')
                #                axs[0, n].vlines([l, l+36], k, k+36, colors='y', label='{}'.format(label))
                #                axs[0, n].hlines([k, k+36], l, l+36, colors='y', label='{}'.format(label))
                #                axs[2, n].annotate('{}'.format(label), (l, k), color='r')
                #                axs[2, n].vlines([l, l+36], k, k+36, colors='r', label='{}'.format(label))
                #                axs[2, n].hlines([k, k+36], l, l+36, colors='r', label='{}'.format(label))
                tube_t1 = mask[k:k+36, l:l+36] * imgs[n][k:k+36, l:l+36, 0] * 1e3
                tube_t2 = mask[k:k+36, l:l+36] * imgs[n][k:k+36, l:l+36, 1] * 1e3
#                y_t1_std.append(np.std(tube_t1[tube_t1 > 0].flatten()))
                y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
#                y_t2_std.append(np.std(tube_t2[tube_t2 > 0].flatten()))
                y_t2_mean.append(np.mean(tube_t2[tube_t2 > 0].flatten()))
            #                print(np.mean(tube_t2[tube_t2 > 0].flatten()), n, true_t2[label-1])
            #            axs[0, 10].annotate('{}'.format(label), (l, k), color='y')
            #            axs[0, 10].vlines([l, l+36], k, k+36, colors='y', label='{}'.format(label))
            #            axs[0, 10].hlines([k, k+36], l, l+36, colors='y', label='{}'.format(label))
            #            axs[2, 10].annotate('{}'.format(label), (l, k), color='r')
            #            axs[2, 10].vlines([l, l+36], k, k+36, colors='r', label='{}'.format(label))
            #            axs[2, 10].hlines([k, k+36], l, l+36, colors='r', label='{}'.format(label))
            tube_t1 = map[-k:-k-36:-1, l:l+36, 0] * 1e3
            tube_t2 = map[-k:-k-36:-1, l:l+36, 1] * 1e3
#            y_t1_std.append(np.std(tube_t1[tube_t1 > 0].flatten()))
            y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
#            y_t2_std.append(np.std(tube_t2[tube_t2 > 0].flatten()))
            y_t2_mean.append(np.mean(tube_t2[tube_t2 > 0].flatten()))
            ind = np.where(tube_t1 > 0)
            offset = [k * np.ones_like(ind[0]), l * np.ones_like(ind[1])]
            img_gt[ind[0] + offset[0], ind[1] + offset[1], :] = np.array([true_t1[label-1], true_t2[label-1]])
            #            axs[0, 11].annotate('{}'.format(label), (l, k), color='y')
            #            axs[0, 11].vlines([l, l+36], k, k+36, colors='y', label='{}'.format(label))
            #            axs[0, 11].hlines([k, k+36], l, l+36, colors='y', label='{}'.format(label))
            #            axs[2, 11].annotate('{}'.format(label), (l, k), color='r')
            #            axs[2, 11].vlines([l, l+36], k, k+36, colors='r', label='{}'.format(label))
            #            axs[2, 11].hlines([k, k+36], l, l+36, colors='r', label='{}'.format(label))
            tube_t1 = mask[k:k+36, l:l+36] * dl_map[:, :, 0].T[k:k+36, l:l+36] * 1e3
            tube_t2 = mask[k:k+36, l:l+36] * dl_map[:, :, 1].T[k:k+36, l:l+36] * 1e3
#            y_t1_std.append(np.std(tube_t1[tube_t1 > 0].flatten()))
            y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
#            y_t2_std.append(np.std(tube_t2[tube_t2 > 0].flatten()))
            y_t2_mean.append(np.mean(tube_t2[tube_t2 > 0].flatten()))
            aprev = a
            #            data_t1.append(y_t1_mean)
            #            data_t1_std.append(y_t1_std)
            #            data_t2.append(y_t2_mean)
            #            data_t2_std.append(y_t2_std)
            label += 1
#data_t1 = np.array(data_t1)
#data_t2 = np.array(data_t2)
#data_t1_std = np.array(data_t1_std)
#data_t2_std = np.array(data_t2_std)

def plot_brain_results_len(length):
    figlen, axlen = plt.subplots(2, 3, figsize=(15, 10))
    t1_len = axlen[0, 0].imshow(mask * imgs[length][:, :, 0] * 1e3, cmap='hot', origin='lower', vmin=0, vmax=3000)
#    t1_err = axlen[0, 1].imshow(
#        (np.abs(mask * imgs[length][:, :, 0] * 1e3 - img_gt[:, :, 0]) / (img_gt[:, :, 0] + 1e-6)) * 1e2,
#        cmap='plasma', origin='lower', vmin=0, vmax=100)
    t1_err = axlen[0, 1].imshow(
        (np.abs(mask * imgs[length][:, :, 0] - map[-1:-257:-1, :, 0]) / (map[-1:-257:-1, :, 0] + 1e-6)) * 1e2,
        cmap='plasma', origin='lower', vmin=0, vmax=100)
#    scdm = axlen[0, 2].scatter(img_gt[:, :, 0], mask[:, :] * imgs[length][:, :, 0] * 1e3, c='b', marker='.', alpha=0.1)
#    scnet = axlen[0, 2].scatter(img_gt[:, :, 0], mask[:, :] * dl_map[:, :, 0].T * 1e3, c='r', marker='.', alpha=0.1)
    scdm = axlen[0, 2].scatter(map[-1: -257:-1, :, 0] * 1e3, mask * imgs[length][:, :, 0] * 1e3, c='r', marker='.',
                               alpha=0.1)
    scnet = axlen[0, 2].scatter(mask * dl_map[:, :, 0].T * 1e3, mask * imgs[length][:, :, 0] * 1e3, c='b', marker='.',
                                alpha=0.1)
    axlen[0, 2].plot([x for x in range(4000)], [x for x in range(4000)], 'g--')
    axlen[0, 0].set_axis_off()
    axlen[0, 1].set_axis_off()
    figlen.colorbar(t1_len, ax=axlen[0, 0])
    figlen.colorbar(t1_err, ax=axlen[0, 1])
    axlen[0, 0].set_title(r'\textbf{T1 (ms), time step \#}' + '{}'.format(length + 1))
    axlen[0, 1].set_title(r'\textbf{T1 percentage error, time step \#}' + '{}'.format(length + 1))
    axlen[0, 2].set_title(r'\textbf{T1 scatter plot, time step \#}' + '{}'.format(length + 1))
    axlen[0, 2].set_xlabel(r'Dictionary matching / MRF net (ms)')
    axlen[0, 2].set_ylabel(r'LSTM / MRF net predictions (ms)')
    axlen[0, 2].legend((scdm, scnet), (r'DM', r'MRF net'))
    axlen[0, 2].legend((scdm, scnet), (r'LSTM', r'MRF net'))
    axlen[0, 2].set_xbound(lower=0, upper=4000)
    axlen[0, 2].set_ybound(lower=0, upper=4000)
    t2_len = axlen[1, 0].imshow(mask * imgs[length][:, :, 1] * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
#    t2_err = axlen[1, 1].imshow(
#        (np.abs(mask * imgs[length][:, :, 1] * 1e3 - img_gt[:, :, 1]) / (img_gt[:, :, 1] + 1e-6)) * 1e2,
#        cmap='viridis', origin='lower', vmin=0, vmax=100)
    t2_err = axlen[1, 1].imshow(
        (np.abs(mask * imgs[length][:, :, 1] - map[-1:-257:-1, :, 1]) / (map[-1:-257:-1, :, 1] + 1e-6)) * 1e2,
        cmap='viridis', origin='lower', vmin=0, vmax=100)
#    scdm = axlen[1, 2].scatter(img_gt[:, :, 1], mask[:, :] * imgs[length][:, :, 1] * 1e3, c='b', marker='.', alpha=0.1)
#    scnet = axlen[1, 2].scatter(img_gt[:, :, 1], mask[:, :] * dl_map[:, :, 1].T * 1e3, c='r', marker='.', alpha=0.1)
    scdm = axlen[1, 2].scatter(map[-1: -257:-1, :, 1] * 1e3, mask * imgs[length][:, :, 1] * 1e3, c='r', marker='.',
                               alpha=0.1)
    scnet = axlen[1, 2].scatter(mask * dl_map[:, :, 1].T * 1e3, mask * imgs[length][:, :, 1] * 1e3, c='b', marker='.',
                                alpha=0.1)
    axlen[1, 2].plot([x for x in range(600)], [x for x in range(600)], 'g--')
    axlen[1, 0].set_axis_off()
    axlen[1, 1].set_axis_off()
    figlen.colorbar(t2_len, ax=axlen[1, 0])
    figlen.colorbar(t2_err, ax=axlen[1, 1])
    axlen[1, 0].set_title(r'\textbf{T2 (ms), time step \#}' + '{}'.format(length + 1))
    axlen[1, 1].set_title(r'\textbf{T2 percentage error, time step \#}' + '{}'.format(length + 1))
    axlen[1, 2].set_title(r'\textbf{T2 scatter plot, time step \#}' + '{}'.format(length + 1))
    axlen[1, 2].set_xlabel(r'Dictionary matching / MRF net (ms)')
    axlen[1, 2].set_ylabel(r'LSTM predictions (ms)')
    axlen[1, 2].legend((scdm, scnet), (r'DM', r'MRF net'))
#    axlen[1, 2].legend((scdm, scnet), (r'LSTM', r'MRF net'))
    axlen[1, 2].set_xbound(lower=0, upper=600)
    axlen[1, 2].set_ybound(lower=0, upper=600)
    figlen.show()
    return figlen

#fig, axs = plt.subplots(12, 4, figsize=(14, 42))

figs = []
for k in range(len(imgs)):
    figs.append(plot_brain_results_len(k))
#    t1_pred = axs[k, 0].imshow(mask * imgs[k][:, :, 0] * 1e3, cmap='hot', origin='lower', vmin=0, vmax=3000)
#    axs[k, 0].set_title('{}'.format(k+1)+r'\textbf{ time step LSTM, T1 (ms)}', weight='bold')
#    axs[k, 0].set_axis_off()
#    fig.colorbar(t1_pred, ax=axs[k, 0])
#    t2_pred = axs[k, 2].imshow(mask * imgs[k][:, :, 1] * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
#    axs[k, 2].set_title('{}'.format(k+1)+r'\textbf{ time step LSTM, T2 (ms)}', weight='bold')
#    axs[k, 2].set_axis_off()
#    fig.colorbar(t2_pred, ax=axs[k, 2])
#    scdm = axs[k, 1].scatter(img_gt[:, :, 0], mask[:, :] * imgs[k][:, :, 0] * 1e3, c='b', marker='.', alpha=0.1)
#    scnet = axs[k, 1].scatter(img_gt[:, :, 0], mask[:, :] * dl_map[:, :, 0].T * 1e3, c='r', marker='.', alpha=0.1)
#    scdm = axs[k, 1].scatter(map[-1:-257:-1, :, 0] * 1e3 , mask[:, :] * imgs[k][:, :, 0] * 1e3, c='b', marker='.', alpha=0.1)
#    scnet = axs[k, 1].scatter(mask[:, :] * dl_map[:, :, 0].T * 1e3, mask[:, :] * imgs[k][:, :, 0] * 1e3, c='r', marker='.', alpha=0.1)
#    axs[k, 1].plot([x for x in range(4000)], [x for x in range(4000)], '--g')
#  axs[k, 1].plot(true_t1, data_t1[:, k], '*y')
#    axs[k, 1].set_title('{}'.format(k+1)+r'\textbf{ time step LSTM, T1 scatter plot}', weight='bold')
#    axs[k, 1].set_xlabel(r'Dictionary matching / MRF net (ms)')
#    axs[k, 1].set_xbound(lower=0, upper=4000)
#    axs[k, 1].set_ylabel(r'Prediction (ms)')
#    axs[k, 1].set_ybound(lower=0, upper=4000)
#    axs[k, 1].legend((scdm, scnet), (r'LSTM', r'MRF-net'), loc=4)
#    scdm = axs[k, 3].scatter(img_gt[:, :, 1], mask[:, :] * imgs[k][:, :, 1] * 1e3, c='b', marker='.', alpha=0.1)
#    scnet = axs[k, 3].scatter(img_gt[:, :, 1], mask[:, :] * dl_map[:, :, 1].T * 1e3, c='r', marker='.', alpha=0.1)
#    scdm = axs[k, 3].scatter(map[-1:-257:-1, :, 1] * 1e3 , mask[:, :] * imgs[k][:, :, 1] * 1e3, c='b', marker='.', alpha=0.1)
#    scnet = axs[k, 3].scatter(mask[:, :] * dl_map[:, :, 1].T * 1e3, mask[:, :] * imgs[k][:, :, 1] * 1e3, c='r', marker='.', alpha=0.1)
#    axs[k, 3].plot([x for x in range(600)], [x for x in range(600)], '--g')
#  axs[k, 3].plot(true_t2, data_t2[:, k], '*y')
#    axs[k, 3].set_title('{}'.format(k+1)+r'\textbf{ time step LSTM, T2 scatter plot}', weight='bold')
#    axs[k, 3].set_xlabel(r'Dictionary matching / MRF net (ms)')
#    axs[k, 3].set_xbound(lower=0, upper=600)
#    axs[k, 3].set_ylabel(r'Prediction (ms)')
#    axs[k, 3].set_ybound(lower=0, upper=600)
#    axs[k, 3].legend((scdm, scnet), (r'DM', r'MRF-net'), loc=4)
fig_comp, ax_comp = plt.subplots(1, 4, figsize=(20, 5))
t1_DM = ax_comp[0].imshow(map[-1:-257:-1, :, 0] * 1e3, cmap='hot', origin='lower', vmin=0, vmax=3000)
ax_comp[0].set_title(r'\textbf{Dictionary matching, T1 (ms)}', weight='bold')
ax_comp[0].set_axis_off()
fig_comp.colorbar(t1_DM, ax=ax_comp[0])
t2_DM = ax_comp[1].imshow(map[-1:-257:-1, :, 1] * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
ax_comp[1].set_title(r'\textbf{Dictionary matching, T2 (ms)}', weight='bold')
ax_comp[1].set_axis_off()
fig_comp.colorbar(t2_DM, ax=ax_comp[1])
t1_DL = ax_comp[2].imshow(mask * dl_map[:, :, 0].T * 1e3, cmap='hot', origin='lower', vmin=0, vmax=3000)
ax_comp[2].set_title(r'\textbf{MRF net, T1 (ms)}', weight='bold')
ax_comp[2].set_axis_off()
fig_comp.colorbar(t1_DL, ax=ax_comp[2])
t2_DL = ax_comp[3].imshow(mask * dl_map[:, :, 1].T * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
ax_comp[3].set_title(r'\textbf{MRF net, T2 (ms)}', weight='bold')
ax_comp[3].set_axis_off()
fig_comp.colorbar(t2_DL, ax=ax_comp[3])
#fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#t1 = axs[0].imshow(mask*imgs[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
#fig.colorbar(t1, ax=axs[0])
#t2 = axs[1].imshow(mask*imgs[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
#fig.colorbar(t2, ax=axs[1])
#axs[10, 1].set_axis_off()
#axs[11, 1].set_axis_off()
#axs[10, 3].set_axis_off()
#axs[11, 3].set_axis_off()
fig_comp.show()
