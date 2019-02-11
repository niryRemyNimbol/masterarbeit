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

data_path = '../recon_q_examples/data/Exam52004/Series5/recon_data'
mask_path = '../recon_q_examples/data/Exam52004/Series5/mask.dat'
map_path = '../recon_q_examples/data/Exam52004/Series5/qmaps.dat'
dl_path = '../recon_q_examples/data/Exam52004/Series5/dl_qmaps.dat'
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
#series = series[0:400, :]
times_max = np.array([4., .6])

# Network Parameters
num_input = 64
timesteps = 10 # timesteps
num_hidden = 8
num_output = 2 # number of output parameters

# Fully Connected Layer Parameters
num_in_fc = 100

times=[]
for timestep in range(1, timesteps+1):
    with tf.variable_scope('len{}'.format(timestep)):
        X = tf.placeholder("float", [None, timestep, num_in_fc])
        logits = RNN_with_fc(X, num_input, timestep, num_hidden, num_output)

    # Define loss and optimizer
#    loss_op = tf.losses.mean_squared_error(Y, logits)
    #loss_op = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Y, logits), Y))) # mean averaged percentage error
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
#    mse_t1 = tf.losses.mean_squared_error(labels=times_max[0]*Y[:, 0], predictions=times_max[0]*logits[:, 0])
#    mse_t2 = tf.losses.mean_squared_error(labels=times_max[1]*Y[:, 1], predictions=times_max[1]*logits[:, 1])
        out = times_max * logits

    # Initialize the variables (i.e. assign their default value)
#    init = tf.global_variables_initializer()

    # Summaries to view in tensorboard
    #            train_loss_summary = tf.summary.scalar('training_loss', loss_op)
    #    val_loss_summary = tf.summary.scalar('validation_loss', loss_op)
    #merged = tf.summary.merge_all()

    # Saver
        saver = tf.train.Saver()

    # Restoration directory
        ckpt_dir = '../rnn_model_len/rnn_model_len{}/'.format(timestep) #'../rnn_model_len_new/rnn_model_len{}'.format(timestep)
        ckpt_epochs = [990, 940, 1000, 940, 1000, 950, 990, 1000, 950, 980] #[970, 970, 980, 990, 920, 990, 980, 900, 970, 990]

    # Start training
        with tf.Session() as sess:

        # Run the initializer
        #    sess.run(init)
            ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint{}.ckpt'.format(timestep*num_in_fc, ckpt_epochs[timestep-1])
            saver.restore(sess, ckpt_file)

            times.append(sess.run(out, feed_dict={X: series[:timestep*num_in_fc, :].T.reshape((series.shape[1], timestep, num_in_fc))}))

imgs = [time.reshape((256,256,2), order='C') for time in times]
#imgs = times.reshape((256,256,2), order='C')

plt.rc('text', usetex=True)

label = 1
aprev = 10
true_t1 = np.array([604, 596,1448, 1262, 444, 754, 903, 1276, 1034, 745, 1160, 966])
true_t1_std = 0.03 * true_t1
true_t2 = np.array([95, 136, 390, 184, 154, 116, 137, 204, 167, 157, 214, 224])
true_t2_std = 0.03 * true_t2
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
            y_t1_std = []
            y_t1_mean = []
            y_t2_std = []
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
                y_t1_std.append(np.std(tube_t1[tube_t1 > 0].flatten()))
                y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
                y_t2_std.append(np.std(tube_t2[tube_t2 > 0].flatten()))
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
            y_t1_std.append(np.std(tube_t1[tube_t1 > 0].flatten()))
            y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
            y_t2_std.append(np.std(tube_t2[tube_t2 > 0].flatten()))
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
            y_t1_std.append(np.std(tube_t1[tube_t1 > 0].flatten()))
            y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
            y_t2_std.append(np.std(tube_t2[tube_t2 > 0].flatten()))
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
    t1_err = axlen[0, 1].imshow(
        (np.abs(mask * imgs[length][:, :, 0] * 1e3 - img_gt[:, :, 0]) / (img_gt[:, :, 0] + 1e-6)) * 1e2,
        cmap='hot', origin='lower', vmin=0, vmax=100)
#    t1_err = axlen[0, 1].imshow(
#        (np.abs(mask * imgs[length][:, :, 0] - map[-1:-257:-1, :, 0]) / (map[-1:-257:-1, :, 0] + 1e-6)) * 1e2,
#        cmap='hot', origin='lower', vmin=0, vmax=100)
    scdm = axlen[0, 2].scatter(img_gt[:, :, 0], mask[:, :] * imgs[length][:, :, 0] * 1e3, c='b', marker='.', alpha=0.1)
    scnet = axlen[0, 2].scatter(img_gt[:, :, 0], mask[:, :] * dl_map[:, :, 0].T * 1e3, c='r', marker='.', alpha=0.1)
#    scdm = axlen[0, 2].scatter(map[-1: -257:-1, :, 0] * 1e3, mask * imgs[length][:, :, 0] * 1e3, c='r', marker='.',
#                               alpha=0.1)
#    scnet = axlen[0, 2].scatter(mask * dl_map[:, :, 0].T * 1e3, mask * imgs[length][:, :, 0] * 1e3, c='b', marker='.',
#                                alpha=0.1)
    axlen[0, 2].plot([x for x in range(4000)], [x for x in range(4000)], 'g--')
    axlen[0, 0].set_axis_off()
    axlen[0, 1].set_axis_off()
    figlen.colorbar(t1_len, ax=axlen[0, 0])
    figlen.colorbar(t1_err, ax=axlen[0, 1])
    axlen[0, 0].set_title('{}'.format(length + 1) + r'\textbf{ time step LSTM, T1 (ms)}')
    axlen[0, 1].set_title('{}'.format(length + 1) + r'\textbf{ time step LSTM, T1 percentage error}')
    axlen[0, 2].set_title('{}'.format(length + 1) + r'\textbf{ time step LSTM, T1 scatter plot}')
    axlen[0, 2].set_xlabel(r'Dictionary matching / MRF net (ms)')
    axlen[0, 2].set_ylabel(r'LSTM predictions (ms)')
#    axlen[0, 2].legend((scdm, scnet), (r'DM', r'MRF net'))
    axlen[0, 2].legend((scdm, scnet), (r'LSTM', r'MRF net'))
    axlen[0, 2].set_xbound(lower=0, upper=4000)
    axlen[0, 2].set_ybound(lower=0, upper=4000)
    t2_len = axlen[1, 0].imshow(mask * imgs[length][:, :, 1] * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
    t2_err = axlen[1, 1].imshow(
        (np.abs(mask * imgs[length][:, :, 1] * 1e3 - img_gt[:, :, 1]) / (img_gt[:, :, 1] + 1e-6)) * 1e2,
        cmap='copper', origin='lower', vmin=0, vmax=100)
#    t2_err = axlen[1, 1].imshow(
#        (np.abs(mask * imgs[length][:, :, 1] - map[-1:-257:-1, :, 1]) / (map[-1:-257:-1, :, 1] + 1e-6)) * 1e2,
#        cmap='copper', origin='lower', vmin=0, vmax=100)
    scdm = axlen[1, 2].scatter(img_gt[:, :, 1], mask[:, :] * imgs[length][:, :, 1] * 1e3, c='b', marker='.', alpha=0.1)
    scnet = axlen[1, 2].scatter(img_gt[:, :, 1], mask[:, :] * dl_map[:, :, 1].T * 1e3, c='r', marker='.', alpha=0.1)
#    scdm = axlen[1, 2].scatter(map[-1: -257:-1, :, 1] * 1e3, mask * imgs[length][:, :, 1] * 1e3, c='r', marker='.',
#                               alpha=0.1)
#    scnet = axlen[1, 2].scatter(mask * dl_map[:, :, 1].T * 1e3, mask * imgs[length][:, :, 1] * 1e3, c='b', marker='.',
#                                alpha=0.1)
    axlen[1, 2].plot([x for x in range(600)], [x for x in range(600)], 'g--')
    axlen[1, 0].set_axis_off()
    axlen[1, 1].set_axis_off()
    figlen.colorbar(t2_len, ax=axlen[1, 0])
    figlen.colorbar(t2_err, ax=axlen[1, 1])
    axlen[1, 0].set_title('{}'.format(length + 1) + r'\textbf{ time step LSTM, T2 (ms)}')
    axlen[1, 1].set_title('{}'.format(length + 1) + r'\textbf{ time step LSTM, T2 percentage error}')
    axlen[1, 2].set_title('{}'.format(length + 1) + r'\textbf{ time step LSTM, T2 scatter plot}')
    axlen[1, 2].set_xlabel(r'Dictionary matching / MRF net (ms)')
    axlen[1, 2].set_ylabel(r'LSTM predictions (ms)')
#    axlen[1, 2].legend((scdm, scnet), (r'DM', r'MRF net'))
    axlen[1, 2].legend((scdm, scnet), (r'LSTM', r'MRF net'))
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

#x = [x for x in range(1, 13)]

#fig, axs = plt.subplots(3, 4, figsize=(20, 15))
#m = 0
#n = 0

#gt1 = np.repeat(true_t1, data_t1.shape[1]).reshape(data_t1.T.shape)
#gt2 = np.repeat(true_t2, data_t2.shape[1]).reshape(data_t2.T.shape)
#error_t1 = np.abs(data_t1 - gt1) / gt1
#error_t2 = np.abs(data_t2 - gt2) / gt2
#error_t1 = np.sqrt((data_t1 - gt1) ** 2)
#error_t2 = np.sqrt((data_t2 - gt2) ** 2)
#lines = ['Tube {}'.format(k) for k in range(1, data_t1.shape[1]+1, 1)]
#for i in range(axs.shape[1]):
#    if i < 10:
#        cols = ['{}-cell LSTM'.format(i+1), 'Ground truth']
#    elif i == 10:
#        cols = ['DM', 'Ground truth']
#    else:
#        cols = ['MRF-net', 'Ground truth']
#    dt1 = np.concatenate((data_t1[:, i], true_t1), axis=0).reshape((data_t1.shape[0], 2), order='F')
#    dt2 = np.concatenate((data_t2[:, i], true_t2), axis=0).reshape((data_t2.shape[0], 2), order='F')
#    std1 = np.concatenate((data_t1_std[:, 0], true_t1_std), axis=0).reshape((data_t1.shape[0], 2), order='F')
#    std2 = np.concatenate((data_t2_std[:, 0], true_t2_std), axis=0).reshape((data_t2.shape[0], 2), order='F')
#    cell_text_t1 = []
#    for n in range(dt1.shape[0]):
#        cell_text_t1.append(['{} (+/-{})'.format(np.round(dt1[n, l]), np.round(std1[n, l])) for l in range(dt1.shape[1])])
#    cell_text_t2 = []
#    for n in range(dt2.shape[0]):
#        cell_text_t2.append(['{} (+/-{})'.format(np.round(dt2[n, l]), np.round(std2[n, l])) for l in range(dt2.shape[1])])
#    axs[1, i].table(cellText=cell_text_t1, rowLabels=lines, colLabels=cols, loc='center', fontsize=24)
#    axs[1, i].set_axis_off()
#    axs[1, i].set_title('T1 (ms) comparison table ({})'.format(cols[0], cols[1]), weight='bold')
#    axs[3, i].table(cellText=cell_text_t2, rowLabels=lines, colLabels=cols, loc='center', fontsize=24)
#    axs[3, i].set_axis_off()
#    axs[3, i].set_title('T2 (ms) comparison table ({})'.format(cols[0], cols[1]), weight='bold')


#            gt_t1 = true_t1[label-1] * np.ones((12,1))
#            gt_t2 = true_t2[label-1] * np.ones((12,1))
#            print(y_t1.shape, y_t2.shape, y_t1_mean.shape, y_t1_mean.shape)

#            axs[1, label-1].plot(x, gt_t1, '-')
#            axs[1, label-1].plot(x, y_t1, '.b')
#            axs[1, label-1].plot(x, y_t1_mean, '*r')
#            axs[1, label-1].set_title('Tube {}, ground truth T1: {} ms'.format(label, true_t1[label-1]), weight='bold')
#            axs[1, label-1].set_xlabel('Reconstruction method')
#            axs[1, label-1].set_ylabel('T1 prediction (ms)')
#            axs[3, label-1].plot(x, gt_t2, '-')
#            axs[3, label-1].plot(x, y_t2, '.b')
#            axs[3, label-1].plot(x, y_t2_mean, '*r')
#            axs[3, label-1].set_title('Tube {}, ground truth T2: {} ms'.format(label, true_t2[label-1]), weight='bold')
#            axs[3, label-1].set_xlabel('Reconstruction method')
#            axs[3, label-1].set_ylabel('T2 prediction (ms)')

#            if m == 2:
#                if n <= 2:
#                    n += 1
#                    m = 0
#            else:
#                m += 1
#fig.show()

#fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
#axs2[0].plot(np.mean(error_t1[:, 0:10], axis=0), '*')
#axs2[0].plot(np.repeat(np.mean(error_t1[:, 10]), 10), '-')
#axs2[0].plot(np.repeat(np.mean(error_t1[:, 11]), 10), '-')
#axs2[0].set_title('T1 percentage error vs network length', weight='bold')
#axs2[0].legend(('LSTM RNN', 'DM', 'MRF net'))
#axs2[1].plot(np.mean(error_t2[:, 0:10], axis=0), '*')
#axs2[1].plot(np.repeat(np.mean(error_t2[:, 10]), 10), '-')
#axs2[1].plot(np.repeat(np.mean(error_t2[:, 11]), 10), '-')
#axs2[1].set_title('T2 percentage error vs network length', weight='bold')
#axs2[1].legend(('LSTM RNN', 'DM', 'MRF net'))
#fig2.show()

#fig3, axs3 = plt.subplots(5, 4, figsize=(20, 25))

#for k in range(5):
#    t1_pred1 = axs3[k, 0].imshow((np.abs(mask * imgs[k][:, :, 0] - map[-1:-257:-1, :, 0]) / (map[-1:-257:-1, :, 0]+1e-6)) * 1e2, cmap='hot', origin='lower', vmin=0, vmax=100)
#    t1_pred2 = axs3[k, 2].imshow((np.abs(mask * imgs[k+5][:, :, 0] - map[-1:-257:-1, :, 0]) / (map[-1:-257:-1, :, 0]+1e-6)) * 1e2, cmap='hot', origin='lower', vmin=0, vmax=100)
#    axs3[k, 0].set_title('{}'.format(k+1)+r'\textbf{ time step LSTM, T1 percentage error}', weight='bold')
#    axs3[k, 2].set_title('{}'.format(k+6)+r'\textbf{ time step LSTM, T1 percentage error}', weight='bold')
#    axs3[k, 0].set_axis_off()
#    axs3[k, 2].set_axis_off()
#    fig3.colorbar(t1_pred1, ax=axs3[k, 0])
#    fig3.colorbar(t1_pred2, ax=axs3[k, 2])
#    t2_pred1 = axs3[k, 1].imshow((np.abs(mask * imgs[k][:, :, 1] - map[-1:-257:-1, :, 1]) / (map[-1:-257:-1, :, 1]+1e-6)) * 1e2, cmap='hot', origin='lower', vmin=0, vmax=100)
#    t2_pred2 = axs3[k, 3].imshow((np.abs(mask * imgs[k+5][:, :, 1] - map[-1:-257:-1, :, 1]) / (map[-1:-257:-1, :, 1]+1e-6)) * 1e2, cmap='hot', origin='lower', vmin=0, vmax=100)
#    axs3[k, 1].set_title('{}'.format(k+1)+r'\textbf{ time step LSTM, T2 percentage error}', weight='bold')
#    axs3[k, 3].set_title('{}'.format(k+6)+r'\textbf{ time step LSTM, T2 percentage error}', weight='bold')
#    axs3[k, 1].set_axis_off()
#    axs3[k, 3].set_axis_off()
#    fig.colorbar(t2_pred1, ax=axs3[k, 1])
#    fig.colorbar(t2_pred2, ax=axs3[k, 3])
#fig3.show()
