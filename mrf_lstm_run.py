import tensorflow as tf
import numpy as np
import dic
import rnn_functions
import display_functions
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2

data_path = '../recon_q_examples/data/Exam52004/Series5/recon_data'
mask_path = '../recon_q_examples/data/Exam52004/Series5/mask.dat'
map_path = '../recon_q_examples/data/Exam52004/Series5/dm_qmaps.dat'
dl_path = '../recon_q_examples/data/Exam52004/Series5/nn_qmaps.dat'

size = 200
Nreps = 1000

series = dic.read_mrf_data(data_path, Nreps, size)
mask = dic.load_relaxation_times_map(mask_path, size, method=2)
dm_map = dic.load_relaxation_times_map(map_path, size, method=0)
dl_map = dic.load_relaxation_times_map(dl_path, size, method=1)

num_in = 100
num_fc = 64
timesteps = 10
num_hidden = 8
num_out = 2
times_max = np.array([4., .6])
#times_max = np.array([4., 1.])

X = tf.placeholder("float", [None, timesteps, num_in])
net = rnn_functions.LSTM(X, timesteps, num_hidden, num_out, activation=tf.sigmoid, fc=True, tr=True)
#out = times_max * net
out = [times_max * net_ for net_ in net]
saver = tf.train.Saver()

epoch = 1940

with tf.Session() as sess:
    rnn_functions.load_lstm(saver, sess, epoch, '_sigmoid_mape_tr')
    times = sess.run(out, feed_dict={X: series[:timesteps*num_in, :].T.reshape((series.shape[1], timesteps, num_in))})

img = times[9].reshape((size, size, 2), order='C')
#img = times.reshape((size, size, 2), order='C')
img[:, :, 0] *= mask * 1e3
img[:, :, 1] *= mask * 1e3
img_dl = dl_map * 1e3
img_dl[:, :, 0] = mask * img_dl[:, :, 0].T
img_dl[:, :, 1] = mask * img_dl[:, :, 1].T
img_dm = dm_map * 1e3
img_dm[:, :, 0] = img_dm[:, :, 0].T
img_dm[:, :, 1] = img_dm[:, :, 1].T

true_t1 = np.array([604, 596,1448, 1262, 444, 754, 903, 1276, 1034, 745, 1160, 966])
true_t2 = np.array([95, 136, 390, 184, 154, 116, 137, 204, 167, 157, 214, 224])

corners = display_functions.detect_phantom_tubes(img, mask, 28, 2)
img_gt, _, _ = display_functions.compare_to_gt(img, mask, corners, 28, true_t1, true_t2)
fig_t1, ax_t1, fig_t2, ax_t2 = display_functions.plot_results(img, phantom=True, gt=img_gt)
fig_t1_dm, ax_t1_dm, fig_t2_dm, ax_t2_dm = display_functions.plot_comparison_method(img_dm, img, phantom=True, gt=img_gt)
fig_t1_dl, ax_t1_dl, fig_t2_dl, ax_t2_dl = display_functions.plot_comparison_method(img_dl, img, phantom=True, gt=img_gt, method=1)
#display_functions.draw_bounding_boxes(ax_t1, corners, 28)
#display_functions.draw_bounding_boxes(ax_t2, corners, 28)

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
imgs = []
for k in range(len(times)):
    img = times[k].reshape((size, size, 2), order='C')
    img[:, :, 0] *= mask * 1e3
    img[:, :, 1] *= mask * 1e3
    imgs.append(img)

#fig_tr, ax_tr = plt.subplots(10, 6, figsize=(18, 30))
#for k in range(len(imgs)):
#    t1 = ax_tr[k][0].imshow(imgs[k][:, :, 0], cmap='hot', vmax=4000, vmin=0)
#    t1_err = ax_tr[k][1].imshow(np.abs(imgs[k][:, :, 0] - img_dm[:, :, 0])/(img_dm[:, :, 0] + 1e-6) * 1e2, cmap='Reds', vmax=100, vmin=0)
#    ax_tr[k][2].scatter(img_dm[:, :, 0], imgs[k][:, :, 0], c='b', marker='.', alpha=0.1)
#    t2 = ax_tr[k][3].imshow(imgs[k][:, :, 1], cmap='copper', vmax=300, vmin=0)
#    t2_err = ax_tr[k][4].imshow(np.abs(imgs[k][:, :, 1] - img_dm[:, :, 1])/(img_dm[:, :, 1] + 1e-6) * 1e2, cmap='Reds', vmax=100, vmin=0)
#    ax_tr[k][5].scatter(img_dm[:, :, 1], imgs[k][:, :, 1], c='b', marker='.', alpha=0.1)
#    r2_t1 = r2(img_dm[:, :, 0], imgs[k][:, :, 0])
#    r2_t2 = r2(img_dm[:, :, 1], imgs[k][:, :, 1])
#    ax_tr[k][0].text(-35, 100, r'\Huge {:d}'.format((k+1)))
#    ax_tr[k][2].text(1, 3550, r'\Large R2 = {:5f}'.format(r2_t1))
#    ax_tr[k][2].set_xlabel(r'\Large DM (ms)')
#    ax_tr[k][2].set_ylabel(r'\Large LSTM (ms)')
#    ax_tr[k][2].set_xbound(lower=0, upper=4000)
#    ax_tr[k][2].set_ybound(lower=0, upper=4000)
#    ax_tr[k][2].plot([x for x in range(4000)], [x for x in range(4000)], 'g--')
#    asp = np.diff(ax_tr[k][2].get_xlim())[0] / np.diff(ax_tr[k][2].get_ylim())[0]
#    ax_tr[k][2].set_aspect(asp)
#    ax_tr[k][2].ticklabel_format(style='sci', axis='both', scilimits=(3, 3))
#    ax_tr[k][0].set_axis_off()
#    ax_tr[k][1].set_axis_off()
#    ax_tr[k][5].text(1, 550, r'\Large R2 = {:5f}'.format(r2_t2))
#    ax_tr[k][5].set_xlabel(r'\Large DM (ms)')
#    ax_tr[k][5].set_ylabel(r'\Large LSTM (ms)')
#    ax_tr[k][5].set_xbound(lower=0, upper=600)
#    ax_tr[k][5].set_ybound(lower=0, upper=600)
#    ax_tr[k][5].plot([x for x in range(600)], [x for x in range(600)], 'g--')
#    asp = np.diff(ax_tr[k][5].get_xlim())[0] / np.diff(ax_tr[k][5].get_ylim())[0]
#    ax_tr[k][5].set_aspect(asp)
#    ax_tr[k][5].ticklabel_format(style='sci', axis='both', scilimits=(2, 2))
#    ax_tr[k][5].set_xticks(ax_tr[k][5].get_yticks()[1:-1])
#    ax_tr[k][3].set_axis_off()
#    ax_tr[k][4].set_axis_off()

#fig_tr.colorbar(t1, fraction=0.05, pad=-0.05, ax=ax_tr[9][0], orientation='horizontal')
#fig_tr.colorbar(t1_err, fraction=0.05, pad=-0.05, ax=ax_tr[9][1], orientation='horizontal')
#fig_tr.colorbar(t2, fraction=0.05, pad=-0.05, ax=ax_tr[9][3], orientation='horizontal')
#fig_tr.colorbar(t2_err, fraction=0.05, pad=-0.05, ax=ax_tr[9][4], orientation='horizontal')
#ax_tr[0][0].set_title(r'\Huge \textbf{T1 (ms)}')
#ax_tr[0][3].set_title(r'\Huge \textbf{T2 (ms)}')
#ax_tr[0][1].set_title(r'\Huge \textbf{T1 Error (\%)')
#ax_tr[0][4].set_title(r'\Huge \textbf{T2 Error (\%)')

#fig_comp, ax_comp = plt.subplots(2, 4, figsize=(20, 10))
#t1 = ax_comp[0][0].imshow(imgs[9][:, :, 0], cmap='hot', vmax=4000, vmin=0)
#t1_err = ax_comp[0][2].imshow(np.abs(imgs[9][:, :, 0] - img_dm[:, :, 0])/(img_dm[:, :, 0] + 1e-6) * 1e2, cmap='Reds', vmax=100, vmin=0)
#t1_dm = ax_comp[0][1].imshow(img_dm[:, :, 0], cmap='hot', vmax=4000, vmin=0)
#ax_comp[0][3].scatter(img_dm[:, :, 0], imgs[9][:, :, 0], c='b', marker='.', alpha=0.1)
#t2 = ax_comp[1][0].imshow(imgs[9][:, :, 1], cmap='copper', vmax=300, vmin=0)
#t2_err = ax_comp[1][2].imshow(np.abs(imgs[9][:, :, 1] - img_dm[:, :, 1])/(img_dm[:, :, 1] + 1e-6) * 1e2, cmap='Reds', vmax=100, vmin=0)
#t2_dm = ax_comp[1][1].imshow(img_dm[:, :, 1], cmap='copper', vmax=300, vmin=0)
#ax_comp[1][3].scatter(img_dm[:, :, 1], imgs[9][:, :, 1], c='b', marker='.', alpha=0.1)
#r2_t1 = r2(img_dm[:, :, 0], imgs[9][:, :, 0])
#r2_t2 = r2(img_dm[:, :, 1], imgs[9][:, :, 1])
#ax_comp[0][3].text(1, 3550, r'\huge R2 = {:5f}'.format(r2_t1))
#ax_comp[0][3].set_xlabel(r'\huge DM (ms)')
#ax_comp[0][3].set_ylabel(r'\huge LSTM (ms)')
#ax_comp[0][3].set_xbound(lower=0, upper=4000)
#ax_comp[0][3].set_ybound(lower=0, upper=4000)
#ax_comp[0][3].plot([x for x in range(4000)], [x for x in range(4000)], 'g--')
#ax_comp[0][0].set_axis_off()
#ax_comp[0][1].set_axis_off()
#ax_comp[0][2].set_axis_off()
#ax_comp[1][3].text(1, 550, r'\huge R2 = {:5f}'.format(r2_t2))
#ax_comp[1][3].set_xlabel(r'\huge Dictionary matching (ms)')
#ax_comp[1][3].set_ylabel(r'\huge LSTM (ms)')
#ax_comp[1][3].set_xbound(lower=0, upper=600)
#ax_comp[1][3].set_ybound(lower=0, upper=600)
#ax_comp[1][3].plot([x for x in range(600)], [x for x in range(600)], 'g--')
#ax_comp[1][0].set_axis_off()
#ax_comp[1][1].set_axis_off()
#ax_comp[1][2].set_axis_off()
#fig_comp.colorbar(t1, fraction=0.05, pad=-0.05, ax=ax_comp[0][0], orientation='horizontal')
#fig_comp.colorbar(t1_dm, fraction=0.05, pad=-0.05, ax=ax_comp[0][1], orientation='horizontal')
#fig_comp.colorbar(t1_err, fraction=0.05, pad=-0.05, ax=ax_comp[0][2], orientation='horizontal')
#fig_comp.colorbar(t2, fraction=0.05, pad=-0.05, ax=ax_comp[1][0], orientation='horizontal')
#fig_comp.colorbar(t2_dm, fraction=0.05, pad=-0.05, ax=ax_comp[1][1], orientation='horizontal')
#fig_comp.colorbar(t2_err, fraction=0.05, pad=-0.05, ax=ax_comp[1][2], orientation='horizontal')
#ax_comp[0][0].text(-80, 100, r'\Huge \textbf{T1 (ms)}')
#ax_comp[1][0].text(-80, 100, r'\Huge \textbf{T2 (ms)}')
#ax_comp[0][0].set_title(r'\Huge \textbf{LSTM}')
#ax_comp[0][1].set_title(r'\Huge \textbf{DM}')
#ax_comp[0][2].set_title(r'\Huge \textbf{Absolute Percentage Error')
