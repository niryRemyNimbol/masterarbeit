import tensorflow as tf
import numpy as np
import dic
import rnn_functions
import display_functions

data_path = '../recon_q_examples/data/Exam52006/Series5/recon_data'
mask_path = '../recon_q_examples/data/Exam52006/Series5/mask.dat'
map_path = '../recon_q_examples/data/Exam52006/Series5/dm_qmaps.dat'
#dl_path = '../recon_q_examples/data/Exam52004/Series5/dl_qmaps.dat'

size = 200
Nreps = 1000

series = dic.read_mrf_data(data_path, Nreps, size)
mask = dic.load_relaxation_times_map(mask_path, size, method=2)
dm_map = dic.load_relaxation_times_map(map_path, size, method=0)
#dl_map = dic.load_relaxation_times_map(dl_path, size, method=1)

num_in = 100
num_fc = 64
timesteps = 10
num_hidden = 8
num_out = 2
times_max = np.array([4., .6])

X = tf.placeholder("float", [None, timesteps, num_in])
net = rnn_functions.LSTM(X, timesteps, num_hidden, num_out, fc=True)
out = times_max * net
saver = tf.train.Saver()

epoch = 1000

with tf.Session() as sess:
    rnn_functions.load_lstm(saver, sess, epoch)
    times = sess.run(out, feed_dict={X: series[:timesteps*num_in, :].T.reshape((series.shape[1], timesteps, num_in))})

img = times.reshape((size, size, 2), order='C')
img[:, :, 0] *= mask * 1e3
img[:, :, 1] *= mask * 1e3
img_dm = dm_map * 1e3
img_dm[:, :, 0] = img_dm[:, :, 0].T
img_dm[:, :, 1] = img_dm[:, :, 1].T

#true_t1 = np.array([604, 596,1448, 1262, 444, 754, 903, 1276, 1034, 745, 1160, 966])
#true_t2 = np.array([95, 136, 390, 184, 154, 116, 137, 204, 167, 157, 214, 224])

#corners = display_functions.detect_phantom_tubes(img, mask, 28, 2)
#img_gt, _, _ = display_functions.compare_to_gt(img, mask, corners, 28, true_t1, true_t2)
fig_t1, ax_t1, fig_t2, ax_t2 = display_functions.plot_results(img, phantom=False, gt=img_dm)
#fig_t1_dm, ax_t1_dm, fig_t2_dm, ax_t2_dm = display_functions.plot_comparison_method(img_dm, img, phantom=True, gt=img_gt)
#display_functions.draw_bounding_boxes(ax_t1, corners, 28)
#display_functions.draw_bounding_boxes(ax_t2, corners, 28)
