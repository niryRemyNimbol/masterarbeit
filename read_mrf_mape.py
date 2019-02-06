import numpy as np
import tensorflow as tf
from rnn_functions import RNN_MAPE
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

# Network Parameters
num_input = 64
timesteps = 10 # timesteps
num_hidden = 8
num_output = 2 # number of output parameters

# Fully Connected Layer Parameters
num_in_fc = 100

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_in_fc])

logits = RNN_MAPE(X, num_input, timesteps, num_hidden, num_output)
#logits = RNN_with_tr(X, num_input, timesteps, num_hidden, num_output)

out = logits
#out = [times_max*logit for logit in logits]

# Saver
saver = tf.train.Saver()

# Restoration directory
ckpt_dir = '../rnn_model_mape/'

with tf.Session() as sess:
    #    ckpt_file = ckpt_dir + 'model_fc_checkpoint1500.ckpt'
    ckpt_file = ckpt_dir + 'model_mape_checkpoint880.ckpt'
    #    ckpt_file = ckpt_dir + 'model_var_tr_norm_1_checkpoint300.ckpt'
    #    ckpt_file = ckpt_dir + 'model_tr_checkpoint100000.ckpt'
    #    ckpt_file = ckpt_dir + 'model_var_tr_norm_checkpoint975.ckpt'
    saver.restore(sess, ckpt_file)

    times = sess.run(out, feed_dict={X: series.T.reshape((series.shape[1], timesteps, num_in_fc))})

#imgs = [time.reshape((256,256,2), order='C') for time in times]
imgs = times.reshape((256,256,2), order='C')

#fig, axs = plt.subplots(2, 10, figsize=(50,10))
#for k in range(len(imgs)):
#    t1 = axs[0, k].imshow(imgs[k][:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
#    fig.colorbar(t1, ax=axs[0, k])
#    axs[0, k].set_title('T1, timestep {}'.format(k+1), weight='bold')
#    t2 = axs[1, k].imshow(imgs[k][:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
#    fig.colorbar(t2, ax=axs[1, k])
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
t1 = axs[0].imshow(mask * imgs[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
fig.colorbar(t1, ax=axs[0])
t2 = axs[1].imshow(mask * imgs[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
fig.colorbar(t2, ax=axs[1])
fig.show()
