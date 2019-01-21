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
#data_path = '../recon_q_examples/data/recon_data'
mrf = read_mrf_data(data_path, 1000, 256)
series = mrf.reshape((1000, 256**2))
series /= np.linalg.norm(series, axis=0)
mask_id = open(mask_path, 'rb')
map_id = open(map_path, 'rb')
mask = np.reshape(np.fromfile(mask_id, np.float32), [256,256])
map = np.reshape(np.fromfile(map_id, np.float32), [256,256,2],order='F')
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
    ckpt_dir = '../rnn_model_len/rnn_model_len{}/'.format(timestep)

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        #    sess.run(init)
        ckpt_file = ckpt_dir + 'model_fc_len{}_checkpoint5000.ckpt'.format(timestep*num_in_fc)
        saver.restore(sess, ckpt_file)

        times.append(sess.run(out, feed_dict={X: series[:timestep*num_in_fc, :].T.reshape((series.shape[1], timestep, num_in_fc))}))
# tf Graph input
#X = tf.placeholder("float", [None, timesteps, num_in_fc])

#logits = RNN_with_fc(X, num_input, timesteps, num_hidden, num_output)
#logits = RNN_with_tr(X, num_input, timesteps, num_hidden, num_output)

#out = times_max * logits
#out = [times_max*logit for logit in logits]

# Saver
#saver = tf.train.Saver()

# Restoration directory
#ckpt_dir = '../rnn_model_len/'

#with tf.Session() as sess:
#    ckpt_file = ckpt_dir + 'model_fc_checkpoint10000.ckpt'
#    ckpt_file = ckpt_dir + 'rnn_model_len4/model_fc_len400_checkpoint5000.ckpt'
#    ckpt_file = ckpt_dir + 'model_var_tr_norm_1_checkpoint300.ckpt'
#    ckpt_file = ckpt_dir + 'model_tr_checkpoint100000.ckpt'
#    ckpt_file = ckpt_dir + 'model_var_tr_norm_checkpoint975.ckpt'
#    saver.restore(sess, ckpt_file)


imgs = [time.reshape((256,256,2), order='C') for time in times]
#imgs = times.reshape((256,256,2), order='C')

fig, axs = plt.subplots(4, 11, figsize=(50,20))
for k in range(len(imgs)):
    t1_pred = axs[0, k].imshow(mask * imgs[k][:, :, 0] * 1e3, cmap='hot', origin='lower', vmin=0, vmax=3000)
    axs[0, k].set_title('T1, timestep {} (ms)'.format(k+1), weight='bold')
    fig.colorbar(t1_pred, ax=axs[0, k])
    t2_pred = axs[1, k].imshow(mask * imgs[k][:, :, 1] * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
    axs[0, 1].set_title('T2; timesttep {} (ms)'.format(k+1), weight='bold')
    fig.colorbar(t2_pred, ax=axs[1, k])
    a = (mask[0:-1, :] * imgs[k][0:-1, :, 0] * 1e3).flatten()
    b = (map[-1:0:-1, :, 0] * 1e3).flatten()
    axs[2, k].plot(b, a, '.')
    axs[2, k].plot(b, b, '--')
    axs[2, k].set_title('T1, timestep {}'.format(k+1), weight='bold')
    axs[2, k].set_xlabel('Dictionary matching (ms)')
    axs[2, k].set_ylabel('Prediction (ms)')
    a = (mask[0:-1, :] * imgs[k][0:-1, :, 1] * 1e3).flatten()
    b = (map[-1:0:-1, :, 1] * 1e3).flatten()
    axs[3, k].plot(b, a, '.')
    axs[3, k].plot(b, b, '--')
    axs[3, k].set_title('T2, timestep {}'.format(k+1), weight='bold')
    axs[3, k].set_xlabel('Dictionary matching (ms)')
    axs[3, k].set_ylabel('Prediction (ms)')
t1_DM = axs[1, 10].imshow(map[-1:0:-1, :, 0] * 1e3, cmap='hot', origin='lower', vmin=0, vmax=3000)
axs[1, 10].set_title('T1, dictionary matching (ms)', weight='bold')
fig.colorbar(t1_DM, ax=axs[1, 10])
t2_DM = axs[2, 10].imshow(map[-1:0:-1, :, 1] * 1e3, cmap='copper', origin='lower', vmin=0, vmax=300)
axs[2, 10].set_title('T2, dictionary matching (ms)', weight='bold')
fig.colorbar(t2_DM, ax=axs[2, 10])
#fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#t1 = axs[0].imshow(mask*imgs[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3.0)
#fig.colorbar(t1, ax=axs[0])
#t2 = axs[1].imshow(mask*imgs[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=0.3)
#fig.colorbar(t2, ax=axs[1])
fig.show()

#angles=[(k,l) for k in range(0,221,2) for l in range(0,221,2)]
#fig, axs = plt.subplots(3, 4, figsize=(20, 15))
#m = 0
#n = 0
#aprev = 10
#for k, l in angles:
#    if mask[k:k + 36, l].sum() == 0 and mask[k:k + 36, l + 35].sum() == 0 and mask[k, l:l + 36].sum() == 0 and mask[k + 35, l:l + 36].sum() == 0 and mask[k:k + 36, l:l + 36].sum() > 0:
#        a = np.mean(mask[k:k + 36, l:l + 36] * imgs[k:k + 36, l:l + 36, 1], axis=(0, 1))
#        if np.abs(a - aprev) > 1e-4:
#            axs[m, n].imshow(mask[k:k + 36, l:l + 36] * imgs[k:k + 36, l:l + 36, 0], cmap='hot', origin='lower', vmin=0,
#                             vmax=3.0)
#            aprev = a
#            if m == 2:
#                if n <= 2:
#                    n += 1
#                    m = 0
#            else:
#                m += 1
#fig.show()
