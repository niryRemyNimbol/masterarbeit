import dic
import matplotlib.pyplot as plt
import numpy as np

dictionary = dic.dic('../recon_q_examples/dict/', 'fisp_mrf_const_tr', 1000, 10)
D = dictionary.D[:, dictionary.lut[0, :]>=dictionary.lut[1, :]]

series_mag = np.abs(D.T + 0.02 * np.max(np.real(D)) * np.random.normal(0.0, 1.0, D.T.shape) + 1j * 0.02 * np.max(np.imag(D)) * np.random.normal(0.0, 1.0, D.T.shape)).T
series_mag /= np.linalg.norm(series_mag, axis=0)
series_mag = series_mag.T

relaxation_times = dictionary.lut[:, dictionary.lut[0, :] >= dictionary.lut[1, :]][0:2].T

fa_path = '../recon_q_examples/data/fa.dat'
fa_id = open(fa_path, 'rb')
fa = np.fromfile(fa_id, np.float32)

plt.rc('text', usetex=True)
fig_fa, ax_fa = plt.subplots(1, 1, figsize=(5, 5))
ax_fa.plot(fa)
ax_fa.set_title(r'\textbf{Acquisition scheme')
ax_fa.set_ylabel(r'Flip angle (degrees)')
fig_fa.show()

fig_series, ax_series = plt.subplots(1, 1, figsize=(5, 5))
ax_series.plot(series_mag[10, :])
ax_series.set_title(r'Noisy simulated series, T1: '
                    +str(int(1000*relaxation_times[10, 0]))+r' ms, T2: '
                    +str(int(1000*relaxation_times[10, 1]))+r' ms')
fig_series.show()
