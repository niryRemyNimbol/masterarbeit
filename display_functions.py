import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score as r2

plt.rc('text', usetex=True)

def detect_phantom_tubes(phantom_map, mask, width, search_step):
    corners = []
    aprev = 10
    sh = phantom_map.shape
    angles = [(k,l) for k in range(0, sh[0]-width, search_step) for l in range(0, sh[1]-width, search_step)]
    for k, l in angles:
        if mask[k:k+width+1, l].sum() == 0 and mask[k:k+width+1, l+width].sum() == 0 and mask[k, l:l+width+1].sum() == 0 and mask[k+width, l:l+width+1].sum() == 0 and mask[k:k+width+1, l:l+width+1].sum() > 10:
            a = np.mean(mask[k:k+width+1, l:l+width+1] * phantom_map[k:k+width+1, l:l+width+1, 0], axis=(0, 1))
            if np.abs(a - aprev) > 1e-3:
                corners.append((k, l))
                aprev = a
    return corners

def draw_bounding_boxes(ax, corners, width):
    label = 1
    for k, l in corners:
        ax[0].annotate('{}'.format(label), (l, k), color='y')
        ax[0].vlines([l, l+width+1], k, k+width+1, colors='y', label='{}'.format(label))
        ax[0].hlines([k, k+width+1], l, l+width+1, colors='y', label='{}'.format(label))
        ax[1].annotate('{}'.format(label), (l, k), color='r')
        ax[1].vlines([l, l+width+1], k, k+width+1, colors='r', label='{}'.format(label))
        ax[1].hlines([k, k+width+1], l, l+width+1, colors='r', label='{}'.format(label))
        label += 1
    return ax

def compare_to_gt(phantom_map, mask, corners, width, true_t1, true_t2):
    img_gt = np.zeros_like(phantom_map)
    y_t1_mean = []
    y_t2_mean = []
    label = 1
    for k, l in corners:
        tube_t1 = mask[k:k+width+1, l:l+width+1] * phantom_map[k:k+width+1, l:l+width+1, 0] * 1e3
        tube_t2 = mask[k:k+width+1, l:l+width+1] * phantom_map[k:k+width+1, l:l+width+1, 1] * 1e3
        y_t1_mean.append(np.mean(tube_t1[tube_t1 > 0].flatten()))
        y_t2_mean.append(np.mean(tube_t2[tube_t2 > 0].flatten()))
        ind = np.where(tube_t1 > 0)
        offset = [k * np.ones_like(ind[0]), l * np.ones_like(ind[1])]
        img_gt[ind[0] + offset[0], ind[1] + offset[1], :] = np.array([true_t1[label-1], true_t2[label-1]])
        label += 1
    return img_gt, y_t1_mean, y_t2_mean

def plot_results(img, phantom=False, gt=None, len=False, tr=False, step=10):
    fig_t1, ax_t1 = plt.subplots(3, 1, figsize=(5, 15))
    fig_t2, ax_t2 = plt.subplots(3, 1, figsize=(5, 15))
    t1_len = ax_t1[0].imshow(img[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3000)
    t2_len = ax_t2[0].imshow(img[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=300)
    if phantom:
        t1_err = ax_t1[1].imshow(
            (np.abs(img[:, :, 0] - gt[:, :, 0]) / (gt[:, :, 0] + 1e-6)) * 1e2,
            cmap='Reds', origin='lower', vmin=0, vmax=100)
        t2_err = ax_t2[1].imshow(
            (np.abs(img[:, :, 1] - gt[:, :, 1]) / (gt[:, :, 1] + 1e-6)) * 1e2,
            cmap='Reds', origin='lower', vmin=0, vmax=100)
        ax_t1[2].scatter(gt[:, :, 0], img[:, :, 0], c='b', marker='.', alpha=0.1)
        ax_t2[2].scatter(gt[:, :, 1], img[:, :, 1], c='b', marker='.', alpha=0.1)
        r2_t1 = r2(gt[:, :, 0], img[:, :, 0])
        r2_t2 = r2(gt[:, :, 1], img[:, :, 1])
        ax_t1[2].set_xlabel(r'Ground truth (ms)')
        ax_t2[2].set_xlabel(r'Ground truth (ms)')
    else:
        t1_err = ax_t1[1].imshow(
            (np.abs(img[:, :, 0] - gt[:, :, 0]) / (gt[:, :, 0] + 1e-6)) * 1e2,
            cmap='Reds', origin='lower', vmin=0, vmax=100)
        t2_err = ax_t2[1].imshow(
            (np.abs(img[:, :, 1] - gt[:, :, 1]) / (gt[:, :, 1] + 1e-6)) * 1e2,
            cmap='Reds', origin='lower', vmin=0, vmax=100)
        ax_t1[2].scatter(gt[:, :, 0], img[:, :, 0], c='r', marker='.',
                            alpha=0.1)
        ax_t2[2].scatter(gt[:, :, 1], img[:, :, 1], c='r', marker='.',
                            alpha=0.1)
        r2_t1 = r2(gt[:, :, 0], img[:, :, 0])
        r2_t2 = r2(gt[:, :, 1], img[:, :, 1])
        ax_t1[2].set_xlabel(r'Dictionary matching (ms)')
        ax_t2[2].set_xlabel(r'Dictionary matching (ms)')
    ax_t1[2].plot([x for x in range(4000)], [x for x in range(4000)], 'g--')
    ax_t1[0].set_axis_off()
    ax_t1[1].set_axis_off()
    fig_t1.colorbar(t1_len, ax=ax_t1[0])
    fig_t1.colorbar(t1_err, ax=ax_t1[1])
    ax_t1[2].text(1, 3550, r'R2 = {:5f}'.format(r2_t1))
    ax_t1[2].set_ylabel(r'Predictions (ms)')
    ax_t1[2].set_xbound(lower=0, upper=4000)
    ax_t1[2].set_ybound(lower=0, upper=4000)
    ax_t2[2].plot([x for x in range(600)], [x for x in range(600)], 'g--')
    ax_t2[0].set_axis_off()
    ax_t2[1].set_axis_off()
    fig_t2.colorbar(t2_len, ax=ax_t2[0])
    fig_t2.colorbar(t2_err, ax=ax_t2[1])
    ax_t2[2].text(1, 550, r'R2 = {:5f}'.format(r2_t2))
    ax_t2[2].set_ylabel(r'Predictions (ms)')
    ax_t2[2].set_xbound(lower=0, upper=600)
    ax_t2[2].set_ybound(lower=0, upper=600)
    if len:
        if step > 0:
            ax_t1[0].set_title(r'\textbf{T1 (ms), }' + '{}'.format(step + 1) + r'\textbf{ time steps}')
            ax_t1[1].set_title(r'\textbf{T1 percentage error, }' + '{}'.format(step + 1) + r'\textbf{ time steps}')
            ax_t1[2].set_title(r'\textbf{T1 scatter plot, }' + '{}'.format(step + 1) + r'\textbf{ time step}')
            ax_t2[0].set_title(r'\textbf{T2 (ms), }' + '{}'.format(step + 1) + r'\textbf{ time steps}')
            ax_t2[1].set_title(r'\textbf{T2 percentage error, }' + '{}'.format(step + 1) + r'\textbf{ time steps}')
            ax_t2[2].set_title(r'\textbf{T2 scatter plot, }' + '{}'.format(step + 1) + r'\textbf{ time step}')
        else:
            ax_t1[0].set_title(r'\textbf{T1 (ms), }' + '{}'.format(step + 1) + r'\textbf{ time step}')
            ax_t1[1].set_title(r'\textbf{T1 percentage error, }' + '{}'.format(step + 1) + r'\textbf{ time step}')
            ax_t1[2].set_title(r'\textbf{T1 scatter plot, }' + '{}'.format(step + 1) + r'\textbf{ time step}')
            ax_t2[0].set_title(r'\textbf{T2 (ms), }' + '{}'.format(step + 1) + r'\textbf{ time step}')
            ax_t2[1].set_title(r'\textbf{T2 percentage error, }' + '{}'.format(step + 1) + r'\textbf{ time step}')
            ax_t2[2].set_title(r'\textbf{T2 scatter plot, }' + '{}'.format(step + 1) + r'\textbf{ time step}')
    elif tr:
        ax_t1[0].set_title(r'\textbf{T1 (ms), time step \#}' + '{}'.format(step + 1))
        ax_t1[1].set_title(r'\textbf{T1 percentage error, time step \#}' + '{}'.format(step + 1))
        ax_t1[2].set_title(r'\textbf{T1 scatter plot, time step \#}' + '{}'.format(step + 1))
        ax_t2[0].set_title(r'\textbf{T2 (ms), time step \#}' + '{}'.format(step + 1))
        ax_t2[1].set_title(r'\textbf{T2 percentage error, time step \#}' + '{}'.format(step + 1))
        ax_t2[2].set_title(r'\textbf{T2 scatter plot, time step \#}' + '{}'.format(step + 1))
    else:
        ax_t1[0].set_title(r'\textbf{T1 (ms), LSTM}')
        ax_t1[1].set_title(r'\textbf{T1 percentage error}')
        ax_t1[2].set_title(r'\textbf{T1 scatter plot}')
        ax_t2[0].set_title(r'\textbf{T2 (ms), LSTM}')
        ax_t2[1].set_title(r'\textbf{T2 percentage error}')
        ax_t2[2].set_title(r'\textbf{T2 scatter plot}')
    return fig_t1, ax_t1, fig_t2, ax_t2

def plot_comparison_method(img, img_lstm, phantom=False, gt=None, method=0):
    if not phantom:
        if method == 0:
            fig_t1, ax_t1 = plt.subplots(1, 1, figsize=(5, 5))
            fig_t2, ax_t2 = plt.subplots(1, 1, figsize=(5, 5))
            t1 = ax_t1.imshow(img[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3000)
            t2 = ax_t2.imshow(img[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=300)
            fig_t1.colorbar(t1, ax=ax_t1)
            fig_t1.colorbar(t2, ax=ax_t2)
            ax_t1.set_title(r'\textbf{T1 (ms)}')
            ax_t2.set_title(r'\textbf{T2 (ms)}')
    else:
        fig_t1, ax_t1 = plt.subplots(3, 1, figsize=(5, 15))
        fig_t2, ax_t2 = plt.subplots(3, 1, figsize=(5, 15))
        t1 = ax_t1[0].imshow(img[:, :, 0], cmap='hot', origin='lower', vmin=0, vmax=3000)
        t2 = ax_t2[0].imshow(img[:, :, 1], cmap='copper', origin='lower', vmin=0, vmax=300)
        t1_err = ax_t1[1].imshow((np.abs(img[:, :, 0] - gt[:, :, 0]) / (gt[:, :, 0] + 1e-6)) * 1e2,
                                 cmap='Reds', origin='lower', vmin=0, vmax=100)
        t2_err = ax_t2[1].imshow((np.abs(img[:, :, 1] - gt[:, :, 1]) / (gt[:, :, 1] + 1e-6)) * 1e2,
                                 cmap='Reds', origin='lower', vmin=0, vmax=100)
        sclstm1 = ax_t1[2].scatter(gt[:, :, 0], img_lstm[:, :, 0], c='b', marker='.', alpha=0.1)
        sc1 = ax_t1[2].scatter(gt[:, :, 0], img[:, :, 0], c='r', marker='.', alpha=0.1)
        sclstm2 = ax_t2[2].scatter(gt[:, :, 1], img_lstm[:, :, 1], c='b', marker='.', alpha=0.1)
        sc2 = ax_t2[2].scatter(gt[:, :, 1], img[:, :, 1], c='r', marker='.', alpha=0.1)
        r2_t1_lstm = r2(gt[:, :, 0], img_lstm[:, :, 0])
        r2_t2_lstm = r2(gt[:, :, 1], img_lstm[:, :, 1])
        r2_t1 = r2(gt[:, :, 0], img[:, :, 0])
        r2_t2 = r2(gt[:, :, 1], img[:, :, 1])
        ax_t1[2].plot([x for x in range(4000)], [x for x in range(4000)], 'g--')
        ax_t1[0].set_axis_off()
        ax_t1[1].set_axis_off()
        fig_t1.colorbar(t1, ax=ax_t1[0])
        fig_t1.colorbar(t1_err, ax=ax_t1[1])
        ax_t1[2].set_xbound(lower=0, upper=4000)
        ax_t1[2].set_ybound(lower=0, upper=4000)
        ax_t2[2].plot([x for x in range(600)], [x for x in range(600)], 'g--')
        ax_t2[0].set_axis_off()
        ax_t2[1].set_axis_off()
        fig_t2.colorbar(t2, ax=ax_t2[0])
        fig_t2.colorbar(t2_err, ax=ax_t2[1])
        ax_t2[2].set_xbound(lower=0, upper=600)
        ax_t2[2].set_ybound(lower=0, upper=600)
        if phantom:
            ax_t1[2].set_xlabel(r'Ground truth (ms)')
            ax_t2[2].set_xlabel(r'Ground truth (ms)')
        else:
            ax_t1[2].set_xlabel(r'Dictionary matching (ms)')
            ax_t2[2].set_xlabel(r'Dictionary matching (ms)')
        if method == 1:
            ax_t1[0].set_title(r'\textbf{T1 (ms), MRF net}')
            ax_t1[1].set_title(r'\textbf{T1 percentage error}')
            ax_t1[2].set_title(r'\textbf{T1 scatter plot}')
            ax_t1[2].text(1, 3550, r'R2 = {:5f} (LSTM) / {:5f} (MRF net)'.format(r2_t1_lstm, r2_t1))
            ax_t1[2].set_ylabel(r'Predictions (ms)')
            ax_t1[2].legend((sclstm1, sc1), (r'LSTM', r'MRF net'), loc=4)
            ax_t2[0].set_title(r'\textbf{T2 (ms), MRF net}')
            ax_t2[1].set_title(r'\textbf{T2 percentage error}')
            ax_t2[2].set_title(r'\textbf{T2 scatter plot}')
            ax_t2[2].text(1, 550, r'R2 = {:5f} (LSTM) / {:5f} (MRF net)'.format(r2_t2_lstm, r2_t2))
            ax_t2[2].set_ylabel(r'Predictions (ms)')
            ax_t2[2].legend((sclstm2, sc2), (r'LSTM', r'MRF net'), loc=4)
        else:
            ax_t1[0].set_title(r'\textbf{T1 (ms), DM}')
            ax_t1[1].set_title(r'\textbf{T1 percentage error}')
            ax_t1[2].set_title(r'\textbf{T1 scatter plot}')
            ax_t1[2].text(1, 3550, r'R2 = {:5f} (LSTM) / {:5f} (DM)'.format(r2_t1_lstm, r2_t1))
            ax_t1[2].set_ylabel(r'Predictions (ms)')
            ax_t1[2].legend((sclstm1, sc1), (r'LSTM', r'DM'), loc=4)
            ax_t2[0].set_title(r'\textbf{T2 (ms), DM}')
            ax_t2[1].set_title(r'\textbf{T2 percentage error}')
            ax_t2[2].set_title(r'\textbf{T2 scatter plot}')
            ax_t2[2].text(1, 550, r'R2 = {:5f} (LSTM) / {:5f} (DM)'.format(r2_t2_lstm, r2_t2))
            ax_t2[2].set_ylabel(r'Predictions (ms)')
            ax_t2[2].legend((sclstm2, sc2), (r'LSTM', r'DM'), loc=4)
    return fig_t1, ax_t1, fig_t2, ax_t2

