from src.misc.torch_utils import torch2numpy

import os
import numpy as np

import matplotlib.pyplot as plt


def plot_pca_predictions(actual, predicted, ts, args, num_obs=10, name='plt_latents'):
    num_obs = np.minimum(actual.shape[0], num_obs)
    idx = np.random.permutation(actual.shape[0])
    actual = actual[idx]
    predicted = predicted[idx]

    num_obs = min(actual.shape[0], num_obs)
    for n in range(num_obs):
        fig, axs = plt.subplots(1, actual.shape[-1], figsize=(4 * actual.shape[-1], 3))
        for j in range(actual.shape[-1]):
            axs[j].scatter(torch2numpy(ts)[:actual[n, :, j].shape[0]], torch2numpy(actual[n, :, j]),
                           c='k', marker='.', s=10, zorder=0, alpha=0.9)
            axs[j].plot(torch2numpy(ts), torch2numpy(predicted[n, :, j]),
                        c='r', alpha=0.7, zorder=3)
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        if name is None:
            plt.show()
        else:
            fig.savefig(os.path.join(args.save, name + "_{}.png".format(n)), dpi=160,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)


def plot_data_predictions(actual, predicted, ts, args, num_obs=10, name='plt_predictions'):
    num_obs = np.minimum(actual.shape[0], num_obs)
    idx = np.random.permutation(actual.shape[0])
    actual = actual[idx]
    predicted = predicted[idx]

    num_obs = min(actual.shape[0], num_obs)
    for n in range(num_obs):
        fig, axs = plt.subplots(10, 5, figsize=(4 * 5, 3 * 10))
        count = 0
        for i in range(10):
            for j in range(5):
                axs[i, j].scatter(torch2numpy(ts)[:actual[n, :, count].shape[0]], torch2numpy(actual[n, :, count]),
                                  c='k', marker='.', s=10, zorder=0, alpha=0.9)
                axs[i, j].plot(torch2numpy(ts), torch2numpy(predicted[n, :, count]),
                               c='r', alpha=0.7, zorder=3)
                count += 1
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        if name is None:
            plt.show()
        else:
            fig.savefig(os.path.join(args.save, name + "_{}.png".format(n)), dpi=160,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
