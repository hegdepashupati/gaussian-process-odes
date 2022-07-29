from src.misc.torch_utils import torch2numpy

import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_longitudinal(data, test_pred, args, make_plot=False):
    fig, axs = plt.subplots(1, 2, figsize=(8 * 2, 3 * 1))
    for d in range(2):
        axs[d].plot(data.tst.ts, test_pred[:, d], c='r', alpha=0.7, zorder=3)
        axs[d].plot(data.tst.ts, data.tst.ys[0, :, d], c='k', alpha=0.7, zorder=2)
        axs[d].scatter(data.trn.ts, data.trn.ys[0, :, d], c='k', s=100, marker='.', zorder=200)
        axs[d].set_title("State {}".format(d + 1))
        axs[d].set_xlabel("Time")
        axs[d].scatter([], [], c='k', s=10, marker='.', label='train obs')
        axs[d].plot([], [], c='k', alpha=0.7, label='true trajectory')
        axs[d].plot([], [], c='r', alpha=0.7, label='predictive trajectory')
    axs[d].legend(loc="upper right")
    fig.suptitle("Predictive plot for NeuralODE")
    plt.show()
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, 'plt_longitudinal.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)


def plot_vectorfield(flow, data, test_pred, args, make_plot=False):
    grid_size = 30
    xlim = data.xlim
    ylim = data.ylim
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex='all', sharey='all')

    grid_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
    grid_f = []
    for gx in grid_x:
        grid_f.append(data.f(gx, None))
    grid_f = np.stack(grid_f)

    ax1.streamplot(xx, yy,
                   grid_f[:, 0].reshape(xx.shape),
                   grid_f[:, 1].reshape(xx.shape),
                   color='grey')
    ax1.set_title('True vectorfield')
    ax1.scatter(data.trn.ys[:, :, 0], data.trn.ys[:, :, 1], marker='.', c='k', alpha=0.8, s=200)
    ax1.scatter([None], [None], marker='.', c='k', alpha=0.8, s=200, label='Training obs')

    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.legend(loc='lower right')

    grid_x = torch.tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1), dtype=torch.float32)
    with torch.no_grad():
        grid_f = torch2numpy(flow.odefunc(None, grid_x))
    ax2.streamplot(xx, yy,
                   grid_f[:, 0].reshape(xx.shape),
                   grid_f[:, 1].reshape(xx.shape),
                   color='k')

    points = test_pred.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, color='g', linestyle='solid', alpha=0.3, zorder=3)
    lc.set_linewidth(2.5)
    ax2.add_collection(lc)

    for n in range(data.tst.ys.shape[0]):
        points = data.tst.ys[n].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, color='k', linestyle='solid', alpha=1.0, zorder=4)
        lc.set_linewidth(.5)
        ax2.add_collection(lc)

    ax2.scatter(data.tst.ys[:, :, 0], data.tst.ys[:, :, 1], s=50, marker='.', c='k', alpha=0.9, zorder=4)
    ax2.plot([None], [None], color='g', linestyle='solid', alpha=0.7, label='predicted trajectory')
    ax2.plot([None], [None], color='k', linestyle='solid', marker='.', alpha=0.7, label='true trajectory')
    ax2.legend(loc='lower left')
    ax2.set_title('Learned vectorfield')
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_ylim(ylim[0], ylim[1])

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, 'plt_vectorfield.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)