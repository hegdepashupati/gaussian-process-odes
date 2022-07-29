from src.misc.torch_utils import torch2numpy

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import numpy as np


def plot_model_initialization(ax, data, model, ys, ts, predictive_function, has_latents, num_eval_samples=20):
    model.flow.odefunc.diffeq.build_cache()
    grid_size = 30
    xlim = data.xlim
    ylim = data.ylim
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))
    grid_x = torch.tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1), dtype=torch.float32)

    grid_f = []
    with torch.no_grad():
        for _ in range(100):
            model.flow.odefunc.diffeq.build_cache()
            grid_f.append(model.flow.odefunc.diffeq.forward(None, grid_x))
        grid_f = torch.stack(grid_f).mean(0)

    ax.streamplot(xx, yy,
                  torch2numpy(grid_f)[:, 0].reshape(xx.shape),
                  torch2numpy(grid_f)[:, 1].reshape(xx.shape),
                  color='grey')
    ax.set_xticks([])
    ax.set_yticks([])

    pred_ys = torch2numpy(predictive_function(model, ts, num_eval_samples).mean(0))
    line_segments = LineCollection(pred_ys[0][None, ...], linestyle='solid', colors='r', alpha=1.0, zorder=1)
    ax.add_collection(line_segments)
    ax.scatter(torch2numpy(pred_ys[0, :, 0]), torch2numpy(pred_ys[0, :, 1]), marker='x', c='r', alpha=0.9, zorder=2)
    ax.scatter(torch2numpy(ys[0, :, 0]), torch2numpy(ys[0, :, 1]), marker='x', c='k', alpha=0.9, zorder=2)
    ax.scatter([], [], marker='x', c='k', label='observed ys')
    ax.scatter([], [], marker='x', c='r', label='predicted ys')

    if has_latents:
        pred_xs = []
        with torch.no_grad():
            for _ in range(50):
                pred_xs.append(model.state_distribution.sample().squeeze(0))
            pred_xs = torch.stack(pred_xs).mean(0)
        ax.scatter(torch2numpy(pred_xs[0, :, 0]), torch2numpy(pred_xs[0, :, 1]), marker='x', c='b', alpha=0.9, zorder=3)
        ax.scatter(torch2numpy(pred_xs[0, 0, 0]), torch2numpy(pred_xs[0, 0, 1]), marker='o', c='b', alpha=0.9, zorder=4)
        ax.scatter([], [], marker='x', c='b', label='latent xs (mean)')
        ax.scatter([], [], marker='o', c='b', label='latent x0 (mean)')
    else:
        pred_x0 = []
        with torch.no_grad():
            for _ in range(50):
                pred_x0.append(model.x0_distribution.sample().squeeze(0))
            pred_x0 = torch.stack(pred_x0).mean(0)
        ax.scatter(torch2numpy(pred_x0[0, 0]), torch2numpy(pred_x0[0, 1]), marker='o', c='b', alpha=0.9, zorder=4)
        ax.scatter([], [], marker='o', c='b', label='latent x0 (mean)')
    ax.legend(loc='lower right')


def plot_longitudinal(data, test_pred, noisevar):
    test_pred_mean, test_pred_postvar = test_pred.mean(0), test_pred.var(0)
    test_pred_predvar = test_pred_postvar + noisevar

    for n in range(test_pred_mean.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(8 * 2, 3 * 1))
        for d in range(2):
            axs[d].plot(data.tst.ts, test_pred_mean[0, :, d], c='r', alpha=0.7, zorder=3)
            axs[d].fill_between(data.tst.ts,
                                test_pred_mean[0, :, d] - 2 * test_pred_postvar[0, :, d] ** 0.5,
                                test_pred_mean[0, :, d] + 2 * test_pred_postvar[0, :, d] ** 0.5,
                                color='r', alpha=0.1, zorder=1, label="posterior")
            axs[d].fill_between(data.tst.ts,
                                test_pred_mean[0, :, d] - 2 * test_pred_predvar[0, :, d] ** 0.5,
                                test_pred_mean[0, :, d] + 2 * test_pred_predvar[0, :, d] ** 0.5,
                                color='b', alpha=0.1, zorder=0, label="predictive")

            axs[d].plot(data.tst.ts, data.tst.ys[0, :, d], c='k', alpha=0.7, zorder=2)
            axs[d].scatter(data.trn.ts, data.trn.ys[0, :, d], c='k', s=100, marker='.', zorder=200)
            axs[d].set_title("State {}".format(d + 1))
            axs[d].set_xlabel("Time")
            axs[d].scatter([], [], c='k', s=10, marker='.', label='train obs')
            axs[d].plot([], [], c='k', alpha=0.7, label='true')
            axs[d].plot([], [], c='r', alpha=0.7, label='predicted')
        axs[-1].legend(loc="upper right")
        fig.suptitle("Predictive posterior")
        fig.subplots_adjust(wspace=0.2, hspace=0.2)


def plot_vectorfield_posterior(model, data, test_pred):
    cmap = cm.get_cmap('bwr')

    grid_size = 30
    xlim = data.xlim
    ylim = data.ylim
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7),
                                        sharex='all', sharey='all',
                                        gridspec_kw={'width_ratios': [1, 1.25, 1]})

    grid_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
    grid_f = []
    for gx in grid_x:
        grid_f.append(data.f(gx, None))
    grid_f = np.stack(grid_f)

    ax1.streamplot(xx, yy,
                   grid_f[:, 0].reshape(xx.shape),
                   grid_f[:, 1].reshape(xx.shape),
                   color='grey')
    ax1.set_title('True vector field')
    ax1.scatter(data.trn.ys[:, :, 0], data.trn.ys[:, :, 1], marker='.', c='k', alpha=0.8, s=200)
    ax1.scatter([None], [None], marker='.', c='k', alpha=0.8, s=200, label='Training obs')

    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.legend(loc='lower right')

    grid_x = torch.tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1), dtype=torch.float32)
    grid_f = []
    with torch.no_grad():
        for _ in range(100):
            model.flow.odefunc.diffeq.build_cache()
            grid_f.append(model.flow.odefunc.diffeq.forward(None, grid_x))
        grid_f = torch2numpy(torch.stack(grid_f))
    ax2.streamplot(xx, yy,
                   grid_f.mean(0)[:, 0].reshape(xx.shape),
                   grid_f.mean(0)[:, 1].reshape(xx.shape),
                   color='k')

    cs2 = ax2.contourf(xx, yy,
                       np.log(grid_f.std(0).mean(1)).reshape(xx.shape),
                       levels=10,
                       cmap=cmap, alpha=0.6)
    fig.colorbar(cs2, ax=ax2, shrink=0.9)
    ax2.locator_params(nbins=4)
    ax2.set_title('Learned vector field')
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_ylim(ylim[0], ylim[1])

    grid_size = 12
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))
    grid_x = torch.tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1), dtype=torch.float32)
    grid_f = []
    with torch.no_grad():
        for s in range(100):
            model.flow.odefunc.diffeq.build_cache()
            grid_f.append(model.flow.odefunc.diffeq.forward(None, grid_x))
        grid_f = torch2numpy(torch.stack(grid_f))
        grid_fvar = grid_f.std((0)).mean(1)

    for s in range(10):
        ax3.quiver(xx, yy,
                   grid_f[s, :, 0].reshape(xx.shape),
                   grid_f[s, :, 1].reshape(xx.shape),
                   grid_fvar,
                   units='x', width=0.022,
                   scale=1 / 0.15,
                   zorder=2,
                   alpha=0.8,
                   cmap=cmap)

    for s in range(min(test_pred.shape[0], 10)):
        for n in range(test_pred.shape[1]):
            points = test_pred[s, n].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, color='g', linestyle='solid', alpha=0.3, zorder=3)
            lc.set_linewidth(2.5)
            ax3.add_collection(lc)

    for n in range(data.tst.ys.shape[0]):
        points = data.tst.ys[n].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, color='k', linestyle='solid', alpha=1.0, zorder=4)
        lc.set_linewidth(.5)
        ax3.add_collection(lc)

    ax3.scatter(data.tst.ys[:, :, 0], data.tst.ys[:, :, 1], s=50, marker='.', c='k', alpha=0.9, zorder=4)
    ax3.plot([None], [None], color='g', linestyle='solid', alpha=0.7, label='predicted')
    ax3.plot([None], [None], color='k', linestyle='solid', marker='.', alpha=0.7, label='true')
    ax3.scatter([None], [None], c='k', marker=r'$\longrightarrow$', s=200, label='vector field samples')
    ax3.legend(loc='lower left')
    ax3.set_title('Posterior samples')

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
