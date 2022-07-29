from src.misc.torch_utils import torch2numpy

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


def plot_pca_predictions(actual, predicted, ts, args, num_obs=10, name='plt_latents'):
    num_obs = np.minimum(actual.shape[0], num_obs)
    idx = np.random.permutation(actual.shape[0])
    actual = actual[idx]
    predicted = predicted[:, idx]

    pred_mean, pred_std = predicted.mean(0), predicted.std(0)
    num_obs = min(actual.shape[0], num_obs)
    for n in range(num_obs):
        fig, axs = plt.subplots(1, actual.shape[-1], figsize=(4 * actual.shape[-1], 3))
        for j in range(actual.shape[-1]):
            axs[j].scatter(torch2numpy(ts)[:actual[n, :, j].shape[0]], torch2numpy(actual[n, :, j]),
                           c='k', marker='.', s=10, zorder=0, alpha=0.9)
            axs[j].plot(torch2numpy(ts), torch2numpy(pred_mean[n, :, j]),
                        c='r', alpha=0.7, zorder=3)
            axs[j].fill_between(torch2numpy(ts),
                                torch2numpy(pred_mean[n, :, j]) - 2 * torch2numpy(pred_std[n, :, j]),
                                torch2numpy(pred_mean[n, :, j]) + 2 * torch2numpy(pred_std[n, :, j]),
                                color='r', alpha=0.2, zorder=2)
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
    predicted = predicted[:, idx]

    pred_mean, pred_std = predicted.mean(0), predicted.std(0)
    num_obs = min(actual.shape[0], num_obs)
    for n in range(num_obs):
        fig, axs = plt.subplots(10, 5, figsize=(4 * 5, 3 * 10))
        count = 0
        for i in range(10):
            for j in range(5):
                axs[i, j].scatter(torch2numpy(ts)[:actual[n, :, count].shape[0]], torch2numpy(actual[n, :, count]),
                                  c='k', marker='.', s=10, zorder=0, alpha=0.9)
                axs[i, j].plot(torch2numpy(ts), torch2numpy(pred_mean[n, :, count]),
                               c='r', alpha=0.7, zorder=3)
                axs[i, j].fill_between(torch2numpy(ts),
                                       torch2numpy(pred_mean[n, :, count]) - 2 * torch2numpy(pred_std[n, :, count]),
                                       torch2numpy(pred_mean[n, :, count]) + 2 * torch2numpy(pred_std[n, :, count]),
                                       color='r', alpha=0.2, zorder=2)
                count += 1
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        if name is None:
            plt.show()
        else:
            fig.savefig(os.path.join(args.save, name + "_{}.png".format(n)), dpi=160,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)


def plot_latents_3d(sampled_zs, ts, args, num_obs=10, name='plt_latents_3d'):
    num_obs = np.minimum(sampled_zs.shape[1], num_obs)
    idx = np.random.permutation(sampled_zs.shape[1])
    sampled_zs = sampled_zs[:, idx]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    cmap = 'gist_rainbow'
    norm = colors.Normalize(vmin=ts.min(), vmax=ts.max())
    for n in range(num_obs):
        for s in range(sampled_zs.shape[0]):
            points = np.array([sampled_zs[s, n, :, 0],
                               sampled_zs[s, n, :, 1],
                               sampled_zs[s, n, :, 2]]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = Line3DCollection(segments, cmap=cmap, alpha=0.4, norm=norm)
            lc.set_array(ts)
            lc.set_linewidth(2)
            ax.add_collection(lc)

            ax.scatter(sampled_zs[s, n, :, 0], sampled_zs[s, n, :, 1], sampled_zs[s, n, :, 2],
                       c='k', marker='.', s=20, zorder=3)

        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        ax.set_zlabel("Comp 3")
        if name is None:
            plt.show()
        else:
            fig.savefig(os.path.join(args.save, name + ".png"), dpi=160,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add a 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def plot_inducing_posterior_3d(pred, ts, u, z, args, name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    cmap = 'gist_rainbow'
    norm = mpl.colors.Normalize(vmin=ts.min(), vmax=ts.max())
    for n in range(min(pred.shape[1], 10)):
        for s in range(pred.shape[0]):
            points = np.array([pred[s, n, :, 0], pred[s, n, :, 1], pred[s, n, :, 2]]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = Line3DCollection(segments, cmap=cmap, alpha=0.4, norm=norm)
            lc.set_array(ts)
            lc.set_linewidth(2)
            ax.add_collection(lc)

            ax.scatter(pred[s, n, :, 0], pred[s, n, :, 1], pred[s, n, :, 2], c='r', marker='.', s=20, zorder=3)
        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        ax.set_zlabel("Comp 3")

    ax.scatter(z[:, 0].flatten(), z[:, 1].flatten(), z[:, 2].flatten(), marker='o', c='b', s=10, alpha=0.01)

    for m in range(z.shape[0]):
        ax.arrow3D(z[m, 0], z[m, 1], z[m, 2], u[m, 0], u[m, 1], u[m, 2],
                   mutation_scale=20, ec='k', fc='k', alpha=0.2)

    if name is None:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, name + '.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)


def plot_trace(loss_meter, observ_nll_meter, inducing_kl_meter, args, make_plot=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

    ax1.plot(loss_meter.iters,
             loss_meter.vals)
    ax1.set_title("Loss function")
    ax2.plot(observ_nll_meter.iters,
             observ_nll_meter.vals)
    ax2.set_title("Observation NLL")
    ax3.plot(inducing_kl_meter.iters,
             inducing_kl_meter.vals)
    ax3.set_title("Inducing KL")
    fig.subplots_adjust()
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, 'optimization_trace.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)
