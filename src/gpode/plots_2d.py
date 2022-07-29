from src.misc.torch_utils import torch2numpy, insert_zero_t0

import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm


def plot_model_initialization(model, data, args, name):
    model.flow.odefunc.diffeq.build_cache()
    grid_size = 30
    xlim = data.xlim
    ylim = data.ylim
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))
    grid_x = torch.tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1), dtype=torch.float32)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    grid_f = []
    for _ in range(100):
        with torch.no_grad():
            model.flow.odefunc.diffeq.build_cache()
            grid_f.append(model.flow.odefunc.diffeq.forward(None, grid_x))
    grid_f = torch.stack(grid_f).mean(0)

    ax.streamplot(xx, yy,
                  torch2numpy(grid_f)[:, 0].reshape(xx.shape),
                  torch2numpy(grid_f)[:, 1].reshape(xx.shape),
                  color='grey')
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])

    pred_ys = []
    ts = torch.tensor(data.trn.ts)
    ts = insert_zero_t0(ts)
    with torch.no_grad():
        for _ in range(args.eval_sample_size):
            with torch.no_grad():
                pred_ys.append(model(model.x0_distribution.sample()[0], ts))
        pred_ys = torch.stack(pred_ys).mean(0)

    line_segments = LineCollection(torch2numpy(pred_ys[0].unsqueeze(0)), linestyle='solid', colors='r', alpha=1.0,
                                   zorder=1)
    ax.add_collection(line_segments)
    ax.scatter(torch2numpy(pred_ys[0, :, 0]), torch2numpy(pred_ys[0, :, 1]), marker='x', c='r', alpha=0.9, zorder=2,
               label='ys')
    ax.scatter(data.trn.ys[0, :, 0], data.trn.ys[0, :, 1], marker='x', c='k', alpha=0.9, zorder=2, label='obs')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(args.save, name), dpi=160,
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def plot_longitudinal(data, test_pred, noisevar, args, make_plot=False):
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
            axs[d].plot([], [], c='k', alpha=0.7, label='true trajectory')
            axs[d].plot([], [], c='r', alpha=0.7, label='predictive trajectory')
        axs[-1].legend(loc="upper right")
        fig.suptitle("Predictive posterior for GPODE")
        fig.subplots_adjust(wspace=0.2, hspace=0.2)

        if make_plot:
            plt.show()
        else:
            fig.savefig(os.path.join(args.save, 'plt_longitudinal_{}.png'.format(n)), dpi=160,
                        bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)


def plot_vectorfield(model, data, test_pred, args):
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
    ax1.set_title('True vectorfield')
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
    ax2.set_title('Learned vectorfield')
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
    ax3.plot([None], [None], color='g', linestyle='solid', alpha=0.7, label='predictive samples')
    ax3.plot([None], [None], color='k', linestyle='solid', marker='.', alpha=0.7, label='true trajectory')
    ax3.scatter([None], [None], c='k', marker=r'$\longrightarrow$', s=200, label='vectorfield samples')
    ax3.legend(loc='lower left')
    ax3.set_title('Predictive samples')

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(os.path.join(args.save, 'plt_vectorfield.png'), dpi=160,
                bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def plot_long_pred(data, pred, ts, args, name):
    fig, axs = plt.subplots(1, 2, figsize=(16, 3), sharex='all')
    for j in range(2):
        axs[j].scatter(torch2numpy(ts), data[0, :, j], c='k', s=10, marker='.',
                       zorder=200)
        pred_mean, pred_postvar = pred.mean(0), pred.var(0)
        axs[j].plot(torch2numpy(ts), pred_mean[0, :, j].T, c='r', alpha=0.5)
        axs[j].fill_between(torch2numpy(ts),
                            pred_mean[0, :, j] - 2 * pred_postvar[0, :, j] ** 0.5,
                            pred_mean[0, :, j] + 2 * pred_postvar[0, :, j] ** 0.5,
                            color='r', alpha=0.1, zorder=1, label="posterior")
        axs[j].set_title("state {}".format(j + 1))
        axs[j].set_xlabel("Time")
    axs[1].scatter([], [], c='k', s=10, marker='.', zorder=200, label='actual')
    axs[1].plot([], [], c='r', alpha=0.2, label='predictive mean')
    axs[1].legend(loc="lower left")
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(os.path.join(args.save, name), dpi=160,
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def plot_longnoise_pred(data, pred, noise_var, ts, args, name):
    fig, axs = plt.subplots(1, 2, figsize=(16, 3), sharex='all')
    for j in range(2):
        axs[j].scatter(torch2numpy(ts), data[0, :, j], c='k', s=10, marker='.',
                       zorder=200)
        pred_mean, pred_postvar = pred.mean(0), pred.var(0)
        pred_predvar = pred_postvar + noise_var
        axs[j].plot(torch2numpy(ts), pred_mean[0, :, j].T, c='r', alpha=0.5)
        axs[j].fill_between(torch2numpy(ts),
                            pred_mean[0, :, j] - 2 * pred_postvar[0, :, j] ** 0.5,
                            pred_mean[0, :, j] + 2 * pred_postvar[0, :, j] ** 0.5,
                            color='r', alpha=0.1, zorder=1, label="posterior")
        axs[j].fill_between(torch2numpy(ts),
                            pred_mean[0, :, j] - 2 * pred_predvar[0, :, j] ** 0.5,
                            pred_mean[0, :, j] + 2 * pred_predvar[0, :, j] ** 0.5,
                            color='b', alpha=0.1, zorder=0, label="predictive")
        axs[j].set_title("state {}".format(j + 1))
        axs[j].set_xlabel("Time")
    axs[1].scatter([], [], c='k', s=10, marker='.', zorder=200, label='actual')
    axs[1].plot([], [], c='r', alpha=0.2, label='predictive mean')
    axs[1].legend(loc="lower left")
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(os.path.join(args.save, name), dpi=160,
                bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def plot_inducing_posterior(model, data, args):
    grid_size = 30
    xlim = data.xlim
    ylim = data.ylim
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))
    grid_x = torch.tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1), dtype=torch.float32)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    grid_f = []
    for _ in range(100):
        model.flow.odefunc.diffeq.build_cache()
        grid_f.append(model.flow.odefunc.diffeq.forward(None, grid_x))
    grid_f = torch.stack(grid_f).mean(0)

    ax.streamplot(xx, yy,
                  torch2numpy(grid_f)[:, 0].reshape(xx.shape),
                  torch2numpy(grid_f)[:, 1].reshape(xx.shape),
                  color='grey')
    with torch.no_grad():
        z = model.flow.odefunc.diffeq.inducing_loc().clone().detach()
        u = model.flow.odefunc.diffeq.Um().clone().detach()
        Ku = model.flow.odefunc.diffeq.kern.K(model.flow.odefunc.diffeq.inducing_loc())  # MxM or DxMxM
        Lu = torch.cholesky(Ku + torch.eye(model.flow.odefunc.diffeq.M) * 1e-5)  # MxM or DxMxM
        u = torch.einsum('md, dmn -> nd', u, Lu.permute(0, 2, 1)) if args.dimwise else torch.einsum('md, mn -> nd', u,
                                                                                                    Lu.T)  # NxD
    u = torch2numpy(u)
    z = torch2numpy(z)

    for m in range(z.shape[0]):
        ax.arrow(z[m, 0], z[m, 1], u[m, 0], u[m, 1], width=0.02, head_width=0.2, color='k', zorder=100, alpha=0.7)

    ax.scatter(data.trn.ys[:, :, 0], data.trn.ys[:, :, 1], marker='.', c='r', alpha=0.3)
    ax.plot([None], [None], ls='-', c='r', alpha=0.9, label='Observed traj')

    line_segments = LineCollection(data.trn.ys, linestyle='solid', colors='r', alpha=0.3)
    ax.add_collection(line_segments)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc='lower right')
    fig.subplots_adjust()
    fig.savefig(os.path.join(args.save, 'vectorfield.png'), dpi=160,
                bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def plot_trace(loss_meter, observ_nll_meter, inducing_kl_meter, args):
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
    fig.savefig(os.path.join(args.save, 'optimization_trace.png'), dpi=160,
                bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
