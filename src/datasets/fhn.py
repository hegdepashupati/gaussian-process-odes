import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Data:
    def __init__(self, ys, ts):
        self.ts = ts.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.ys[index, ...], self.ts


class FHN(object):
    def __init__(self,
                 S_train=30, T_train=6.0,
                 S_test=None, T_test=None,
                 noise_var=0.1,
                 x0=np.array([[-1., -1.]]),
                 ):
        noise_rng = np.random.RandomState(121)
        S_test = S_test if S_test is not None else S_train * 2
        T_test = T_test if T_test is not None else T_train * 2.0

        self.xlim = (-2.5, 2.5)
        self.ylim = (-2., 2.)

        self.x0 = x0
        self.noise_var = noise_var

        xs_train, ts_train = self.generate_sequence(x0=self.x0, sequence_length=S_train, T=T_train)
        xs_test, ts_test = self.generate_sequence(x0=self.x0, sequence_length=S_test, T=T_test)

        xs_train = xs_train + noise_rng.normal(size=xs_train.shape) * (self.noise_var ** 0.5)

        self.trn = Data(ys=xs_train, ts=ts_train)
        self.tst = Data(ys=xs_test, ts=ts_test)

    def generate_sequence(self, x0, sequence_length, T):
        ts = np.linspace(0, 1.0, sequence_length) * T
        xs = []
        for _x0 in x0:
            xs.append(odeint(self.f, _x0, ts))
        return np.stack(xs), ts

    def f(self, y, t):
        dy = np.zeros(2)
        dy[0] = 3.0 * (y[0] - (y[0] ** 3 / 3.0) + y[1])
        dy[1] = (1.0 / 3.0) * (0.2 - (3.0 * y[0]) - (0.2 * y[1]))
        return dy


def plot_fhn(data):
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 2.5))
    for d, ax in enumerate(axs.flat):
        for n in range(data.trn.ys.shape[0]):
            ax.scatter(data.trn.ts, data.trn.xs[n, :, d], c="k")

    for d, ax in enumerate(axs.flat):
        for n in range(data.tst.ys.shape[0]):
            ax.plot(data.tst.ts, data.tst.ys[n, :, d], c="r")
    axs[0].scatter([], [], c="k", label="observations")
    axs[0].plot([], [], c="r", label="test sequence")
    axs[0].set_xlabel("Time")
    axs[0].set_title("State 1")
    axs[1].set_title("State 2")
    axs[0].legend()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4 * 3, 4), sharex="all", sharey="all")
    ax1.scatter(data.trn.ys[:, :, 0], data.trn.ys[:, :, 1], marker=".", c="k", alpha=1.0)
    ax1.scatter([None], [None], marker=".", c="k", alpha=1.0, label="observations")
    line_segments = LineCollection(data.trn.ys, linestyle="solid", colors="r", alpha=0.4)
    ax1.add_collection(line_segments)
    ax1.set_title("Train sequences")
    ax1.set_xlabel("State 1")
    ax1.set_ylabel("State 2")
    ax1.legend()

    ax2.scatter(data.tst.ys[:, :, 0], data.tst.ys[:, :, 1], marker=".", c="r", alpha=0.9)
    line_segments = LineCollection(data.tst.ys, linestyle="solid", colors="r", alpha=0.4)
    ax2.add_collection(line_segments)
    ax2.set_title("Test sequence")

    grid_size = 30
    xlim = data.xlim
    ylim = data.ylim
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_size),
                         np.linspace(ylim[0], ylim[1], grid_size))
    grid_x = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    grid_drift = []
    for gx in grid_x:
        grid_drift.append(data.f(gx, None))
    grid_drift = np.stack(grid_drift)

    ax3.streamplot(xx, yy,
                   grid_drift[:, 0].reshape(xx.shape),
                   grid_drift[:, 1].reshape(xx.shape),
                   color="grey")
    ax3.set_title("True FHN")
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
