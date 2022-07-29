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


class VanderPol(object):
    def __init__(self,
                 S_train=30, T_train=6.0,
                 S_test=None, T_test=None,
                 noise_var=0.1,
                 x0=np.array([[-1.5, 2.5]]),  # np.array([[-0.5, 2.5]]),
                 mu=0.5,
                 ):
        noise_rng = np.random.RandomState(121)
        init_rng = np.random.RandomState(123)
        S_test = S_test if S_test is not None else S_train
        T_test = T_test if T_test is not None else T_train

        self.xlim = (-3.5, 3.5)
        self.ylim = (-3.5, 3.5)

        self.mu = mu
        self.x0 = x0
        self.noise_var = noise_var
        self.new_x0 = self.x0 + init_rng.normal(size=(100, 2)) * 0.2

        xs_train, ts_train = self.generate_sequence(x0=self.x0, sequence_length=S_train, T=T_train)
        xs_test, ts_test = self.generate_sequence(x0=self.x0, sequence_length=S_test, T=T_test)
        xs_new_x0, ts_new_x0 = self.generate_sequence(x0=self.new_x0, sequence_length=S_train, T=T_train)

        xs_train = xs_train + noise_rng.normal(size=xs_train.shape) * (self.noise_var ** 0.5)

        self.trn = Data(ys=xs_train, ts=ts_train)
        self.tst = Data(ys=xs_test, ts=ts_test)
        self.tst_new_x0 = Data(ys=xs_new_x0, ts=ts_new_x0)

    def generate_sequence(self, x0, sequence_length, T):
        ts = np.linspace(0, 1.0, sequence_length) * T
        xs = []
        for _x0 in x0:
            xs.append(odeint(self.f, _x0, ts))
        return np.stack(xs), ts

    def f(self, y, t):
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -y[0] + self.mu * y[1] * (1 - y[0] ** 2)
        return dy


class VanderPolNonUniform(object):
    def __init__(self,
                 S_train=25, T_train=7.0,
                 S_test=None, T_test=None,
                 noise_var=0.1,
                 x0=np.array([[-1.5, 2.5]]),
                 mu=0.5,
                 ):
        noise_rng = np.random.RandomState(121)
        ts_rng = np.random.RandomState(122)
        S_test = S_test if S_test is not None else S_train
        T_test = T_test if T_test is not None else T_train

        self.xlim = (-3.5, 3.5)
        self.ylim = (-3.5, 3.5)

        self.mu = mu
        self.x0 = x0
        self.noise_var = noise_var

        ts_train = self.generate_random_ts(sequence_length=S_train,
                                           time_range=(0.0, T_train),
                                           rng=ts_rng)
        ts_train[0] = 0.0
        ts_test = self.generate_random_ts(sequence_length=S_test,
                                          time_range=(T_train, T_test),
                                          rng=ts_rng)
        xs_train = self.generate_sequence(x0=self.x0, ts=ts_train)
        xs_test = self.generate_sequence(x0=self.x0, ts=np.insert(ts_test, 0, 0))[:, 1:]

        xs_train = xs_train + noise_rng.normal(size=xs_train.shape) * (self.noise_var ** 0.5)
        self.trn = Data(ys=xs_train, ts=ts_train)
        self.tst = Data(ys=xs_test, ts=ts_test)

    def generate_sequence(self, x0, ts):
        xs = []
        for _x0 in x0:
            xs.append(odeint(self.f, _x0, ts))
        return np.stack(xs)

    def f(self, y, t):
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -y[0] + self.mu * y[1] * (1 - y[0] ** 2)
        return dy

    def generate_random_ts(self, sequence_length, time_range, rng):
        ts = np.sort(rng.random_sample(sequence_length)) * (time_range[1] - time_range[0]) + time_range[0]
        return ts


def plot_vanderpol(data):
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 2.5))
    for d, ax in enumerate(axs.flat):
        for n in range(data.trn.ys.shape[0]):
            ax.scatter(data.trn.ts, data.trn.ys[n, :, d], c="k")

    for d, ax in enumerate(axs.flat):
        for n in range(data.tst.ys.shape[0]):
            ax.plot(data.tst.ts, data.tst.ys[n, :, d], c="r")
    axs[-1].scatter([], [], c="k", label="observations")
    axs[-1].plot([], [], c="r", label="test sequence")
    axs[-1].set_xlabel("Time")
    axs[0].set_title("State 1")
    axs[1].set_title("State 2")
    axs[-1].legend()
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
    ax3.set_title("True vectorfield")
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()
