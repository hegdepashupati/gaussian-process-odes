import numpy as np
import os
from sklearn.decomposition import PCA


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return (x * self.std) + self.mean


class Data:
    def __init__(self, ys, ts):
        self.ts = ts.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, index):
        return self.ys[index, ...], self.ts


class MocapDataset(object):
    def __init__(self, data_path="data/mocap/",
                 subject="09",
                 dt=0.01,
                 pca_components=-1,
                 seqlen=50,
                 data_normalize=False,
                 pca_normalize=True):
        self.data_path = data_path
        self.dt = dt
        self.pca_components = pca_components
        self.data_normalize = data_normalize
        self.pca_normalize = pca_normalize

        assert subject in ["09", "35", "39"], "Wrong subject passed"
        fname = os.path.join(data_path, "mocap" + subject + ".npz")
        mocap_data = np.load(fname)

        xs_test = mocap_data["test"]
        ts_test = dt * np.arange(0, xs_test.shape[1])

        xs_valid = mocap_data["validation"]
        ts_valid = dt * np.arange(0, xs_valid.shape[1])

        xs_train = mocap_data["train"]
        ts_train = dt * np.arange(0, xs_train.shape[1])

        xs_train = self.treat_zero_readings(xs_train)
        xs_valid = self.treat_zero_readings(xs_valid)
        xs_test = self.treat_zero_readings(xs_test)

        self.data_std = xs_train[:, :].std((0, 1), keepdims=True) + 1e-5
        self.data_mean = xs_train[:, :].mean((0, 1), keepdims=True)
        if data_normalize:
            self.data_normalize = Normalize(self.data_mean, self.data_std)
            xs_train, xs_valid, xs_test = self.data_normalize(xs_train), self.data_normalize(
                xs_valid), self.data_normalize(xs_test)
        else:
            self.data_normalize = None

        if pca_components > 0:
            xs_train = self.build_pca(xs_train, train=True)
            xs_valid = self.build_pca(xs_valid, train=False)
            xs_test = self.build_pca(xs_test, train=False)

        if pca_normalize:
            pca_m = xs_train[:, :].mean((0, 1), keepdims=True)
            pca_s = xs_train[:, :].std((0, 1), keepdims=True) + 1e-5
            self.pca_normalize = Normalize(pca_m, pca_s)
            xs_train, xs_valid, xs_test = self.pca_normalize(xs_train), self.pca_normalize(
                xs_valid), self.pca_normalize(xs_test)
        else:
            self.pca_normalize = None

        self.trn = Data(ys=xs_train[:, :seqlen], ts=ts_train[:seqlen])
        self.val = Data(ys=xs_valid, ts=ts_valid)
        self.tst = Data(ys=xs_test, ts=ts_test)

    def treat_zero_readings(self, data):
        data[:, :, (24, 25, 31, 32)] = 1e-6
        return data

    def build_pca(self, x, train=False):
        N, T, D = x.shape
        x_stacked = np.vstack([x[i] for i in range(x.shape[0])])
        if train:
            self.pca = PCA(n_components=self.pca_components)
            x_ = self.pca.fit_transform(x_stacked)
        else:
            x_ = self.pca.transform(x_stacked)
        x_ = np.concatenate([np.expand_dims(x_[i * T:(i + 1) * T], 0) for i in range(N)], 0)
        return x_
