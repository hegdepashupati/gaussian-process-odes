from src.misc.settings import settings

import torch
import numpy as np

from torch.utils.data import Dataset

device = settings.device
dtype = settings.torch_float


class Latent2DataProjector:
    """
    Defines latent/PCA space to observation space transformations
    """

    def __init__(self, dataset):
        self.data_std = torch.tensor(dataset.data_std.astype(np.float32)).to(device)
        self.data_mean = torch.tensor(dataset.data_mean.astype(np.float32)).to(device)

        if dataset.pca_normalize is not None:
            self.pca_normalize_mean = torch.tensor(dataset.pca_normalize.mean.astype(np.float32)).to(device)
            self.pca_normalize_std = torch.tensor(dataset.pca_normalize.std.astype(np.float32)).to(device)
            self.inverse_pca_normalization = lambda x: (x * self.pca_normalize_std) + self.pca_normalize_mean
        else:
            self.inverse_pca_normalization = lambda x: x

        self.pca_components = torch.tensor(dataset.pca.components_.astype(np.float32)).to(device)
        self.inverse_pca = lambda x: torch.einsum('ntl,ld->ntd', x, self.pca_components)

    def __call__(self, x):
        x = self.inverse_pca_normalization(x)
        x = self.inverse_pca(x)
        return x


class CombinedDataset(Dataset):
    def __init__(self, data_pca, data_full):
        self.data_pca = data_pca
        self.data_full = data_full

    def __len__(self):
        return self.data_pca.ys.shape[0]

    def __getitem__(self, index):
        return self.data_full.ys[index, ...], self.data_pca.ys[index, ...], self.data_pca.ts
