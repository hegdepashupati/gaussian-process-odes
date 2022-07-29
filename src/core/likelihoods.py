from src.misc.constraint_utils import softplus, invsoftplus

import torch
import torch.nn as nn
from torch.nn import init

import numpy as np


class Gaussian(nn.Module):
    """
    Gaussian likelihood
    """

    def __init__(self, ndim=1, init_val=0.25):
        super(Gaussian, self).__init__()
        self.unconstrained_variance = torch.nn.Parameter(torch.ones(ndim), requires_grad=True)
        self._initialize(init_val)

    def _initialize(self, x):
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(x)).item())

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)

    def log_prob(self, F, Y):
        return -0.5 * (np.log(2.0 * np.pi) + torch.log(self.variance) + torch.pow(F - Y, 2) / self.variance)


class ProjectedGaussian(Gaussian):
    """
    Gaussian likelihood with a projection from latent space to observation space
    """

    def __init__(self, projection, ndim=1, init_val=0.25):
        super(ProjectedGaussian, self).__init__(ndim, init_val)
        self.projection = projection

    def log_prob(self, F, Y):
        if F.ndim == 4:
            F = torch.stack([self.projection(_F) for _F in F])
        else:
            F = self.projection(F)
        return -0.5 * (np.log(2.0 * np.pi) + torch.log(self.variance) + torch.pow(F - Y, 2) / self.variance)
