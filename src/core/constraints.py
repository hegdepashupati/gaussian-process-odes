from src.misc.constraint_utils import softplus, invsoftplus

import torch
import torch.nn as nn
from torch.nn import init
from torch import distributions


class Gaussian(nn.Module):
    """
    Gaussian with an optionally trainable scale parameter
    """

    def __init__(self, d: int = 1, scale: float = 1.0, requires_grad: bool = True) -> None:
        super(Gaussian, self).__init__()
        self.unconstrained_scale = torch.nn.Parameter(torch.ones(d), requires_grad=requires_grad)
        self._initialize(scale)

    def _initialize(self, x: float) -> None:
        init.constant_(self.unconstrained_scale, invsoftplus(torch.tensor(x)).item())

    @property
    def scale(self):
        return softplus(self.unconstrained_scale)

    def distribution(self, loc: torch.Tensor) -> torch.distributions.Distribution:
        return distributions.Normal(loc=loc, scale=self.scale)

    @property
    def variance(self):
        return self.distribution(loc=torch.zeros_like(self.scale)).variance

    def log_prob(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution(loc=f).log_prob(y)
        assert (log_prob.shape == f.shape)
        return log_prob


class Laplace(nn.Module):
    """
    Laplace with an optionally trainable scale parameter
    """

    def __init__(self, d: int = 1, scale: float = 1.0, requires_grad: bool = True) -> None:
        super(Laplace, self).__init__()
        self.unconstrained_scale = torch.nn.Parameter(torch.ones(d), requires_grad=requires_grad)
        self._initialize(scale)

    def _initialize(self, x: float) -> None:
        init.constant_(self.unconstrained_scale, invsoftplus(torch.tensor(x)).item())

    @property
    def scale(self):
        return softplus(self.unconstrained_scale)

    def distribution(self, loc):
        return distributions.Laplace(loc=loc, scale=self.scale)

    @property
    def variance(self):
        return self.distribution(loc=torch.zeros_like(self.scale)).variance

    def log_prob(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution(loc=f).log_prob(y)
        assert (log_prob.shape == f.shape)
        return log_prob
