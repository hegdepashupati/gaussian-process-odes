from src.misc.param import Param
from src.misc import transforms

import numpy as np
import torch
from torch import nn

from torch.distributions import MultivariateNormal

initial_state_scale = 1e-1
jitter = 1e-5


def sample_normal(shape, seed=None):
    # sample from standard Normal with a given shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.normal(size=shape).astype(np.float32))
    else:
        return torch.tensor(np.random.normal(size=shape).astype(np.float32))


class StateInitialDistribution(nn.Module):
    """
    Base class defining Initial state posterior q(x_0)
    """

    def __init__(self, dim_n, dim_d):
        super(StateInitialDistribution, self).__init__()
        self.dim_n = dim_n
        self.dim_d = dim_d

    def _initialize(self, x):
        raise NotImplementedError

    def sample(self, num_samples, seed=None):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    def kl(self):
        raise NotImplementedError


class StateInitialVariationalGaussian(StateInitialDistribution):
    """
    Full rank multivariate Gaussian approximation for the Initial state posterior q(x_0) = N(m, S)
        where x is (N,D),  m is (N,D), S is (N,D,D)

    N being the number of sequences, D being the number of state dimensions.
    """

    def __init__(self, dim_n, dim_d):
        """
        @param dim_n: N number of sequences
        @param dim_d: D state dimensionality
        """
        super(StateInitialVariationalGaussian, self).__init__(dim_n, dim_d)
        self.param_mean = Param(np.random.normal(size=(dim_n, dim_d)) * 1e-2,
                                name='Initial state distribution (mean)')

        self.param_lchol = Param(np.stack([np.eye(dim_d)] * dim_n) * initial_state_scale,  # NxDxD
                                 transform=transforms.LowerTriangular(dim_d, dim_n),
                                 name='Initial state distribution (scale)')

    def _initialize(self, x):
        self.param_mean().data = x

    def mean(self):
        return self.param_mean()

    def lchol(self):
        return self.param_lchol()

    def distribution(self):
        x0_mean = self.mean()
        x0_lchol = self.lchol()
        x0_qcov = torch.einsum('nij, nkj -> nik', x0_lchol, x0_lchol)
        x0_qcov = x0_qcov + torch.eye(x0_qcov.shape[-1]).unsqueeze(0) * jitter
        dist = MultivariateNormal(loc=x0_mean, covariance_matrix=x0_qcov)
        return dist

    def sample_numpy(self, num_samples=1, seed=None):
        x0_mean = self.mean().unsqueeze(0)
        x0_lchol = self.lchol()
        epsilon = sample_normal(shape=(num_samples, self.dim_n, self.dim_d), seed=seed)  # (S,N,D)
        zs = torch.einsum('nij, snj -> sni', x0_lchol, epsilon)
        return zs + x0_mean  # (S,N,D)

    def sample(self, num_samples=1, seed=None):
        return self.distribution().rsample((num_samples,))

    def log_prob(self, x):
        return self.distribution().log_prob(x)

    def kl(self):
        alpha = self.mean()  # NxD
        Lq = torch.tril(self.lchol())  # force lower triangle # NxDxD
        Lq_diag = torch.diagonal(Lq, dim1=1, dim2=2)  # NxD

        # Mahalanobis term: μqᵀ Σp⁻¹ μq
        mahalanobis = torch.pow(alpha, 2).sum(dim=1, keepdim=True)  # Nx1

        # Log-determinant of the covariance of q(x):
        logdet_qcov = torch.log(torch.pow(Lq_diag, 2)).sum(dim=1, keepdim=True)  # Nx1

        # Trace term: tr(Σp⁻¹ Σq)
        trace = torch.pow(Lq, 2).sum(dim=(1, 2)).unsqueeze(1)  # NxDxD --> Nx1
        logdet_pcov = 0.0
        constant = - torch.tensor(self.dim_d)
        twoKL = logdet_pcov - logdet_qcov + mahalanobis + trace + constant
        kl = 0.5 * twoKL.sum()
        return kl  # Nx1


class StateSequenceVariationalDistribution(nn.Module):
    """
    Base class defining state sequence posterior
    """

    def __init__(self, dim_n, dim_t, dim_d):
        super(StateSequenceVariationalDistribution, self).__init__()
        self.dim_n = dim_n
        self.dim_t = dim_t
        self.dim_d = dim_d

    def _add_intial_state(self):
        raise NotImplementedError

    def _initialize(self, x0, xs):
        raise NotImplementedError

    def sample(self, num_samples, **kwargs):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class StateSequenceVariationalFactorizedGaussian(StateSequenceVariationalDistribution):
    """
    Full rank multivariate Gaussian approximation for the state sequence posterior q(x_s) = N(m, S)
        where x_s is (N,T,D),  m is (N,T,D), S is (N,T,D,D)

    N is the number of sequences, T is the sequence length, D being the number of state dimensions.
    """

    def __init__(self, dim_n, dim_t, dim_d):
        """
        @param dim_n: N number of sequences
        @param dim_n: T sequence length
        @param dim_d: D state dimensionality
        """
        super(StateSequenceVariationalFactorizedGaussian, self).__init__(dim_n, dim_t, dim_d)
        self.param_mean = Param(np.random.normal(size=(self.dim_n, self.dim_t, self.dim_d)) * 1e-1,
                                name='State distribution (mean)')  # (N,T,D)

        self.param_lchol = Param(
            np.stack([np.stack([np.eye(self.dim_d)] * self.dim_t)] * self.dim_n) * initial_state_scale,
            # (N,T,D,D)
            transform=transforms.StackedLowerTriangular(self.dim_d, self.dim_n, self.dim_t),
            name='State distribution (scale)')
        self._add_initial_state()

    def _add_initial_state(self):
        self.x0 = StateInitialVariationalGaussian(self.dim_n, self.dim_d)

    def _initialize(self, x0, xs, xs_std=None):
        self.x0._initialize(x0)
        self.param_mean().data = xs
        if xs_std is not None:
            self.param_scale.optvar.data = self.param_lchol.transform.backward_tensor(torch.diag_embed(xs_std)).data

    def mean(self):
        return self.param_mean()

    def lchol(self):
        return self.param_lchol()

    def distribution(self):
        xs_mean = self.mean()
        xs_lchol = self.lchol()
        xs_qcov = torch.einsum('ntij, ntkj -> ntik', xs_lchol, xs_lchol)
        xs_qcov = xs_qcov + torch.eye(xs_qcov.shape[-1]).unsqueeze(0).unsqueeze(0) * jitter
        dist = MultivariateNormal(loc=xs_mean, covariance_matrix=xs_qcov)
        return dist

    def sample_numpy(self, num_samples=1, seed=None):
        # append sample from the initial state distribution
        epsilon = sample_normal(shape=(num_samples, self.dim_n, self.dim_t, self.dim_d), seed=seed)  # (S,N,T,D)
        zs = torch.einsum('ntij, sntj->snti', self.lchol(), epsilon)
        return torch.cat([self.x0.sample(num_samples, seed).unsqueeze(2), zs + self.mean().unsqueeze(0)],
                         2)  # (S, N, T+1, D)

    def sample(self, num_samples=1, seed=None):
        return torch.cat([self.x0.sample(num_samples).unsqueeze(2),
                          self.distribution().rsample((num_samples,))], 2)  # (S, N, T+1, D)

    def entropy(self):
        return self.distribution().entropy()

    def log_prob(self, x):
        return self.distribution().log_prob(x)
