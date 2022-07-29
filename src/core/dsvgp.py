from src.misc.param import Param
from src.misc import transforms
from src.core.kernels import RBF

import numpy as np
import torch

jitter = 1e-5


def sample_normal(shape, seed=None):
    # sample from standard Normal with a given shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.normal(size=shape).astype(np.float32))
    else:
        return torch.tensor(np.random.normal(size=shape).astype(np.float32))


def sample_uniform(shape, seed=None):
    # random Uniform sample of a given shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.uniform(low=0.0, high=1.0, size=shape).astype(np.float32))
    else:
        return torch.tensor(np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32))


class DSVGP_Layer(torch.nn.Module):
    """
    A layer class implementing decoupled sampling of SVGP posterior
     @InProceedings{pmlr-v119-wilson20a,
                    title = {Efficiently sampling functions from {G}aussian process posteriors},
                    author = {Wilson, James and Borovitskiy, Viacheslav and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc},
                    booktitle = {Proceedings of the 37th International Conference on Machine Learning},
                    pages = {10292--10302},
                    year = {2020},
                    editor = {Hal Daumé III and Aarti Singh},
                    volume = {119},
                    series = {Proceedings of Machine Learning Research},
                     publisher = {PMLR},
                     pdf = {http://proceedings.mlr.press/v119/wilson20a/wilson20a.pdf}
                     }
    """

    def __init__(self, D_in, D_out, M, S, q_diag=False, dimwise=True):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param M: Number of inducing points
        @param S: Number of features to consider for Fourier feature maps
        @param q_diag: Diagonal approximation for inducing posterior
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(DSVGP_Layer, self).__init__()

        self.kern = RBF(D_in, D_out, dimwise)
        self.q_diag = q_diag
        self.dimwise = dimwise

        self.D_out = D_out
        self.D_in = D_in
        self.M = M
        self.S = S

        self.inducing_loc = Param(np.random.normal(size=(M, D_in)), name='Inducing locations')  # (M,D_in)
        self.Um = Param(np.random.normal(size=(M, D_out)) * 1e-1, name='Inducing distribution (mean)')  # (M,D_out)

        if self.q_diag:
            self.Us_sqrt = Param(np.ones(shape=(M, D_out)) * 1e-3,  # (M,D_out)
                                 transform=transforms.SoftPlus(),
                                 name='Inducing distribution (scale)')
        else:
            self.Us_sqrt = Param(np.stack([np.eye(M)] * D_out) * 1e-3,  # (D_out,M,M)
                                 transform=transforms.LowerTriangular(M, D_out),
                                 name='Inducing distribution (scale)')

    def sample_inducing(self):
        """
        Generate a sample from the inducing posterior q(u) ~ N(m, S)
        @return: inducing sample (M,D) tensor
        """
        epsilon = sample_normal(shape=(self.M, self.D_out), seed=None)  # (M, D_out)
        if self.q_diag:
            ZS = self.Us_sqrt() * epsilon  # (M, D_out)
        else:
            ZS = torch.einsum('dnm, md->nd', self.Us_sqrt(), epsilon)  # (M, D_out)

        u_sample = ZS + self.Um()  # (M, D_out)
        return u_sample  # (M, D_out)

    def build_cache(self):
        """
        Builds a cache of computations that uniquely define a sample from posterior process
        1. Generate and fix parameters of Fourier features
        2. Generate and fix inducing posterior sample
        3. Intermediate computations based on the inducing sample for pathwise update
        """
        # generate parameters required for the Fourier feature maps
        self.rff_weights = sample_normal((self.S, self.D_out))  # (S,D_out)
        self.rff_omega = self.kern.sample_freq(self.S)  # (D_in,S) or (D_in,S,D_out)
        phase_shape = (1, self.S, self.D_out) if self.dimwise else (1, self.S)
        self.rff_phase = sample_uniform(phase_shape) * 2 * np.pi  # (S,D_out)

        # generate sample from the inducing posterior
        inducing_val = self.sample_inducing()  # (M,D)

        # compute th term nu = k(Z,Z)^{-1}(u-f(Z)) in whitened form of inducing variables
        # equation (13) from http://proceedings.mlr.press/v119/wilson20a/wilson20a.pdf
        Ku = self.kern.K(self.inducing_loc())  # (M,M) or (D,M,M)
        Lu = torch.cholesky(Ku + torch.eye(self.M) * jitter)  # (M,M) or (D,M,M)
        u_prior = self.rff_forward(self.inducing_loc())  # (M,D)

        if not self.dimwise:
            nu = torch.triangular_solve(u_prior, Lu, upper=False)[0]  # (M,D)
            nu = torch.triangular_solve((inducing_val - nu),
                                        Lu.T, upper=True)[0]  # (M,D)
        else:
            nu = torch.triangular_solve(u_prior.T.unsqueeze(2), Lu, upper=False)[0]  # (D,M,1)
            nu = torch.triangular_solve((inducing_val.T.unsqueeze(2) - nu),
                                        Lu.permute(0, 2, 1), upper=True)[0]  # (D,M,1)
        self.nu = nu  # (D,M)

    def rff_forward(self, x):
        """
        Calculates samples from the GP prior with random Fourier Features
        @param x: input tensor (N,D)
        @return: function values (N,D_out)
        """
        # compute feature map
        xo = torch.einsum('nd,dfk->nfk' if self.dimwise else 'nd,df->nf', x, self.rff_omega)  # (N,S) or (N,S,D_in)
        phi_ = torch.cos(xo + self.rff_phase)  # (N,S) or (N,S,D_in)
        phi = phi_ * torch.sqrt(self.kern.variance / self.S)  # (N,S) or (N,S,D_in)

        # compute function values
        f = torch.einsum('nfk,fk->nk' if self.dimwise else 'nf,fd->nd', phi, self.rff_weights)  # (N,D_out)
        return f  # (N,D_out)

    def build_conditional(self, x, full_cov=False):
        """
        Calculates conditional distribution q(f(x)) = N(m(x), Sigma(x))
            where  m(x) = k(x,Z)k(Z,Z)^{-1}u, k(x,x)
                   Sigma(x) = k(x,Z)k(Z,Z)^{-1}(S-K(Z,Z))k(Z,Z)^{-1}k(Z,x))

        @param x: input tensor (N,D)
        @param full_cov: if True, returns full Sigma(x) else returns only diagonal
        @return: m(x), Sigma(x)
        """
        Ku = self.kern.K(self.inducing_loc())  # (M,M) or (D,M,M)
        Lu = torch.cholesky(Ku + torch.eye(self.M) * jitter)  # (M,M) or (D,M,M)
        Kuf = self.kern.K(self.inducing_loc(), x)  # (M,N) or (D,M,N)
        A = torch.triangular_solve(Kuf, Lu, upper=False)[0]  # (M,M)@(M,N) --> (M,N) or (D,M,M)@(D,M,N) --> (D,M,N)

        Us_sqrt = self.Us_sqrt().T[:, :, None] if self.q_diag else self.Us_sqrt()  # (D,M,1) or (D,M,M)
        SK = (Us_sqrt @ Us_sqrt.permute(0, 2, 1)) - torch.eye(Ku.shape[1]).unsqueeze(0)  # (D,M,M)
        B = torch.einsum('dme, den->dmn' if self.dimwise else 'dmi, in->dmn', SK, A)  # (D,M,N)

        if full_cov:
            delta_cov = torch.einsum('dme, dmn->den' if self.dimwise else 'me, dmn->den', A, B)  # (D,M,N)
            Kff = self.kern.K(x) if self.dimwise else self.kern.K(x).unsqueeze(0)  # (1,N,N) or (D,N,N)
        else:
            delta_cov = ((A if self.dimwise else A.unsqueeze(0)) * B).sum(1)  # (D,N)
            if self.dimwise:
                Kff = torch.diagonal(self.kern.K(x), dim1=1, dim2=2)  # (N,)
            else:
                Kff = torch.diagonal(self.kern.K(x), dim1=0, dim2=1)  # (D,N)

        var = Kff + delta_cov
        mean = torch.einsum('dmn, md->nd' if self.dimwise else 'mn, md->nd', A, self.Um())
        return mean, var.T  # (N,D) , (N,D) or (N,N,D)

    def forward(self, t, x):
        """
        Compute sample from the SVGP posterior using decoupeld sampling approach
        Involves two steps:
            1. Generate sample from the prior :: rff_forward
            2. Compute pathwise updates using samples from inducing posterior :: build_cache

        @param t: time value, usually None as we define time-invariant ODEs
        @param x: input tensor in (N,D)
        @return: f(x) where f is a sample from GP posterior
        """
        # generate a prior sample using rff
        f_prior = self.rff_forward(x)  # (N,D))

        # compute pathwise updates
        if not self.dimwise:
            Kuf = self.kern.K(self.inducing_loc(), x)  # (M,N)
            f_update = torch.einsum('md, mn -> nd', self.nu, Kuf)  # (N,D)
        else:
            Kuf = self.kern.K(self.inducing_loc(), x)  # (D,M,N)
            f_update = torch.einsum('dm, dmn -> nd', self.nu.squeeze(2), Kuf)  # (N,D)

        # sample from the GP posterior
        dx = f_prior + f_update  # (N,D)

        return dx  # (N,D)

    def kl(self):
        """
        Computes KL divergence for inducing variables in whitened form
        Calculated as KL between multivariate Gaussians q(u) ~ N(m,S) and p(U) ~ N(0,I)

        @return: KL divergence value tensor
        """
        alpha = self.Um()  # (M,D)

        if self.q_diag:
            Lq = Lq_diag = self.Us_sqrt()  # (M,D)
        else:
            Lq = torch.tril(self.Us_sqrt())  # (D,M,M)
            Lq_diag = torch.diagonal(Lq, dim1=1, dim2=2).t()  # (M,D)

        # compute Mahalanobis term
        mahalanobis = torch.pow(alpha, 2).sum(dim=0, keepdim=True)  # (1,D)

        # log-determinant of the covariance of q(u)
        logdet_qcov = torch.log(torch.pow(Lq_diag, 2)).sum(dim=0, keepdim=True)  # (1,D)

        # trace term
        if self.q_diag:
            trace = torch.pow(Lq, 2).sum(dim=0, keepdim=True)  # (M,D) --> (1,D)
        else:
            trace = torch.pow(Lq, 2).sum(dim=(1, 2)).unsqueeze(0)  # (D,M,M) --> (1,D)

        logdet_pcov = 0.0
        constant = - torch.tensor(self.M)
        twoKL = logdet_pcov - logdet_qcov + mahalanobis + trace + constant
        kl = 0.5 * twoKL.sum()
        return kl
