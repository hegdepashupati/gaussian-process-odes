from src.misc.constraint_utils import invsoftplus

import torch
import numpy as np
from scipy.cluster.vq import kmeans2


def initialize_inducing(model, data_xs, ts_max, data_noise=1e0):
    """
    Initialization of inducing variables for GPODE model
    Inducing locations are initialized at cluster centers
    Inducing values are initialized using empirical data gradients.

    @param model: a gpode.SequenceModel object
    @param data_xs: observed sequence (N,T,D)
    @param ts_max: max observation time, observations are assumed to start at time=0.
    @param data_noise: an initial guess for observation noise.
    @return: the model object after initialization
    """

    # compute empirical gradients and scale them according to observation time.
    f_xt = data_xs[:, 1:, :] - data_xs[:, :-1, :]
    f_xt = f_xt.reshape(-1, data_xs.shape[-1]) * (data_xs.shape[1] / ts_max)
    data_xs = data_xs[:, :-1, :]
    data_xs = data_xs.reshape(-1, data_xs.shape[-1])

    with torch.no_grad():
        num_obs_for_initialization = np.minimum(1000, data_xs.shape[0])
        obs_index = np.random.choice(data_xs.shape[0], num_obs_for_initialization, replace=False)

        inducing_loc = torch.tensor(kmeans2(data_xs, k=model.flow.odefunc.diffeq.Um().shape[0], minit='points')[0])
        data_xs = torch.tensor(data_xs[obs_index])
        f_xt = torch.tensor(f_xt[obs_index])

        Kxx = model.flow.odefunc.diffeq.kern.K(data_xs)  # (N,N) or (D,N,N)
        Kxz = model.flow.odefunc.diffeq.kern.K(data_xs, inducing_loc)  # (N,M) or (D,N,M)
        Kzz = model.flow.odefunc.diffeq.kern.K(inducing_loc)  # (M,M) or (D,M,M)
        Lxx = torch.cholesky(Kxx + torch.eye(Kxx.shape[1]) * data_noise)  # (N,N) or (D,N,N)
        Lzz = torch.cholesky(Kzz + torch.eye(Kzz.shape[1]) * 1e-6)  # (M,M) or (D,M,M)

        if not model.flow.odefunc.diffeq.dimwise:
            alpha = torch.triangular_solve(f_xt, Lxx, upper=False)[0]  # (N,D)
            alpha = torch.triangular_solve(alpha, Lxx.T, upper=True)[0]  # (N,D)
            f_update = torch.einsum('nm, nd -> md', Kxz, alpha)  # (M,D)
        else:
            alpha = torch.triangular_solve(f_xt.T.unsqueeze(2), Lxx, upper=False)[0]  # (N,D)
            alpha = torch.triangular_solve(alpha, Lxx.permute(0, 2, 1), upper=True)[0]  # (N,D)
            f_update = torch.einsum('dnm, dn -> md', Kxz, alpha.squeeze(2))  # (M,D)

        inducing_val = torch.triangular_solve(f_update.T.unsqueeze(2), Lzz, upper=False)[0].squeeze(2).T  # (M,D)

        model.flow.odefunc.diffeq.inducing_loc().data = inducing_loc.data  # (M,D)
        model.flow.odefunc.diffeq.Um().data = inducing_val.data  # (M,D)
        return model


def initialize_latents_with_data(model, data_xs, data_ts, num_samples=50):
    """
    Initializes initial state distribution from data.
    Initial state distribution is initialized by solving the ODE backward in time from the first
        observation after inducing variables are initialized.

    @param model: a gpode.SequenceModel object
    @param data_xs: observed state sequence
    @param data_ts: observed time sequence
    @param num_samples: number of samples to consider for initial state initialization
    @return: the model object after initialization
    """
    with torch.no_grad():
        ts = torch.tensor(data_ts)

        # run the ODE backwards-in-time to find the initial state
        init_ts = torch.flip(ts[:2], [0])
        init_x0 = []
        for _ in range(num_samples):
            init_x0.append(model.build_flow(torch.tensor(data_xs[:, 0]), init_ts).clone().detach().data[:, -1])
        init_x0 = torch.stack(init_x0).mean(0)
        model.x0_distribution._initialize(init_x0)
    return model


def initialize_noisevar(model, init_noisevar):
    """
    Initializes likelihood observation noise variance parameter

    @param model: a gpode.SequenceModel object
    @param init_noisevar: initialization value
    @return: the model object after initialization
    """
    model.likelihood.unconstrained_variance.data = invsoftplus(
        torch.tensor(init_noisevar)).data
    return model


def initialize_and_fix_kernel_parameters(model, lengthscale_value=1.25, variance_value=0.5, fix=False):
    """
    Initializes and optionally fixes kernel parameter 

    @param model: a gpode.SequenceModel object
    @param lengthscale_value: initialization value for kernel lengthscales parameter
    @param variance_value: initialization value for kernel signal variance parameter
    @param fix: a flag variable to fix kernel parameters during optimization
    @return: the model object after initialization
    """
    model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.data = invsoftplus(
        lengthscale_value * torch.ones_like(model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.data))
    model.flow.odefunc.diffeq.kern.unconstrained_variance.data = invsoftplus(
        variance_value * torch.ones_like(model.flow.odefunc.diffeq.kern.unconstrained_variance.data))
    if fix:
        model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.requires_grad_(False)
        model.flow.odefunc.diffeq.kern.unconstrained_variance.requires_grad_(False)
    return model
