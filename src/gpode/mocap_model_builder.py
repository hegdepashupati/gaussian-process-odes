from src.core.dsvgp import DSVGP_Layer
from src.core.likelihoods import ProjectedGaussian
from src.core.flow import Flow
from src.core.states import StateInitialVariationalGaussian
from src.gpode.models import SequenceModel
from src.misc import torch_utils
from src.misc.torch_utils import insert_zero_t0

from tqdm import tqdm
import torch
import numpy as np

from scipy.stats import norm
from scipy.special import logsumexp


def build_model(data_full_ys, data_pca_ys, latent2data_projection, args):
    """
    Builds a model object of gpode.SequenceModel based on the MoCap experimental setup

    @param data_full_ys: data sequence in observed space (N,T,D_full)
    @param data_pca_ys: data sequence in latent/PCA space (N,T,D)
    @param latent2data_projection: an object of misc.mocap_utils.Latent2DataProjector class
    @param args: model setup arguments
    @return: an object of gpode.SequenceModel class
    """
    N, T, D = data_pca_ys.shape
    D_full = data_full_ys.shape[2]

    gp = DSVGP_Layer(D_in=D,
                     D_out=D,
                     S=args.num_features,
                     M=args.num_inducing,
                     dimwise=args.dimwise,
                     q_diag=args.q_diag)

    flow = Flow(diffeq=gp, solver=args.solver, use_adjoint=args.use_adjoint)

    likelihood = ProjectedGaussian(projection=latent2data_projection, ndim=D_full)
    num_observations = N * T * D_full

    model = SequenceModel(flow=flow,
                          num_observations=num_observations,
                          x0_distribution=StateInitialVariationalGaussian(dim_n=N, dim_d=D),
                          likelihood=likelihood,
                          ts_dense_scale=args.ts_dense_scale)

    return model


def compute_loss(model, ys, ts, **kwargs):
    """
    Compute loss for GPODE optimization
    @param model: a gpode.SequenceModel object
    @param ys: true observation sequence
    @param ts: observation times
    @param kwargs: additional parameters passed to the model.build_lowerbound_terms() method
    @return: loss, nll, initial_state_kl, inducing_kl
    """
    observ_loglik, init_state_kl = model.build_lowerbound_terms(ys, ts, **kwargs)
    inducing_kl = model.build_kl()
    loss = -(observ_loglik - init_state_kl - inducing_kl)
    return loss, -observ_loglik, init_state_kl, inducing_kl


def compute_predictions(model, ts, eval_sample_size=10):
    """
    Compute predictions or ODE sequences from a GPODE model from an optimized initial state
    Useful while making predictions/extrapolation to novel time points from an optimized initial state.

    @param model: a gpode.SequenceModel object
    @param ts: prediction times (T,)
    @param eval_sample_size: number of samples for evaluation : S
    @return: predictive samples (S,N,T,D)
    """
    model.eval()

    # add additional time point accounting the initial state
    ts = insert_zero_t0(ts)

    pred_samples = []
    for _ in tqdm(range(eval_sample_size)):
        with torch.no_grad():
            pred_samples.append(model(model.x0_distribution.sample().squeeze(0), ts))
    return torch.stack(pred_samples, 0)[:, :, 1:]  # (S,N,T,D)


def compute_test_predictions(model, y0, ts, eval_sample_size=10):
    """
    Compute predictions or ODE sequences from a GPODE model from a given initial state

    @param model: a gpode.SequenceModel object
    @param y0: initial state for computing predictions (N,D)
    @param ts: prediction times (T,)
    @param eval_sample_size: number of samples for evaluation S
    @return: predictive samples (S,N,T,D)
    """
    model.eval()
    pred_samples = []
    for _ in tqdm(range(eval_sample_size)):
        with torch.no_grad():
            pred_samples.append(model(y0, ts))
    return torch.stack(pred_samples, 0)  # (S,N,T,D)


def compute_summary(actual, predicted, noise_var, ys=1.0):
    """
    Computes MSE and MLL as summary metrics between actual and predicted sequences
    @param actual: true observation sequence
    @param predicted: predicted sequence
    @param noise_var: noise var predicted by the model
    @param ys: optional scaling factor for standardized data
    @return: MLL(actual, predicted),  MSE(actual, predicted)
    """
    actual = actual * ys  # (S,N,T,D)
    predicted = predicted * ys  # (S,N,T,D)
    noise_var = noise_var * ys ** 2 + 1e-8  # (D,)

    def lig_lik(actual, predicted, noise_var):
        lik_samples = norm.logpdf(actual, loc=predicted, scale=noise_var ** 0.5)
        lik = logsumexp(lik_samples, 0, b=1 / float(predicted.shape[0]))
        return lik

    def mse(actual, predicted):
        return np.power(actual - predicted.mean(0), 2)

    return lig_lik(actual, predicted, noise_var).mean(), mse(actual, predicted).mean()  # noqa


def compute_inducing_variables_for_plotting(model):
    """
    Uniwhiten the inducing variables for generating plots
    @param model:  a gpode.SequenceModel object
    @return: inducing values, inducing locations
    """
    z = model.flow.odefunc.diffeq.inducing_loc().clone().detach()
    u = model.flow.odefunc.diffeq.Um().clone().detach()
    Ku = model.flow.odefunc.diffeq.kern.K(model.flow.odefunc.diffeq.inducing_loc())  # MxM or DxMxM
    Lu = torch.cholesky(Ku + torch.eye(Ku.shape[1]) * 1e-5)  # MxM or DxMxM
    if model.flow.odefunc.diffeq.dimwise:
        u = torch.einsum('mde, dnm -> nde', u.unsqueeze(2), Lu).squeeze(2)  # DxMx1
    else:
        u = torch.einsum('md, mn -> nd', u, Lu.T)  # NxD
    u = torch_utils.torch2numpy(u) / 1.5
    z = torch_utils.torch2numpy(z)
    return u, z
