from src.core.dsvgp import DSVGP_Layer
from src.core.likelihoods import Gaussian
from src.core.flow import Flow
from src.core.states import StateInitialVariationalGaussian
from src.gpode.models import SequenceModel
from src.misc import meter_utils as utils
from src.misc.torch_utils import insert_zero_t0

import time
from tqdm import tqdm
import torch
import numpy as np

from scipy.stats import norm
from scipy.special import logsumexp


def build_model(args, data_ys):
    """
    Builds a model object of gpode.SequenceModel based on training sequence

    @param data_ys: observed/training sequence of (N,T,D) dimensions
    @param args: model setup arguments
    @return: an object of gpode.SequenceModel class
    """
    N, T, D = data_ys.shape
    gp = DSVGP_Layer(D_in=D, D_out=D,
                     M=args.num_inducing,
                     S=args.num_features,
                     dimwise=args.dimwise,
                     q_diag=args.q_diag)

    flow = Flow(diffeq=gp, solver=args.solver, use_adjoint=args.use_adjoint)

    likelihood = Gaussian(ndim=D)

    model = SequenceModel(flow=flow,
                          num_observations=N * T * D,
                          x0_distribution=StateInitialVariationalGaussian(dim_n=N, dim_d=D),
                          likelihood=likelihood,
                          ts_dense_scale=args.ts_dense_scale)

    return model


def compute_loss(model, ys, ts):
    """
    Compute loss for GPODE optimization
    @param model: a gpode.SequenceModel object
    @param ys: true observation sequence
    @param ts: observation times
    @return: loss, nll, initial_state_kl, inducing_kl
    """
    observ_loglik, init_state_kl = model.build_lowerbound_terms(ys, ts)
    kl = model.build_kl()
    loss = -(observ_loglik - init_state_kl - kl)
    return loss, -observ_loglik, init_state_kl, kl


def compute_predictions(model, ts, eval_sample_size=10):
    """
    Compute predictions or ODE sequences from a GPODE model from an optimized initial state
    Useful while making predictions/extrapolation to novel time points from an optimized initial state.

    @param model: a gpode.SequenceModel object
    @param ts: observation times
    @param eval_sample_size: number of samples for evaluation
    @return: predictive samples
    """
    model.eval()

    # add additional time point accounting the initial state
    ts = insert_zero_t0(ts)
    pred_samples = []
    for _ in tqdm(range(eval_sample_size)):
        with torch.no_grad():
            pred_samples.append(model(model.x0_distribution.sample().squeeze(0), ts))
    return torch.stack(pred_samples, 0)[:, :, 1:]


def compute_test_predictions(model, x0, ts, eval_sample_size=10):
    """
    Compute predictions or ODE sequences from a GPODE model from a given initial state

    @param model: a gpode.SequenceModel object
    @param x0: initial state for computing predictions (N,D)
    @param ts: observation times
    @param eval_sample_size: number of samples for evaluation
    @return: predictive samples
    """
    model.eval()
    pred_samples = []
    for _ in tqdm(range(eval_sample_size)):
        with torch.no_grad():
            pred_samples.append(model(x0, ts))
    return torch.stack(pred_samples, 0)


def compute_summary(actual, predicted, noise_var, ys=1.0):
    """
    Computes MSE and MLL as summary metrics between actual and predicted sequences
    @param actual: true observation sequence
    @param predicted: predicted sequence
    @param noise_var: noise var predicted by the model
    @param ys: optional scaling factor for standardized data
    @return: MLL(actual, predicted),  MSE(actual, predicted)
    """
    actual = actual * ys
    predicted = predicted * ys
    noise_var = noise_var * ys ** 2 + 1e-8

    def lig_lik(actual, predicted, noise_var):
        lik_samples = norm.logpdf(actual, loc=predicted, scale=noise_var ** 0.5)
        lik = logsumexp(lik_samples, 0, b=1 / float(predicted.shape[0]))
        return lik

    def mse(actual, predicted):
        return np.power(actual - predicted.mean(0), 2)

    return lig_lik(actual, predicted, noise_var).mean(), mse(actual, predicted).mean()  # noqa


class Trainer:
    """
    A trainer class for GPODE model. Stores optimization trace for monitoring/plotting purpose
    """

    def __init__(self):
        self.loss_meter = utils.CachedRunningAverageMeter(0.98)
        self.observ_nll_meter = utils.CachedRunningAverageMeter(0.98)
        self.init_kl_meter = utils.CachedRunningAverageMeter(0.98)
        self.inducing_kl_meter = utils.CachedRunningAverageMeter(0.98)
        self.time_meter = utils.CachedAverageMeter()
        self.compute_loss = compute_loss

    def train(self, model, loss_function, ys, ts, num_iter, lr, log_freq):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for itr in range(1, num_iter):
            try:
                model.train()
                begin = time.time()
                optimizer.zero_grad()

                loss, observ_nll, init_kl, inducing_kl = loss_function(model, ys, ts)

                loss.backward()
                optimizer.step()

                self.loss_meter.update(loss.item(), itr)
                self.observ_nll_meter.update(observ_nll.item(), itr)
                self.init_kl_meter.update(init_kl.item(), itr)
                self.inducing_kl_meter.update(inducing_kl.item(), itr)
                self.time_meter.update(time.time() - begin, itr)

                if itr % log_freq == 0:
                    log_message = (
                        'Iter {:04d} | Loss {:.2f}({:.2f}) |'
                        'OBS NLL {:.2f}({:.2f})|'
                        'X0 KL {:.2f}({:.2f}) | IND KL {:.2f}({:.2f})'.format(
                            itr, self.loss_meter.val, self.loss_meter.avg,
                            self.observ_nll_meter.val, self.observ_nll_meter.avg,
                            self.init_kl_meter.val, self.init_kl_meter.avg,
                            self.inducing_kl_meter.val, self.inducing_kl_meter.avg
                        )
                    )
                    print(log_message, end="\r")

            except KeyboardInterrupt:
                break
        return model, optimizer
