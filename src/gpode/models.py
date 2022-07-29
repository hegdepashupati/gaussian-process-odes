from src.misc.torch_utils import insert_zero_t0, compute_ts_dense

from torch import nn


class SequenceModel(nn.Module):
    """
    GPODE model for learning unknown ODEs based on sequences of observations from the system.
    Currently we assume observations on a uniform time grid in the manuscript,
        but this implementation can be used as-it-is for non-uniform observations over time as well.

    Defines following main methods:
        build_flow: given an initial state and time sequence, perform forward ODE integration
        build_lowerbound_terms: given observed states and time, builds individual terms for the lowerbound computation
        build_kl: computes KL divergence between inducing prior and posterior.
        forward: a wrapper for build_flow method
    """

    def __init__(self, flow,
                 num_observations,
                 x0_distribution,
                 likelihood,
                 ts_dense_scale=1):
        super(SequenceModel, self).__init__()
        self.flow = flow
        self.num_observations = num_observations
        self.x0_distribution = x0_distribution

        self.likelihood = likelihood
        self.ts_dense_scale = ts_dense_scale

    def build_flow(self, x0, ts):
        """
        Given an initial state and time sequence, perform forward ODE integration
        Optionally, the time sequence can be made dense based on self.ts_dense_scale parameter

        @param x0: initial state tensor (N,D)
        @param ts: time sequence tensor (T,)
        @return: forward solution tensor (N,T,D)
        """
        ts = compute_ts_dense(ts, self.ts_dense_scale)
        ys = self.flow(x0, ts)
        return ys[:, ::self.ts_dense_scale - 1, :]

    def build_lowerbound_terms(self, ys, ts):
        """
        Given observed states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @return: nll, initial state KL
        """
        ts = insert_zero_t0(ts)
        x0_samples = self.x0_distribution.sample(num_samples=1)[0]
        x0_kl = self.x0_distribution.kl()
        xs = self.build_flow(x0_samples, ts)[:, 1:]
        loglik = self.likelihood.log_prob(xs, ys)
        return loglik.mean(), x0_kl.mean() / self.num_observations

    def build_kl(self):
        """
        Computes KL divergence between inducing prior and posterior.

        @return: inducing KL scaled by the number of observations
        """
        return self.flow.kl() / self.num_observations

    def forward(self, x0, ts):
        """
        A wrapper for build_flow method
        @param x0: initial state tensor (N,D)
        @param ts: time sequence tensor (T,)
        @return: forward solution tensor (N,T,D)
        """
        return self.build_flow(x0, ts)
