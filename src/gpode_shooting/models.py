from src.misc.torch_utils import compute_ts_dense

from torch import nn


def stack_segments(unstacked):
    return unstacked.reshape(-1, unstacked.shape[-1])


def unstack_segments(stacked, unstacked_shape):
    return stacked.reshape(unstacked_shape)


class BaseSequenceModel(nn.Module):
    """
    Implements base class for shooting GPODE model for learning unknown ODEs.
    Model setup for observations on non-uniform grid or mini-batching over time can be derived from this class.

    Defines following methods:
        build_flow: given an initial state and time sequence, perform forward ODE integration
        build_flow_and_divergence: performs coupled forward ODE integration for states and density change
        build_lowerbound_terms: given observed states and time, builds individual terms for the lowerbound computation
        build_inducing_kl: computes KL divergence between inducing prior and posterior.
        forward: a wrapper for build_flow method
    """

    def __init__(self, flow, num_observations,
                 state_distribution,
                 likelihood,
                 constraint,
                 ts_dense_scale=2):
        super(BaseSequenceModel, self).__init__()
        self.flow = flow
        self.num_observations = num_observations
        self.state_distribution = state_distribution

        self.likelihood = likelihood
        self.constraint = constraint
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
        ys = self.flow(x0, ts, return_divergence=False)
        return ys[:, ::self.ts_dense_scale - 1, :]

    def build_lowerbound_terms(self, ys, ts, **kwargs):
        raise NotImplementedError

    def build_objective(self, ys, ts):
        """
        Computes lowerbound.

        @param ys: observed states tensor (N,T,D)
        @param ts: observed time sequence tensor (T,)
        @return: negative lowerbound
        """
        observ_loglik, state_constraint_loglik, state_entropy, initial_state_kl = self.build_lowerbound_terms(ys, ts)
        inducing_kl = self.build_inducing_kl()
        loss = -(observ_loglik + state_constraint_loglik + state_entropy - initial_state_kl - inducing_kl)
        return loss

    def build_inducing_kl(self):
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


class UniformSequenceModel(BaseSequenceModel):
    """
    Implements shooting GPODE model for data observed on uniform time grid.

    Defines following methods:
        build_lowerbound_terms: given observed states and time, builds individual terms for the lowerbound computation
    """

    def __init__(self, flow, num_observations,
                 state_distribution,
                 likelihood,
                 constraint,
                 ts_dense_scale=2):
        super(UniformSequenceModel, self).__init__(flow=flow,
                                                   num_observations=num_observations,
                                                   state_distribution=state_distribution,
                                                   likelihood=likelihood,
                                                   constraint=constraint,
                                                   ts_dense_scale=ts_dense_scale)

    def build_lowerbound_terms(self, ys, ts, num_samples=1, **kwargs):
        """
        Given observed states and time, builds the individual terms for the lowerbound computation

        @param ys: observed sequence tensor (N,T,D)
        @param ts: observed time sequence (T,)
        @param num_samples: number of reparametrized samples used to compute lowerbound
        @return: nll, state cross-entropy, state entropy, initial state KL
        """
        # generate initial states for shooting segments
        # as samples from the variational posteriors for shooting states
        ss_samples = self.state_distribution.sample(num_samples=num_samples)  # (S,N,T,D)
        (S, N, T, D) = ss_samples.shape

        # integrate shooting states forward-in-time to compute the end points of shooting segments
        predicted_xs = self.flow(x0=stack_segments(ss_samples),
                                 ts=ts[:2])  # (SxNxT, 2, D)
        predicted_xs = unstack_segments(predicted_xs[:, -1], (S, N, T, D))  # (S, N, T, D)

        # compute data log likelihood
        observation_loglik = self.likelihood.log_prob(predicted_xs, ys.unsqueeze(0))  # (S,N,T,D)

        # compute the entropy of variational posteriors for shooting states
        state_entropy = self.state_distribution.entropy()  # (N,T-1)

        # compute the constraint likelihoods
        state_constraint_logprob = self.constraint.log_prob(ss_samples[:, :, 1:, :],
                                                            predicted_xs[:, :, :-1, :]).sum(3)  # (S,N,T-1)

        # compute initial state KL
        initial_state_kl = self.state_distribution.x0.kl()  # (1,)

        assert (state_entropy.shape == (N, T - 1))
        assert (state_constraint_logprob.shape == (S, N, T - 1))

        scaled_state_constraint_loglik = state_constraint_logprob.mean(0).sum() / self.num_observations
        scaled_state_entropy = state_entropy.sum() / self.num_observations
        scaled_initial_state_kl = initial_state_kl / self.num_observations
        return observation_loglik.mean(), scaled_state_constraint_loglik, scaled_state_entropy, scaled_initial_state_kl
