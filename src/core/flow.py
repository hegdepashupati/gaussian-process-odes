import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_nonadjoint


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        """
        Defines the ODE function:
            mainly calls layer.build_cache() method to fix the draws from random variables.
        Modified from https://github.com/rtqichen/ffjord/

        @param diffeq: Layer of GPODE/npODE/neuralODE
        """
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, return_divergence, rebuild_cache):
        self.return_divergence = return_divergence
        self._num_evals.fill_(0)
        if rebuild_cache:
            self.diffeq.build_cache()

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        self._num_evals += 1
        if self.return_divergence:
            y = states[0]
            dy, divergence = self.diffeq.forward_divergence(t, y)
            return dy, divergence
        else:
            dy = self.diffeq(t, states)
            return dy


class Flow(nn.Module):
    def __init__(self, diffeq, solver='dopri5', atol=1e-6, rtol=1e-6, use_adjoint=False):
        """
        Defines an ODE flow:
            mainly defines forward() method for forward numerical integration of an ODEfunc object
        See https://github.com/rtqichen/torchdiffeq for more information on numerical ODE solvers.

        @param diffeq: Layer of GPODE/npODE/neuralODE
        @param solver: Solver to be used for ODE numerical integration
        @param atol: Absolute tolerance for the solver
        @param rtol: Relative tolerance for the solver
        @param use_adjoint: Use adjoint method for computing loss gradients, calls odeint_adjoint from torchdiffeq
        """
        super(Flow, self).__init__()
        self.odefunc = ODEfunc(diffeq)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint

    def forward(self, x0, ts, return_divergence=False):
        """
        Numerical solution of an IVP, and optionally compute divergence term for density transformation computation
        @param x0: Initial state (N,D) tensor x(t_0).
        @param ts: Time sequence of length T, first value is considered as t_0
        @param return_divergence: Bool flag deciding the divergence computation
        @return: xs: (N,T,D) tensor
        """
        odeint = odeint_adjoint if self.use_adjoint else odeint_nonadjoint
        self.odefunc.before_odeint(return_divergence=return_divergence, rebuild_cache=True)
        if return_divergence:
            states = odeint(
                self.odefunc,
                (x0, torch.zeros(x0.shape[0], 1).to(x0)),
                ts,
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            xs, divergence = states
            return xs.permute(1, 0, 2), divergence.permute(1, 0, 2)  # (N,T,D), # (N,T,1)
        else:
            xs = odeint(
                self.odefunc,
                x0,
                ts,
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            return xs.permute(1, 0, 2)  # (N,T,D)

    def inverse(self, x0, ts, return_divergence=False):
        odeint = odeint_adjoint if self.use_adjoint else odeint_nonadjoint
        self.odefunc.before_odeint(return_divergence=return_divergence, rebuild_cache=False)
        if return_divergence:
            states = odeint(
                self.odefunc,
                (x0, torch.zeros(x0.shape[0], 1).to(x0)),
                torch.flip(ts, [0]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            xs, divergence = states
            return xs.permute(1, 0, 2), divergence.permute(1, 0, 2)  # (N,T,D), # (N,T,1)
        else:
            xs = odeint(
                self.odefunc,
                x0,
                torch.flip(ts, [0]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            return xs.permute(1, 0, 2)  # (N,T,D)

    def num_evals(self):
        return self.odefunc.num_evals()

    def kl(self):
        """
        Calls KL() computation from the diffeq layer
        """
        return self.odefunc.diffeq.kl().sum()

    def log_prior(self):
        """
        Calls log_prior() computation from the diffeq layer
        """
        return self.odefunc.diffeq.log_prior().sum()
