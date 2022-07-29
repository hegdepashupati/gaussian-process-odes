import torch
from torch import nn
import numpy as np
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_nonadjoint


class ODEFunc(nn.Module):
    """
    Defines the ODE function:
    Stolen from https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
    """
    def __init__(self, D, H=128):
        """
        @param D: Observation dimension
        @param H: Number of hidden units
        """
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(D, H),
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, D),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class Flow(nn.Module):
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
    def __init__(self, odefunc, solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(Flow, self).__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint

    def forward(self, x0, ts):
        odeint = odeint_adjoint if self.use_adjoint else odeint_nonadjoint
        xs = odeint(
            self.odefunc,
            x0,
            ts,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )
        return xs.permute(1, 0, 2)


def build_model(args, data_ys):
    """
    Builds a NeuralODE model based on the MoCap experimental setup

    @param data_ys: observation sequence of shape (N,T,D)
    @param args: model setup arguments
    """
    N, T, D = data_ys.shape
    odefunc = ODEFunc(D, args.num_hidden)
    flow = Flow(odefunc=odefunc, solver=args.solver, use_adjoint=args.use_adjoint)
    return flow


def compute_loss(pred_ys, ys):
    """
    Computes loss function for NeuralODE training
    @param pred_ys: predicted sequence
    @param ys: actual sequence
    """
    loss = torch.mean(torch.pow(pred_ys - ys, 2))
    return loss


def compute_predictions(model, y0, ts):
    """
    Given an initial state y0, predicts NeuralODE dynamics for time sequence ts
    @param y0: initial state tensor (N,D)
    @param ts: time sequence tensor (T,)
    @return: predicted dynamics tensor (N,T,D)
    """
    model.eval()
    with torch.no_grad():
        pred = model(y0, ts)
    return pred


def compute_summary(actual, predicted, ys=1.0):
    """
    Computes MSE between actual and predicted values
    """
    actual = actual * ys
    predicted = predicted * ys
    return np.power(actual - predicted, 2).mean()
