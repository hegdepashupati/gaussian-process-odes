import os
import argparse
import time
import json
import torch
import numpy as np

from src.gpode.model_builder import build_model
from src.gpode.model_builder import compute_loss, compute_predictions, compute_summary
from src.gpode.model_initialization import initialize_inducing, initialize_latents_with_data
from src.gpode.plots_2d import plot_model_initialization, plot_trace
from src.gpode.plots_2d import plot_inducing_posterior, plot_longitudinal, plot_vectorfield

from src.misc import meter_utils as meters
from src.misc import io_utils
from src.misc.torch_utils import torch2numpy, save_model_optimizer, seed_everything

from src.datasets.vanderpol import VanderPol

from src.misc.settings import settings

device = settings.device
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
parser = argparse.ArgumentParser('Learning Van der Pol system with GPODE')

# model parameters
parser.add_argument('--num_features', type=int, default=256,
                    help="Number of Fourier basis functions (for pathwise sampling from GP)")
parser.add_argument('--num_inducing', type=int, default=16,
                    help="Number of inducing points for the sparse GP")
parser.add_argument('--dimwise', type=eval, default=True,
                    help="Specify separate lengthscales for every output dimension")
parser.add_argument('--q_diag', type=eval, default=False,
                    help="Diagonal posterior approximation for inducing variables")

# data parameters
parser.add_argument('--data_obs_S', type=int, default=25,
                    help="Sequence length for training data simulation")
parser.add_argument('--data_obs_T', type=float, default=7.,
                    help="Sequence integration time for training data simulation")
parser.add_argument('--data_obs_noise_var', type=float, default=0.05,
                    help="Observation noise variance for data simulation")

# ode solver arguments
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--ts_dense_scale', type=int, default=4,
                    help="Factor for making a dense integration time grid (useful for explicit solvers)")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")

# training arguments
parser.add_argument('--num_iter', type=int, default=5_000,
                    help="Number of gradient steps for model training")
parser.add_argument('--lr', type=float, default=0.005,
                    help="Learning rate for model training")
parser.add_argument('--eval_sample_size', type=int, default=128,
                    help="Number of posterior samples to evaluate the model predictive performance")

parser.add_argument('--save', type=str, default='results/vdp/gpode',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--log_freq', type=int, default=10,
                    help="Logging frequency while training")

if __name__ == '__main__':
    args = parser.parse_args()

    # logger
    io_utils.makedirs(args.save)
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))

    # set global random seed
    seed_everything(args.seed)
    
    # dump training specs
    with open(args.save + 'train_args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # simulate vdp system
    # n_ahead refers to number of points for testing the model extrapolation
    n_ahead = args.data_obs_S
    data = VanderPol(S_train=args.data_obs_S, T_train=args.data_obs_T,
                     S_test=args.data_obs_S + n_ahead,
                     T_test=(args.data_obs_T * (args.data_obs_S + n_ahead - 1) / (args.data_obs_S - 1)),
                     noise_var=args.data_obs_noise_var,
                     x0=np.array([[-1.5, 2.5]]), mu=0.5)
    N, T, D = data.trn.ys.shape

    # build model and initialize with empirical gradients
    train_ys, train_ts = torch.tensor(data.trn.ys), torch.tensor(data.trn.ts)
    test_ys, test_ts = torch.tensor(data.tst.ys), torch.tensor(data.tst.ts)

    model = build_model(args, data.trn.ys)

    plot_model_initialization(model, data, args, "model_before_initialization.png")
    model = initialize_inducing(model, data.trn.ys, data.trn.ts.max())
    model = initialize_latents_with_data(model, data.trn.ys, data.trn.ts)
    plot_model_initialization(model, data, args, "model_after_initialization.png")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_meter = meters.CachedRunningAverageMeter(0.98)
    observ_nll_meter = meters.CachedRunningAverageMeter(0.98)
    init_kl_meter = meters.CachedRunningAverageMeter(0.98)
    inducing_kl_meter = meters.CachedRunningAverageMeter(0.98)
    time_meter = meters.CachedAverageMeter()

    # training loop
    for itr in range(1, args.num_iter):
        try:
            model.train()
            begin = time.time()
            optimizer.zero_grad()

            loss, observ_nll, init_kl, inducing_kl = compute_loss(model, train_ys, train_ts)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), itr)
            observ_nll_meter.update(observ_nll.item(), itr)
            init_kl_meter.update(init_kl.item(), itr)
            inducing_kl_meter.update(inducing_kl.item(), itr)
            time_meter.update(time.time() - begin, itr)

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:06d} | Time {:0.4f}({:.4f}) | Loss {:.3f}({:.3f}) |'
                    'OBS NLL {:.2f}({:.2f}) | X0 KL {:.2f}({:.2f}) | IND KL {:.2f}({:.2f})'.format(
                        itr, time_meter.sum, time_meter.avg, loss_meter.val, loss_meter.avg,
                        observ_nll_meter.val, observ_nll_meter.avg,
                        init_kl_meter.val, init_kl_meter.avg,
                        inducing_kl_meter.val, inducing_kl_meter.avg
                    )
                )
                logger.info(log_message)

        except KeyboardInterrupt:
            logger.info('Stopping optimization')
            break
    logger.info('********** Optimization completed **********')
    save_model_optimizer(model, optimizer, os.path.join(args.save, 'checkpt.pth'))

    test_pred = torch2numpy(compute_predictions(model, test_ts, args.eval_sample_size))
    train_pred = torch2numpy(compute_predictions(model, train_ts, args.eval_sample_size))

    train_ll, train_mse = compute_summary(actual=data.trn.ys,
                                          predicted=train_pred,
                                          noise_var=torch2numpy(model.likelihood.variance)
                                          )
    test_ll, test_mse = compute_summary(actual=data.tst.ys[:, T:],
                                        predicted=test_pred[:, :, T:],
                                        noise_var=torch2numpy(model.likelihood.variance)
                                        )

    logger.info("[TRAIN] LL {:.3f} | MSE {:.3f}".format(train_ll, train_mse))
    logger.info("[TEST]  LL {:.3f} | MSE {:.3f}".format(test_ll, test_mse))

    logger.info("Kernel lengthscales {}".format(model.flow.odefunc.diffeq.kern.lengthscales.data))
    logger.info("Kernel variance {}".format(model.flow.odefunc.diffeq.kern.variance.data))
    logger.info("Observation likelihood variance {}".format(model.likelihood.variance.data))

    plot_longitudinal(data, test_pred, torch2numpy(model.likelihood.variance), args)
    plot_vectorfield(model, data, test_pred, args)
    plot_inducing_posterior(model, data, args)
    plot_trace(loss_meter, observ_nll_meter, inducing_kl_meter, args)

    np.savez(file=os.path.join(args.save, 'model_predictions.npz'),
             train_ts=data.trn.ts,
             train_ys=data.trn.ys,
             train_pred=train_pred,
             test_ts=data.tst.ts,
             test_ys=data.tst.ys,
             obs_noisevar=torch2numpy(model.likelihood.variance)
             )
