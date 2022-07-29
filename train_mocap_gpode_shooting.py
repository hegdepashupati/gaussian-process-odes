import os
import argparse
import time
import json
import torch
import numpy as np

from src.misc.settings import settings

device = settings.device
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from src.gpode_shooting.mocap_model_builder import build_model
from src.gpode_shooting.mocap_model_builder import compute_predictions
from src.gpode_shooting.mocap_model_builder import compute_test_predictions, compute_summary, compute_loss
from src.gpode_shooting.mocap_model_builder import compute_inducing_variables_for_plotting
from src.gpode_shooting.mocap_initialization import initialize_inducing, initialize_latents_with_data, \
    initialize_noisevar, \
    initialize_and_fix_kernel_parameters
from src.gpode_shooting.plots_mocap import plot_pca_predictions, plot_data_predictions, plot_inducing_posterior_3d
from src.gpode_shooting.plots_mocap import plot_trace

from src.misc import meter_utils as meters
from src.misc import io_utils
from src.misc.torch_utils import torch2numpy, save_model_optimizer, seed_everything

from src.datasets.mocap import MocapDataset
from src.misc.mocap_utils import Latent2DataProjector

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
CONSTRAINTS = ["gauss", "laplace"]
parser = argparse.ArgumentParser('Learning human motion dynamics with shooting GPODE')

# model parameters
parser.add_argument('--num_features', type=int, default=256,
                    help="Number of Fourier basis functions (for pathwise sampling from GP)")
parser.add_argument('--num_inducing', type=int, default=100,
                    help="Number of inducing points for the sparse GP")
parser.add_argument('--dimwise', type=eval, default=True,
                    help="Specify separate lengthscales for every output dimension")
parser.add_argument('--q_diag', type=eval, default=False,
                    help="Diagonal posterior approximation for inducing variables")
parser.add_argument('--num_latents', type=int, default=5,
                    help="Number of latent dimensions for training")

# constraint parameters
parser.add_argument('--constraint_type', type=str, default='gauss', choices=CONSTRAINTS,
                    help="Prior specification for shooting constraints")
parser.add_argument('--constraint_trainable', type=bool, default=False,
                    help="Learn the scale parameter for shooting prior during training")
parser.add_argument('--constraint_initial_scale', type=float, default=1e-3,
                    help="Prior scale parameter for shooting constraint")

# data processing arguments
parser.add_argument('--data_subject', type=str, default='09',
                    help="MoCap subject name")
parser.add_argument('--data_seqlen', type=int, default=100,
                    help="Training sequence length")

# ode solver arguments
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--ts_dense_scale', type=int, default=2,
                    help="Factor for making a dense integration time grid (useful for explicit solvers)")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")

# training arguments
parser.add_argument('--num_iter', type=int, default=10_000,
                    help="Number of gradient steps for model training")
parser.add_argument('--num_samples', type=int, default=5,
                    help="Number of reparameterized samples for computing gradients while training")
parser.add_argument('--lr', type=float, default=0.005,
                    help="Learning rate for model training")
parser.add_argument('--eval_sample_size', type=int, default=128,
                    help="Number of posterior samples to evaluate the model predictive performance")

parser.add_argument('--save', type=str, default='results/mocap/gpode-shooting',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--log_freq', type=int, default=20,
                    help="Logging frequency while training")

if __name__ == '__main__':
    args = parser.parse_args()

    # setup output directory and logger
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save, '')
    io_utils.makedirs(args.save)
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))

    # set global random seed
    seed_everything(args.seed)

    # dump training specs
    with open(args.save + 'train_args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # load mocap dataset - both in observation space and latent space (PCA projections)
    data_full = MocapDataset(data_path='data/mocap', subject=args.data_subject,
                             pca_components=-1,
                             data_normalize=False, pca_normalize=False,
                             dt=0.01, seqlen=args.data_seqlen)
    data_pca = MocapDataset(data_path='data/mocap', subject=args.data_subject,
                            pca_components=args.num_latents,
                            data_normalize=False, pca_normalize=True,
                            dt=0.01, seqlen=args.data_seqlen)
    latent2data_projection = Latent2DataProjector(data_pca)

    # build model and initialize with empirical gradients
    model = build_model(data_full.trn.ys, data_pca.trn.ys, latent2data_projection, args)

    train_ys_full = torch.tensor(data_full.trn.ys)
    train_ys = torch.tensor(data_pca.trn.ys)
    train_ts = torch.tensor(data_full.trn.ts)

    test_ys_full = torch.tensor(data_full.tst.ys)
    test_ys = torch.tensor(data_pca.tst.ys)
    test_ts = torch.tensor(data_full.tst.ts)

    with torch.no_grad():
        predicted_zs = compute_predictions(model, train_ts, eval_sample_size=args.eval_sample_size)
        predicted_ys = torch.stack([latent2data_projection(p) for p in predicted_zs])

    plot_pca_predictions(actual=data_pca.trn.ys,
                         predicted=predicted_zs,
                         ts=data_pca.trn.ts, args=args, num_obs=5,
                         name='plt_latents_before_initialization')
    plot_data_predictions(actual=data_full.trn.ys,
                          predicted=predicted_ys,
                          ts=data_pca.trn.ts, args=args, num_obs=5,
                          name='plt_data_before_initialization')

    model = initialize_and_fix_kernel_parameters(model, lengthscale_value=1.25, variance_value=0.5, fix=False)
    model = initialize_inducing(model, data_pca.trn.ys, data_pca.trn.ts.max(), 1e-0)
    model = initialize_latents_with_data(model, data_pca.trn.ys, data_pca.trn.ts)
    with torch.no_grad():
        predicted_zs = compute_predictions(model, train_ts, eval_sample_size=args.eval_sample_size)
        predicted_ys = torch.stack([latent2data_projection(p) for p in predicted_zs])
    model = initialize_noisevar(model, 1.5 * (data_full.trn.ys - torch2numpy(predicted_ys)).var((0, 1, 2)) + 1e-6)

    plot_pca_predictions(actual=data_pca.trn.ys,
                         predicted=predicted_zs,
                         ts=data_pca.trn.ts, args=args, num_obs=5,
                         name='plt_latents_after_initialization')
    plot_data_predictions(actual=data_full.trn.ys,
                          predicted=predicted_ys,
                          ts=data_pca.trn.ts, args=args, num_obs=5,
                          name='plt_data_after_initialization')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_meter = meters.CachedRunningAverageMeter(0.98)
    observ_nll_meter = meters.CachedRunningAverageMeter(0.98)
    state_kl_meter = meters.CachedRunningAverageMeter(0.98)
    init_kl_meter = meters.CachedRunningAverageMeter(0.98)
    inducing_kl_meter = meters.CachedRunningAverageMeter(0.98)
    time_meter = meters.CachedAverageMeter()

    # training loop
    for itr in range(1, args.num_iter):
        try:
            model.train()
            begin = time.time()
            optimizer.zero_grad()

            loss, observ_nll, state_kl, init_kl, inducing_kl = compute_loss(model, train_ys_full, train_ts,
                                                                            num_samples=args.num_samples)

            loss.backward()
            optimizer.step()

            if itr > 100:
                loss_meter.update(loss.item(), itr)
                observ_nll_meter.update(observ_nll.item(), itr)
                state_kl_meter.update(state_kl.item(), itr)
                init_kl_meter.update(init_kl.item(), itr)
                inducing_kl_meter.update(inducing_kl.item(), itr)
                time_meter.update(time.time() - begin, itr)

                if itr % args.log_freq == 0:
                    log_message = (
                        'Iter {:06d} | Time {:0.4f}({:.4f}) | Loss {:.3f}({:.3f}) |'
                        'OBS NLL {:.2f}({:.2f}) | XS KL {:.2f}({:.2f}) |'
                        'X0 KL {:.2f}({:.2f}) | IND KL {:.2f}({:.2f})'.format(
                            itr, time_meter.sum, time_meter.avg, loss_meter.val, loss_meter.avg,
                            observ_nll_meter.val, observ_nll_meter.avg,
                            state_kl_meter.val, state_kl_meter.avg,
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

    with torch.no_grad():
        train_pred_zs = compute_test_predictions(model=model,
                                                 y0=train_ys[:, 0],
                                                 ts=train_ts,
                                                 eval_sample_size=args.eval_sample_size)
        train_pred_ys = torch.stack([latent2data_projection(p) for p in train_pred_zs])

    with torch.no_grad():
        test_pred_zs = compute_test_predictions(model=model,
                                                y0=test_ys[:, 0],
                                                ts=test_ts,
                                                eval_sample_size=args.eval_sample_size)
        test_pred_ys = torch.stack([latent2data_projection(p) for p in test_pred_zs])
    train_pred_zs = torch2numpy(train_pred_zs)
    train_pred_ys = torch2numpy(train_pred_ys)
    test_pred_zs = torch2numpy(test_pred_zs)
    test_pred_ys = torch2numpy(test_pred_ys)

    ys_std = torch2numpy(data_full.data_std.squeeze(0)) if data_full.data_normalize else 1.0
    obs_noisevar = torch2numpy(model.likelihood.variance)

    train_ll, train_mse = compute_summary(
        actual=data_full.trn.ys,
        predicted=train_pred_ys,
        noise_var=obs_noisevar,
        ys=ys_std
    )
    test_ll, test_mse = compute_summary(
        actual=data_full.tst.ys,
        predicted=test_pred_ys,
        noise_var=obs_noisevar,
        ys=ys_std
    )

    logger.info("[TRAIN] LL {:.3f} | MSE {:.3f}".format(train_ll, train_mse))
    logger.info("[TEST]  LL {:.3f} | MSE {:.3f}".format(test_ll, test_mse))

    logger.info("Kernel length scales {}".format(model.flow.odefunc.diffeq.kern.lengthscales.data))
    logger.info("Kernel variance {}".format(model.flow.odefunc.diffeq.kern.variance.data))
    logger.info("Observation likelihood variance {}".format(obs_noisevar))
    logger.info("Constraint likelihood variance {}".format(model.likelihood.variance.data))

    plot_pca_predictions(actual=data_pca.trn.ys,
                         predicted=train_pred_zs,
                         ts=data_pca.trn.ts, args=args, num_obs=5,
                         name='plt_latents_after_optimization_train')
    plot_data_predictions(actual=data_full.trn.ys,
                          predicted=train_pred_ys,
                          ts=data_pca.trn.ts, args=args, num_obs=5,
                          name='plt_data_after_optimization_train')

    plot_pca_predictions(actual=data_pca.tst.ys,
                         predicted=test_pred_zs,
                         ts=data_pca.tst.ts, args=args, num_obs=5,
                         name='plt_latents_after_optimization_test')
    plot_data_predictions(actual=data_full.tst.ys,
                          predicted=test_pred_ys,
                          ts=data_pca.tst.ts, args=args, num_obs=5,
                          name='plt_data_after_optimization_test')

    plot_trace(loss_meter, observ_nll_meter, inducing_kl_meter, state_kl_meter, args)
    inducing_u, inducing_z = compute_inducing_variables_for_plotting(model)
    latent_samples = model.state_distribution.sample(num_samples=500)

    plot_pca_predictions(actual=data_pca.trn.ys[:, :-1],
                         predicted=torch2numpy(latent_samples)[:, :, 1:],
                         ts=data_pca.trn.ts[1:], args=args, num_obs=10,
                         name='latents_posterior')

    plot_inducing_posterior_3d(pred=train_pred_zs,
                               ts=data_full.trn.ts,
                               u=inducing_u, z=inducing_z, args=args,
                               name='inducing_posterior_train')

    plot_inducing_posterior_3d(pred=test_pred_zs,
                               ts=data_full.tst.ts,
                               u=inducing_u, z=inducing_z, args=args,
                               name='inducing_posterior_test')

    np.savez(file=os.path.join(args.save, 'model_predictions.npz'),
             train_pred_zs=train_pred_zs,
             train_pred_ys=train_pred_ys,
             test_pred_zs=test_pred_zs,
             test_pred_ys=test_pred_ys,
             obs_noisevar=obs_noisevar
             )
