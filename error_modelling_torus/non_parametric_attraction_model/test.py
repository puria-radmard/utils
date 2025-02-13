import torch
from torch import Tensor as _T

from typing import Type

import matplotlib.pyplot as plt

import os
from tqdm import tqdm

import numpy as np

from purias_utils.error_modelling_torus.non_parametric_attraction_model.variational_approximation import SVGPApproximation, NoInputSVGPApproximation
from purias_utils.error_modelling_torus.non_parametric_attraction_model.parameters_gp_prior import WeilandAttractionErrorDistributionParametersPrior, NoInputAttractionErrorDistributionParametersPrior
from purias_utils.error_modelling_torus.non_parametric_attraction_model import main as models
from purias_utils.error_modelling_torus.non_parametric_attraction_model import test_synth_data as data

from purias_utils.error_modelling_torus.non_parametric_attraction_model import emissions_distribution as ed



save_path = '/homes/pr450/repos/all_utils/purias_utils/error_modelling_torus/non_parametric_attraction_model/test_save_path'

device = 'cuda'
model_distn = 'VonMisesAttractionErrorModel'
model_prior = 'gp'

num_models = 3


### Setup model
model_class: Type[models.WrappedStableAttractionErrorModel] = getattr(models, model_distn)
if model_prior == 'no_input':
    num_function_samples = 256
    num_function_samples_display = 128
    prior_class = NoInputAttractionErrorDistributionParametersPrior
    svgp_class = NoInputSVGPApproximation
    svgp_kwargs = {}
elif model_prior == 'gp':
    num_function_samples = 64
    num_function_samples_display = 5
    prior_class = WeilandAttractionErrorDistributionParametersPrior
    svgp_class = SVGPApproximation
    svgp_kwargs = {'R': 12, 'fix_inducing_point_locations': False}

required_wss_params = model_class.wss_param_names
priors = {f'{wss_param}_prior': prior_class(num_models) for wss_param in required_wss_params}
variational_approxes = {f'{wss_param}_variational_approx': svgp_class(num_models, **svgp_kwargs) for wss_param in required_wss_params}

model = model_class(**priors, **variational_approxes).to(device)


### Generate synthetic data - only within model class for now!
num_datapoints = 1024

synthetic_eval_info_function = {
    ('WrappedStableAttractionErrorModel', 'gp'): data.gp_prior_wrapped_stable_generate_synthetic_eval_info,
    ('WrappedStableAttractionErrorModel', 'no_input'): data.no_input_prior_wrapped_stable_generate_synthetic_eval_info,
    ('VonMisesAttractionErrorModel', 'gp'): data.gp_prior_von_mises_generate_synthetic_eval_info,
    ('VonMisesAttractionErrorModel', 'no_input'): data.no_input_von_mises_generate_synthetic_eval_info,
}[(model_distn, model_prior)]
sampling_function = {
    'WrappedStableAttractionErrorModel': ed.skew_wrapped_stable_generate_samples,
    'VonMisesAttractionErrorModel': ed.von_mises_generate_samples,
}[model_distn]

synthetic_target_responses = models.generate_synthetic_target_responses(num_datapoints, device)
synthetic_eval_info = synthetic_eval_info_function(synthetic_target_responses, model, device)
synthetic_errors = models.generate_sample_dataset_from_inference_eval_info(synthetic_eval_info, sampling_function, model)


fig, axes = plt.subplots(len(model.wss_param_names), 2, figsize = (20, 20))
for i, wss_param in enumerate(model.wss_param_names):
    axes[i,0].set_title(wss_param)
    axes[i,1].set_title(wss_param + ' - with link')
    synthetic_eval_info[wss_param].plot_to_axes(
        evaluation_locations = synthetic_target_responses,
        model_idx = 0,
        axes = axes[i,0],
        mean_line_kwargs = {'color': 'red', 'label': 'g.t.'},
        std_fill_kwargs = {'color': 'red', 'alpha': 0.1},
    )
    synthetic_eval_info[wss_param].plot_to_axes(
        evaluation_locations = synthetic_target_responses,
        model_idx = 0,
        axes = axes[i,1],
        mean_line_kwargs = {'color': 'red', 'label': 'g.t.'},
        std_fill_kwargs = {'color': 'red', 'alpha': 0.1},
        link_function = model.get_link_function(wss_param)
    )


### In-model distribution
with torch.no_grad():
    true_model_elbo_terms = model.get_function_marginalised_loglikelihood(
        synthetic_target_responses, synthetic_errors,
        1, 0, False,
        override_variational_posterior_info={
            f'{k}_variational_inference_infos': [v] for k, v in synthetic_eval_info.items()
        }
    )
    in_model_marginalised_loglikelihood = round(true_model_elbo_terms['marginalised_loglikelihood'].sum(-1).item(), 3)
fig.savefig(os.path.join(save_path, 'variational_inference.png'))


def plot_data(targets, errors, labels):
    fig, axes = plt.subplot_mosaic("""
    AAB
    AAB
    """, figsize = (15, 5))
    for tar, err, lab in zip(targets, errors, labels):
        axes['A'].scatter(tar.cpu().numpy(), err.cpu().numpy(), alpha = 0.5, label = lab)
        axes['B'].hist(err.cpu().numpy(), 64, density = True)
    scatter_y = max(axes['A'].get_ylim()); axes['A'].set_ylim(-scatter_y, scatter_y)
    hist_x = max(axes['B'].get_xlim()); axes['B'].set_xlim(-hist_x, hist_x)
    axes['A'].legend()
    fig.savefig(os.path.join(save_path, 'synthetic_examples.png'))


plot_data([synthetic_target_responses], [synthetic_errors], ['training data'])



### Start optimisation
num_training_steps = 1000
logging_freq = 1
full_logging_freq = 5
max_variational_batch_size = 128

optim = torch.optim.Adam(model.parameters(), lr = 0.001)
all_elbos = np.zeros([num_models, num_training_steps])
all_llh = np.zeros([num_models, num_training_steps])
all_kl = {wss_param: np.zeros([num_models, num_training_steps]) for wss_param in model.wss_param_names}

for t in tqdm(range(num_training_steps)):

    elbo_terms = model.get_function_marginalised_loglikelihood(
        synthetic_target_responses, synthetic_errors,
        num_function_samples, max_variational_batch_size,
        True
    )

    loss: _T = - elbo_terms['total_elbo'].sum()
    
    loss.backward()
    optim.step()

    all_llh[:,t] = elbo_terms['marginalised_loglikelihood'].detach().cpu().numpy().sum(-1)
    all_elbos[:,t] = elbo_terms['total_elbo'].detach().cpu().numpy()
    for wss_param in model.wss_param_names:
        all_kl[wss_param][:,t] = elbo_terms['kl'][wss_param].detach().cpu().numpy()

    if t % logging_freq == 0:

        # Plot losses for all models
        fig, axes = plt.subplots(3, 1, figsize = (10, 15))
        for nm in range(num_models):
            axes[0].plot(all_elbos[nm,:t+1], color = 'blue')
            axes[1].plot(all_llh[nm,:t+1], color = 'red')
            for wss_param in model.wss_param_names:
                axes[2].plot(all_kl[wss_param][nm,:t+1], color = 'purple', label = f"${wss_param}$" if not nm else None)
        axes[0].set_title('ELBO over training')
        axes[1].set_title(f'LLH over training; in model = {in_model_marginalised_loglikelihood}')
        axes[2].set_title('KL terms')


        fig.savefig(os.path.join(save_path, 'losses.png'))


    if t % full_logging_freq == 0:

        with torch.no_grad():

            # Inference on a grid for the 4 functions
            grid_data, _, grid_eval_info, _ = model.inference_on_grid(max_variational_batch_size = -1)

            # Visualise the approximation (against the ground truth) for one model
            fig, axes = plt.subplots(len(model.wss_param_names), 2, figsize = (20, 20))
            for i, wss_param in enumerate(model.wss_param_names):
                axes[i,0].set_title(wss_param)
                axes[i,1].set_title(wss_param + ' - with link')
                synthetic_eval_info[wss_param].plot_to_axes(
                    evaluation_locations = synthetic_target_responses,
                    model_idx = 0,
                    axes = axes[i,0],
                    mean_line_kwargs = {'color': 'red', 'label': 'g.t.'},
                    std_fill_kwargs = {'color': 'red', 'alpha': 0.1},
                )
                synthetic_eval_info[wss_param].plot_to_axes(
                    evaluation_locations = synthetic_target_responses,
                    model_idx = 0,
                    axes = axes[i,1],
                    mean_line_kwargs = {'color': 'red', 'label': 'g.t.'},
                    std_fill_kwargs = {'color': 'red', 'alpha': 0.1},
                    link_function = model.get_link_function(wss_param)
                )
                grid_eval_info[f'{wss_param}_variational_inference_infos'][0].plot_to_axes(
                    evaluation_locations = grid_data,
                    model_idx = 0,
                    num_samples = num_function_samples_display,
                    axes = axes[i,0],
                    mean_line_kwargs = {'color': 'blue', 'label': 'approx'},
                    std_fill_kwargs = {'color': 'blue', 'alpha': 0.1},
                )
                grid_eval_info[f'{wss_param}_variational_inference_infos'][0].plot_to_axes(
                    evaluation_locations = grid_data,
                    model_idx = 0,
                    num_samples = num_function_samples_display,
                    axes = axes[i,1],
                    mean_line_kwargs = {'color': 'blue', 'label': 'approx'},
                    std_fill_kwargs = {'color': 'blue', 'alpha': 0.1},
                    link_function = model.get_link_function(wss_param)
                )
                axes[i,0].legend()

                model.plot_inducing_points_to_axes(
                    param_name = wss_param,
                    axes = axes[i,0],
                    link_function = False
                )
                model.plot_inducing_points_to_axes(
                    param_name = wss_param,
                    axes = axes[i,1],
                    link_function = True
                )

            fig.savefig(os.path.join(save_path, 'variational_inference.png'))


            model_drawn_errors = model.generate_sample_data(synthetic_target_responses, max_variational_batch_size)
            plot_data(
                [synthetic_target_responses, synthetic_target_responses],
                [synthetic_errors, model_drawn_errors],
                ['training data', 'synthetic data from model']
            )

        plt.clf()




