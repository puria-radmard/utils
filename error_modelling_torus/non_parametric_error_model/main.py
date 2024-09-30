import torch
from torch import Tensor as _T
from torch.nn import Module, ModuleDict

from numpy import ndarray as _A

from typing import Dict, Optional, List

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel

import matplotlib.pyplot as plt

from purias_utils.util.plotting import standard_swap_model_simplex_plots, legend_without_repeats, lighten_color
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


class WorkingMemoryFullSwapModel(Module):

    def __init__(
        self, 
        generative_model: NonParametricSwapErrorsGenerativeModel,
        variational_models: Dict[int, NonParametricSwapErrorsVariationalModel],
        num_variational_samples: int,
        num_importance_sampling_samples: int,
    ) -> None:
        super().__init__()
        self.generative_model = generative_model

        if 0 in variational_models.keys():
            assert len(variational_models) == 1
            self.shared_variational_model = True
        else:
            self.shared_variational_model = False
        self.variational_models = ModuleDict(variational_models)

        self.num_variational_samples = num_variational_samples  # I
        self.num_importance_sampling_samples = num_importance_sampling_samples  # K

        assert generative_model.num_models == variational_models.num_models
        self.num_models = generative_model.num_models

    def get_variational_model(self, N):
        return self.variational_models[0] if self.shared_variational_model else self.variational_models[N]

    def get_elbo_terms(self, deltas: _T, data: Optional[_T], training_method: str = 'errors', max_variational_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}) -> Dict[str, Optional[_T]]:

        Q, M, N, D = deltas.shape
        I = self.num_variational_samples
        assert Q == variational_model.num_models
        assert D == self.variational_models[N].num_features
        
        variational_model = self.get_variational_model(N)
        R = variational_model.R

        all_deduplicated_deltas, M_minis = variational_model.deduplicate_deltas(deltas, max_variational_batch_size)  # "~M/batch_size length list of entries of shape [Q, ~batch*N, D]"

        # Use kernel all here:
        K_dds = [self.generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]                         # each [Q, ~batch*N, ~batch*N]
        K_uu = self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z)                                                                               # [Q, R, R]
        k_uds = [self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]    # each [Q, R, ~batch*N]

        # Inverse isn't always symmetric!!
        K_uu_inv = torch.linalg.inv(K_uu)
        assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
        if not torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all():
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))

        # Get the KL term of the loss
        if return_kl:
            kl_term = variational_model.kl_loss(K_uu = K_uu, K_uu_inv=K_uu_inv)     # [Q]
        else:
            kl_term = torch.ones(Q) * torch.nan # Won't plot!

        # Make variational inferences for q(f)
        mus, sigma_chols = [], []
        for k_ud, K_dd in zip(k_uds, K_dds):
            _mu, _sigma, _sigma_chol = variational_model.variational_gp_inference(k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv)  # [Q, ~batch*N], [Q, ~batch*N, ~batch*N], [Q, ~batch*N, ~batch*N]
            mus.append(_mu), sigma_chols.append(_sigma_chol)

        # Get the samples of f evaluated at the deltas
        all_f_samples = [
            variational_model.reparameterised_sample(num_samples = I, mu = mu, sigma_chol = sigma_chol, M = M, N = N)
            for mu, sigma_chol, M in zip(mus, sigma_chols, M_minis)
        ]   # Each of shape [Q, I, ~batchsize, N]

        # Shouldn't be any numerical problems after this
        f_samples = torch.concat(all_f_samples, 2)  # [Q, I, M, N]

        prior_info = self.generative_model.swap_function.generate_pi_vectors(
            set_size = N, model_evaulations = f_samples
        )
        priors = prior_info['pis']

        # Get the ELBO first term, depending on training mode (data is usually errors)
        if training_method == 'error':
            assert (Q, M, N) == tuple(data.shape)
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = self.generative_model.get_marginalised_log_likelihood(
                estimation_deviations = data, pi_vectors = priors,
                kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
            )
        elif training_method == 'beta':
            raise Exception
            llh_term = self.generative_model.get_component_log_likelihood(
                selected_components = data, pi_vectors = pi_vectors
            )
            posterior, unaggregated_lh = None, None
        elif training_method == 'none':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = None, None, None

        distance_loss = K_uu.tril(-1).max() / K_uu.max()

        return {
            'total_log_likelihood': total_log_likelihood, 
            'likelihood_per_datapoint': likelihood_per_datapoint, 
            'posterior': posterior_vectors, 
            'priors': priors, 
            'f_samples': f_samples,             # [Q, I, M, N]
            'kl_term': kl_term, 
            'distance_loss': distance_loss
        }

    def inference(self, deltas: _T):
        
        Q, M, N, D = deltas.shape
        assert Q == variational_model.num_models
        variational_model = self.get_variational_model(N)

        deduplicated_deltas = variational_model.deduplicate_deltas(deltas, 0)[0][0]

        with torch.no_grad():

            R = variational_model.R
            # TODO: add this noise injection to report!
            K_dd = self.generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas)
            K_uu = self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z)
            k_ud = self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z, deduplicated_deltas)
            K_uu_inv = torch.linalg.inv(K_uu)
            # K_dd_inv = torch.linalg.inv(K_dd)

            K_uu_inv = torch.linalg.inv(K_uu)
            assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
            assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

            # Make variational inferences for q(f)
            mu, sigma, sigma_chol = variational_model.variational_gp_inference(
                k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
            )

        return mu, sigma, sigma_chol

    def inference_on_grid(self, set_size: int, grid_count: int, device: str = 'cuda') -> Dict[str, _A]:
        """
        Set up a suitably shaped grid and perform inference over it

        TODO: shapes
        """
        variational_model = self.get_variational_model(set_size)
        D = variational_model.num_features
        if D not in [1, 2]:
            raise NotImplementedError

        full_grid = torch.linspace(-torch.pi, +torch.pi, grid_count)
        grid_locs = torch.tensor([0.0]) if D == 1 else full_grid.clone()
        grid_cols = full_grid.clone()
        grid_x, grid_y = torch.meshgrid(grid_locs, grid_cols, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y], -1).reshape(len(grid_cols) * len(grid_locs), 2).to(device)
        grid_points = grid_points[...,[0]] if D == 1 else grid_points

        grid_points = grid_points.unsqueeze(0).repeat(self.num_models, 1, 1)
        flat_mu_est, flat_sigma_est, sigma_chol = self.inference(grid_points)

        std_est = torch.stack([fse.diag() for fse in flat_sigma_est], 0).sqrt() # [Q, 100]
        eps = torch.randn(self.generative_model.num_models, 3, flat_mu_est.shape[1], dtype = flat_mu_est.dtype, device = flat_mu_est.device) # [Q, 3, MN (dedup size)]
        grid_f_samples = flat_mu_est.unsqueeze(1) + torch.bmm(eps, sigma_chol.transpose(-1, -2))   # [Q, 3, MN]

        return {
            'one_dimensional_grid': full_grid.cpu().numpy(),            # [Q, 100]
            'all_grid_points': grid_points.cpu().numpy(),               # [Q, 100, 1]
            'mean_surface': flat_mu_est.cpu().numpy(),                  # [Q, 100]
            'std_surface': std_est.cpu().numpy(),                       # [Q, 100]
            'function_samples_on_grid': grid_f_samples.cpu().numpy()    # [Q, 3, 100]
        }


    def visualise_variational_approximation(
        self, set_size: int, grid_count: int, 
        pi_u_tildes: _A, pi_1_tildes: _A,
        all_deltas: _A, recent_component_priors: Optional[_A], true_mean_surface: Optional[_A], true_std_surface: Optional[_A],
        min_separation: float, max_separation: float, deltas_label: str
    ):
        """
        Input:
            All taken from util.inference_on_grid

        If D = 1:
            TODO: list everything!

        If D = 2:
            TODO: list everything!

        TODO: shapes!
        TODO: make this not require pi_x_tilde inputs!
        """

        inference_info = self.inference_on_grid(set_size=set_size, grid_count=grid_count)
        one_dimensional_grid = inference_info['one_dimensional_grid']
        all_grid_points = inference_info['all_grid_points']
        mean_surface = inference_info['mean_surface']
        std_surface = inference_info['std_surface']
        function_samples_on_grid = inference_info['function_samples_on_grid']

        if self.num_features > 1:
            raise NotImplementedError

        else:
            Q = self.num_models
            figsize = 8
            fig_surfaces = plt.figure(figsize = (figsize * 4, figsize * (Q+1)))

            # axes_kernel = fig_surfaces.add_subplot(Q+1,4,2*Q)
            axes_hist = fig_surfaces.add_subplot(Q+1,4,4*Q+1)
            axes_hist.hist(all_deltas[:,1:].flatten(), 1024, density=True)
            axes_hist.set_xlabel(deltas_label)

            all_inducing_points = self.Z.detach().cpu().squeeze(-1).numpy()
            all_inducing_points_means = self.m_u.detach().cpu().numpy()
            if self.inducing_point_variational_parameterisation == 'gaussian':
                all_inducing_points_covars = self.S_uu.detach().cpu().numpy()

            for q in range(Q):
                axes1D_linear = fig_surfaces.add_subplot(Q+1,4,q*4+1)
                # axes1D_exponentiated = fig_surfaces.add_subplot(2,3,2)
                axes_Suu = fig_surfaces.add_subplot(Q+1,4,q*4+2)
                axes_simplex = fig_surfaces.add_subplot(Q+1,4,q*4+3)
                axes_simplex_no_u = fig_surfaces.add_subplot(Q+1,4,q*4+4)

                surface_color = axes1D_linear.plot(one_dimensional_grid, mean_surface[q], color = 'blue')[0].get_color()
                lower_error_surface, upper_error_surface = mean_surface[q] - 2 * std_surface[q], mean_surface[q] + 2 * std_surface[q]
                axes1D_linear.fill_between(one_dimensional_grid, lower_error_surface, upper_error_surface, color = surface_color, alpha = 0.2)

                sample_colour = lighten_color(surface_color, 1.6)
                for sample_on_grid in function_samples_on_grid[q]:
                    axes1D_linear.plot(one_dimensional_grid, sample_on_grid, color = sample_colour, alpha = 0.4)

                axes1D_linear.scatter(all_inducing_points[q], all_inducing_points_means[q], color = 'black', marker = 'o', s = 20)
                axes1D_linear.plot([-torch.pi, torch.pi], [pi_u_tildes[q].item(), pi_u_tildes[q].item()], surface_color, linestyle= '-.', linewidth = 3)
                axes1D_linear.plot([-torch.pi, torch.pi], [pi_1_tildes[q].item(), pi_1_tildes[q].item()], surface_color, linewidth = 3)

                if true_mean_surface is not None:
                    flattened_true_mean = true_mean_surface[q].flatten()
                    axes1D_linear.scatter(all_deltas.flatten(), flattened_true_mean, color = 'red', alpha = 0.4, s = 5)
                    if true_std_surface is not None:
                        flattened_true_std = true_std_surface[q].flatten()
                        axes1D_linear.scatter(all_deltas.flatten(), flattened_true_mean + 2 * flattened_true_std, color = 'red', alpha = 0.01, s = 5)
                        axes1D_linear.scatter(all_deltas.flatten(), flattened_true_mean - 2 * flattened_true_std, color = 'red', alpha = 0.01, s = 5)
                
                if self.inducing_point_variational_parameterisation == 'gaussian':
                    axes_Suu.imshow(all_inducing_points_covars[q], cmap = 'gray')

                for sep in [min_separation, max_separation]:
                    y_bot, y_top = axes1D_linear.get_ylim()
                    axes1D_linear.plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes1D_linear.plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes1D_linear.set_ylim(y_bot, y_top)
                    axes1D_linear.set_xlim(-torch.pi, torch.pi)

                if recent_component_priors is not None:
                    standard_swap_model_simplex_plots(recent_component_priors[q], axes_simplex, ax_no_u = axes_simplex_no_u)
                    legend_without_repeats(axes_simplex)
                    legend_without_repeats(axes_simplex_no_u)

            axes1D_linear.set_xlabel(deltas_label)

            return fig_surfaces

    def visualise_pdf_for_example(self, deltas_batch: _T, zeta_targets_batch: _T, theta_count = 360):
        """
        deltas_batch of shape [Q, 1, N, D]
        zeta_targets_batch of shape [1, N, 1]
        """
        assert zeta_targets_batch.shape[0] == 1
        
        theta_axis = torch.linspace(-torch.pi, torch.pi, theta_count + 1, device=deltas_batch.device)[:-1]
        theta_errors = rectify_angles(theta_axis.unsqueeze(1) - zeta_targets_batch[:,:,0]).unsqueeze(0).repeat(self.num_models, 1, 1)   # [Q, 360, N]
        ss = deltas_batch.shape[-2]

        with torch.no_grad():

            inference_info = self.get_elbo_terms(deltas_batch, None, 'none', 0, False, {})
            pi_vectors = inference_info['priors']       # [Q, 1, N+1]

            individual_component_log_likelihoods = self.generative_model.error_emissions.individual_component_likelihoods_from_estimate_deviations(ss, theta_errors) # [Q, 360, N+1]
            pdf_grid = individual_component_log_likelihoods * pi_vectors.repeat(1, theta_count, 1)        # [Q, 360, N+1]
            component_sums = pdf_grid.sum(-1)                                                               # [Q, 360] 
        
        return theta_axis, component_sums



class WorkingMemorySimpleSwapModel(WorkingMemoryFullSwapModel):
    
    def __init__(self, generative_model: NonParametricSwapErrorsGenerativeModel) -> None:
        super(WorkingMemoryFullSwapModel, self).__init__()

        self.generative_model = generative_model

    def get_elbo_terms_easier(self, data: Optional[_T], M: int, N: int, training_method, kwargs_for_individual_component_likelihoods = {}) -> Dict[str, Optional[_T]]:
        prior_info = self.generative_model.swap_function.generate_pi_vectors(set_size = N, batch_size = M)
        priors = prior_info['pis']

        if training_method == 'error':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = self.generative_model.get_marginalised_log_likelihood(
                estimation_deviations = data, pi_vectors = priors, kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
            )
        elif training_method == 'beta':
            raise Exception
            llh_term = self.generative_model.get_component_log_likelihood(
                selected_components = data, pi_vectors = pi_vectors
            )
            posterior, unaggregated_lh = None, None
        elif training_method == 'none':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = None, None, None

        distance_loss = torch.zeros(self.generative_model.num_models, device = priors.device, dtype = priors.dtype)

        return {
            'total_log_likelihood': total_log_likelihood,       # [Q]
            'likelihood_per_datapoint': likelihood_per_datapoint, # [Q, M]
            'posterior': posterior_vectors,     # [Q, M, N+1]
            'priors': priors,                   # [Q, M, N+1]
            'f_samples': None,
            'kl_term': distance_loss,           # [Q]
            'distance_loss': distance_loss      # [Q]
        }

    def get_elbo_terms(self, deltas: _T, data: Optional[_T], training_method: str = 'errors', max_variational_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}) -> Dict[str, Optional[_T]]:
        if data is not None:
            Q, M, N = data.shape
        else:
            Q, M, N, D = deltas.shape
        return self.get_elbo_terms_easier(data, M, N, training_method, kwargs_for_individual_component_likelihoods)
    
    def visualise_variational_approximation(self, *args, **kwargs):
        "TODO: implement this, but just for the simplex plots!"
        raise NotImplementedError
