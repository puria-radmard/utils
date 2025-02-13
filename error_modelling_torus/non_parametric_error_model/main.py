import torch
from torch import Tensor as _T
from torch.nn import Module, ModuleDict

from numpy import ndarray as _A

from typing import Dict, Optional, Callable, Any, List

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel, NonParametricSwapErrorsVariationalModelWithNonZeroMean, HierarchicalNonParametricSwapErrorsVariationalModelWrapper
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel, MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel, HierarchicalNonParametricSwapErrorsGenerativeModelWrapper, ErrorsEmissionsBase

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from math import log as mathlog
from functools import partialmethod

from purias_utils.util.api import return_as_obj
from purias_utils.util.plotting import standard_swap_model_simplex_plots, legend_without_repeats, lighten_color
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from torch.linalg import LinAlgError

try:
    from torch import vmap
except ImportError as ie:
    print('torch.vmap not imported!')
    print(ie)

cmap = plt.get_cmap('rainbow')
cNorm  = colors.Normalize(vmin=-torch.pi, vmax=torch.pi)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
scalarMap.set_array([])


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

        self.num_models = generative_model.num_models
        assert all([self.num_models == vm.num_models for vm in variational_models.values()])

    def get_variational_model(self, N: int) -> NonParametricSwapErrorsVariationalModel:
        return self.variational_models['0'] if self.shared_variational_model else self.variational_models[str(N)]

    def reduce_to_single_model(self, model_index: int = 0) -> None:
        try:
            for var_model in self.variational_models.values():
                var_model.reduce_to_single_model(model_index)
        except AttributeError:
            pass
        self.generative_model.reduce_to_single_model(model_index)
        self.num_models = 1

    @return_as_obj
    def minibatched_inference(self, deltas: _T, max_variational_batch_size = 0, take_samples = True, conditioned_on_variational_samples: Optional[_T] = None, *_, override_generative_model: Optional[NonParametricSwapErrorsGenerativeModel] = None):
        """
        If you provide u samples to condition on, they must be of shape [Q, K, R], checked in variational_gp_inference_conditioned_on_inducing_points_function

        TODO: better documenting the difference between conditioned_on_variational_samples == None and conditioned_on_variational_samples provided
        """

        Q, M_all, N, D = deltas.shape
        variational_model = self.get_variational_model(N)

        if override_generative_model is None:
            override_generative_model = self.generative_model

        I = self.num_variational_samples
        assert Q == variational_model.num_models
        assert D == variational_model.num_features

        all_deduplicated_deltas, M_minis = variational_model.deduplicate_deltas(deltas, max_variational_batch_size)  # "~M/batch_size length list of entries of shape [Q, ~batch*N, D]"

        # Use kernel all here:
        inducing_locations = variational_model.Z
        K_dds = [override_generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]                         # each [Q, ~batch*N, ~batch*N]
        K_uu = override_generative_model.swap_function.evaluate_kernel(N, inducing_locations)                                                                             # [Q, R, R]
        k_uds = [override_generative_model.swap_function.evaluate_kernel(N, inducing_locations, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]    # each [Q, R, ~batch*N]

        # Inverse isn't always symmetric!!
        K_uu_inv = torch.linalg.inv(K_uu)
        if not torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all():
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))

        # Make variational inferences for q(f)
        mus, sigmas, sigma_chols = [], [], []
        for k_ud, K_dd in zip(k_uds, K_dds):
            # [Q, ~batch*N], [Q, ~batch*N, ~batch*N], [Q, ~batch*N, ~batch*N]
            if conditioned_on_variational_samples is None:
                _mu, _sigma, _sigma_chol = variational_model.variational_gp_inference(k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv)
            else:
                _mu, _sigma, _sigma_chol = variational_model.variational_gp_inference_conditioned_on_inducing_points_function(u=conditioned_on_variational_samples, k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv)
            mus.append(_mu)
            sigma_chols.append(_sigma_chol)
            sigmas.append(_sigma.detach())
        
        # For fmse usage
        ### if conditioned_on_variational_samples is None: XXX: not sure why this was here before!
        with torch.no_grad():
            mean_surface = torch.concat([
                variational_model.reinclude_model_evals(mu.unsqueeze(1), M_batch, N, 1).squeeze(1)
                for mu, M_batch in zip(mus, M_minis)
            ], dim = 1)    # [Q, M, N]

            std_surface = torch.concat([
                variational_model.reinclude_model_evals(torch.diagonal(sigma, dim1=-1, dim2=-2).unsqueeze(1), M_batch, N, 1).squeeze(1).sqrt()
                for sigma, M_batch in zip(sigmas, M_minis)
            ], dim = 1)    # [Q, M, N]
        ### else:
        ###     mean_surface = None
        ###     std_surface = None

        f_samples = None

        if take_samples:
            # Get the samples of f evaluated at the deltas
            if conditioned_on_variational_samples is None:
                all_f_samples = [
                    variational_model.reparameterised_sample(num_samples = I, mu = mu, sigma_chol = sigma_chol, M = M, N = N)
                    for mu, sigma_chol, M in zip(mus, sigma_chols, M_minis)
                ]   # Each of shape [Q, I, ~batchsize, N]
            else:
                all_f_samples = [
                    variational_model.reparameterised_sample(num_samples = 1, mu = mu, sigma_chol = sigma_chol, M = M, N = N, unsqueeze_mu=False)
                    for mu, sigma_chol, M in zip(mus, sigma_chols, M_minis)
                ]   # Each of shape [Q, K, ~batchsize, N]                

            # Shouldn't be any numerical problems after this
            f_samples = torch.concat(all_f_samples, 2)  # [Q, I or K, M, N]

        return {
            "mus": mus, "sigmas": sigmas, "sigma_chols": sigma_chols, "f_samples": f_samples,
            "K_uu": K_uu, "k_uds": k_uds, "K_uu_inv": K_uu_inv, "K_dds": K_dds, "M_minis": M_minis,
            "mean_surface": mean_surface, "std_surface": std_surface,
        }

    def _inference_from_neural_samples_inner(self, single_priors: _T, single_sample_errors: _T, kwargs_for_get_marginalised_log_likelihood = {}) -> _T:
        """
        Outside the vmap, takes in priors of shape [1, M, N + 1] and neural samples of shape [M, N, K] 
        Inside vmap, these come as [1, N+1] and [M, N] so are unsqueezed appropriately here

        TODO: maybe could do something with set typicality here?
        """
        total_log_likelihood, likelihood_per_sample, posterior_vectors = self.generative_model.get_marginalised_log_likelihood(
            estimation_deviations = single_sample_errors.unsqueeze(0),  # Should automatically fail if self.num_models != 1
            pi_vectors = single_priors.unsqueeze(1),
            kwargs_for_individual_component_likelihoods = {},
            **kwargs_for_get_marginalised_log_likelihood
        )
        return posterior_vectors[0]    # [M, N+1], which is vmapped to [M, N+1, K]

    try:
        inference_from_neural_samples: Callable[[_T, _T], _T] = vmap(
            func = _inference_from_neural_samples_inner,
            in_dims = (None, None, 2),
            out_dims = -2,
            chunk_size = None
        )
    except NameError as e:
        def inference_from_neural_samples(self, *args, **kwargs) -> _T:
            raise NotImplementedError('inference_from_neural_samples requires torch.vmap!')

    def get_elbo_terms(self, deltas: _T, data: Optional[_T], training_method: str = 'error', max_variational_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}, kwargs_for_get_marginalised_log_likelihood = {}) -> Dict[str, Optional[_T]]:

        Q, M_all, N, D = deltas.shape
        variational_model = self.get_variational_model(N)

        minibatched_inference_info = self.minibatched_inference(deltas, max_variational_batch_size, True)

        prior_info = self.generative_model.swap_function.generate_pi_vectors(
            set_size = N, model_evaulations = minibatched_inference_info.f_samples
        )
        priors = prior_info['pis']

        # Get the KL term of the loss
        if return_kl:
            kl_term = variational_model.kl_loss(K_uu = minibatched_inference_info.K_uu, K_uu_inv=minibatched_inference_info.K_uu_inv)     # [Q]
        else:
            kl_term = torch.ones(Q) * torch.nan # Won't plot!

        # Get the ELBO first term, depending on training mode (data is usually errors)
        if training_method == 'error':
            assert (Q, M_all, N) == tuple(data.shape)
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = self.generative_model.get_marginalised_log_likelihood(
                estimation_deviations = data,
                pi_vectors = priors,
                kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods,
                **kwargs_for_get_marginalised_log_likelihood
            )
        elif training_method == 'beta':
            raise NotImplementedError
            llh_term = self.generative_model.get_component_log_likelihood(
                selected_components = data, pi_vectors = pi_vectors
            )
            posterior, unaggregated_lh = None, None
        elif training_method == 'none':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = None, None, None

        if D > 0:
            distance_loss = minibatched_inference_info.K_uu.tril(-1).max() / minibatched_inference_info.K_uu.max()
        else:
            distance_loss = torch.tensor(0.0)

        return {
            'total_log_likelihood': total_log_likelihood, 
            'likelihood_per_datapoint': likelihood_per_datapoint, 
            'posterior': posterior_vectors, 
            'priors': priors, 
            'f_samples': minibatched_inference_info.f_samples,             # [Q, I, M, N]
            'kl_term': kl_term, 
            'distance_loss': distance_loss,
            'mean_surface': minibatched_inference_info.mean_surface,
            'std_surface': minibatched_inference_info.std_surface,
        }

    def inference(self, deltas: _T, N_override: Optional[int] = None, *_, override_generative_model: Optional[NonParametricSwapErrorsGenerativeModel] = None):
        
        if override_generative_model is None:
            override_generative_model = self.generative_model

        Q, M, N, D = deltas.shape
        if N_override is not None:
            N = N_override
        assert Q == self.num_models
        variational_model = self.get_variational_model(N)

        deduplicated_deltas = variational_model.deduplicate_deltas(deltas, 0)[0][0]

        with torch.no_grad():

            R = variational_model.R
            K_dd = override_generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas)
            K_uu = override_generative_model.swap_function.evaluate_kernel(N, variational_model.Z)
            k_ud = override_generative_model.swap_function.evaluate_kernel(N, variational_model.Z, deduplicated_deltas)
            K_uu_inv = torch.linalg.inv(K_uu)
            # K_dd_inv = torch.linalg.inv(K_dd)

            K_uu_inv = torch.linalg.inv(K_uu)
            # assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
            assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

            # Make variational inferences for q(f)
            mu, sigma, sigma_chol = variational_model.variational_gp_inference(
                k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
            )

        return {
            'mu': mu,
            'sigma': sigma,
            'sigma_chol': sigma_chol,
            'K_dd': K_dd,
            'K_uu': K_uu,
            'k_ud': k_ud,
            'K_uu_inv': K_uu_inv,
        }

    def inference_on_grid(self, set_size: int, grid_count: int, device: str = 'cuda') -> Dict[str, Optional[_A]]:
        """
        Set up a suitably shaped grid and perform inference over it

        TODO: shapes
        """
        variational_model = self.get_variational_model(set_size)
        D = variational_model.num_features

        if D == 0:
            mean_surface = variational_model.swap_logit_mean.detach().cpu().numpy()                 # [Q, 1]
            std_surface = variational_model.swap_logit_std.detach().cpu().numpy()                   # [Q, 1]
            return {
                'mean_surface': mean_surface,
                'std_surface': std_surface,
                'all_grid_points': None,
                'one_dimensional_grid': None,
                'function_samples_on_grid': None
            }

        if D > 2:
            raise NotImplementedError

        full_grid = torch.linspace(-torch.pi, +torch.pi, grid_count)
        grid_locs = torch.tensor([0.0]) if D == 1 else full_grid.clone()
        grid_cols = full_grid.clone()
        grid_x, grid_y = torch.meshgrid(grid_locs, grid_cols, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y], -1).reshape(len(grid_cols) * len(grid_locs), 2).to(device)
        grid_points = grid_points[...,[1]] if D == 1 else grid_points

        grid_points = grid_points.unsqueeze(0).unsqueeze(2).repeat(self.num_models, 1, 2, 1)
        grid_points[:,:,0,:] = 0.0  # Doesn't change anything but nice to see when debugging
        inference_info = self.inference(grid_points, N_override=set_size)
        flat_mu_est, flat_sigma_est, sigma_chol = inference_info['mu'], inference_info['sigma'], inference_info['sigma_chol']
        if not variational_model.fix_non_swap:
            raise NotImplementedError

        std_est = torch.diagonal(flat_sigma_est, dim1=-1, dim2=-2).sqrt()
        eps = torch.randn(self.generative_model.num_models, 3, flat_mu_est.shape[1], dtype = flat_mu_est.dtype, device = flat_mu_est.device) # [Q, 3, MN (dedup size)]
        grid_f_samples = flat_mu_est.unsqueeze(1) + torch.bmm(eps, sigma_chol.transpose(-1, -2))   # [Q, 3, MN]

        return {
            'one_dimensional_grid': full_grid.cpu().numpy(),            # [Q, grid_points]
            'all_grid_points': grid_points.cpu().numpy(),               # [Q, grid_points, 2, D]
            'mean_surface': flat_mu_est.cpu().numpy(),                  # [Q, grid_points...]
            'std_surface': std_est.cpu().numpy(),                       # [Q, grid_points...]
            'function_samples_on_grid': grid_f_samples.cpu().numpy()    # [Q, 3, grid_points...]
        }

    def _no_dim_visualise_variational_approximation(
        self, set_size: int, pi_u_tildes: _A, pi_1_tildes: _A, logit_means: _A, logit_stds: _A, recent_component_priors: Optional[_A],
    ):
        Q = self.num_models
        num_cols = 3
        figsize = 8
        fig_surfaces = plt.figure(figsize = (figsize * num_cols, figsize))

        axes_simplex = fig_surfaces.add_subplot(1,num_cols,1)
        axes_simplex_no_u = fig_surfaces.add_subplot(1,num_cols,2)
        axes_logits = fig_surfaces.add_subplot(1,num_cols,3)

        q_axis = torch.arange(0, self.num_models)

        if pi_u_tildes is not None:
            axes_logits.scatter(q_axis, pi_u_tildes, marker='s', label = '$\\tilde{\pi}_u$')
        if pi_1_tildes is not None:
            axes_logits.scatter(q_axis, pi_1_tildes, marker='s', label = '$\\tilde{\pi}_1$')
        
        logit_means = logit_means.squeeze(-1)
        logit_stds = logit_stds.squeeze(-1)
        
        swap_color = axes_logits.plot(q_axis, logit_means, marker='x', linestyle='none', label = '$\\tilde{\pi}_{swap}$')[0].get_color()

        upper = logit_means + logit_stds
        lower = logit_means - logit_stds

        for q, q_x in enumerate(q_axis):

            axes_logits.plot([q_x,q_x], [lower[q], upper[q]], marker = '_', color = swap_color)

            if recent_component_priors is not None:
                standard_swap_model_simplex_plots(recent_component_priors[q], axes_simplex, ax_no_u = axes_simplex_no_u)
                legend_without_repeats(axes_simplex)
                legend_without_repeats(axes_simplex_no_u)

        return fig_surfaces, 1, num_cols

    def visualise_variational_approximation(
        self, set_size: int, grid_count: int, pi_u_tildes: _A, pi_1_tildes: _A,
        all_deltas: _A, recent_component_priors: Optional[_A], true_mean_surface: Optional[_A], true_std_surface: Optional[_A],
        min_separation: float, max_separation: float, deltas_label: str
    ):
        """
        Input:
            All taken from util.inference_on_grid

        If D = 0:
            TODO: list everything!

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

        variational_model = self.get_variational_model(set_size)
        Q = self.num_models
        D = variational_model.num_features

        if D == 0:
            return self._no_dim_visualise_variational_approximation(
                set_size, pi_u_tildes, pi_1_tildes, mean_surface, std_surface, recent_component_priors
            )

        num_cols = 4 if D == 1 else 6
        num_rows = Q + 1
        figsize = 8
        fig_surfaces = plt.figure(figsize = (figsize * num_cols, figsize * num_rows))

        assert len(min_separation) == len(max_separation) == len(deltas_label) == D

        if D > 2:
            raise NotImplementedError

        all_inducing_points = variational_model.Z.detach().cpu().squeeze(-1).numpy()        # [Q, R, D]
        all_inducing_points_means = variational_model.m_u.detach().cpu().numpy()            # [Q, R]
        if variational_model.inducing_point_variational_parameterisation == 'gaussian':
            all_inducing_points_covars = variational_model.S_uu.detach().cpu().numpy()      # [Q, R, R]

        for q in range(Q):

            qth_mean_surface = mean_surface[q]
            qth_lower_surface, qth_upper_surface = qth_mean_surface - 2 * std_surface[q], qth_mean_surface + 2 * std_surface[q]

            if true_mean_surface is not None:
                qth_true_mean = true_mean_surface[q]
                if true_std_surface is not None:
                    qth_true_std = true_std_surface[q]
                    qth_true_lower, qth_true_upper = qth_true_mean - 2 * qth_true_std, qth_true_mean + 2 * qth_true_std
                else:
                    qth_true_lower, qth_true_upper = float('nan') * qth_true_mean, float('nan') * qth_true_mean

            # axes1D_exponentiated = fig_surfaces.add_subplot(2,3,2)
            axes_simplex = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+1)
            axes_simplex_no_u = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+2)
            axes_Suu = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+3)

            if variational_model.inducing_point_variational_parameterisation == 'gaussian':
                axes_Suu.imshow(all_inducing_points_covars[q], cmap = 'gray')

            if recent_component_priors is not None:
                standard_swap_model_simplex_plots(recent_component_priors[q], axes_simplex, ax_no_u = axes_simplex_no_u)
                legend_without_repeats(axes_simplex)
                legend_without_repeats(axes_simplex_no_u)

            if D == 1:

                axes_hist = fig_surfaces.add_subplot(num_rows,num_cols,num_cols*num_rows)
                axes_hist.hist(all_deltas[:,1:].flatten(), 1024, density=True)
                axes_hist.set_xlabel(deltas_label[0])

                axes_surface = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+4)

                surface_color = axes_surface.plot(one_dimensional_grid, qth_mean_surface, color = 'blue')[0].get_color()
                axes_surface.fill_between(one_dimensional_grid, qth_lower_surface, qth_upper_surface, color = surface_color, alpha = 0.2)

                sample_colour = lighten_color(surface_color, 1.6)
                for sample_on_grid in function_samples_on_grid[q]:
                    axes_surface.plot(one_dimensional_grid, sample_on_grid, color = sample_colour, alpha = 0.4)

                axes_surface.scatter(all_inducing_points[q], all_inducing_points_means[q], color = 'black', marker = 'o', s = 20)
                axes_surface.plot([-torch.pi, torch.pi], [pi_u_tildes[q].item(), pi_u_tildes[q].item()], surface_color, linestyle= '-.', linewidth = 3)
                axes_surface.plot([-torch.pi, torch.pi], [pi_1_tildes[q].item(), pi_1_tildes[q].item()], surface_color, linewidth = 3)

                if true_mean_surface is not None:
                    axes_surface.scatter(all_deltas.flatten(), qth_true_mean.flatten(), color = 'red', alpha = 0.4, s = 5)
                    axes_surface.scatter(all_deltas.flatten(), qth_true_lower.flatten(), color = 'red', alpha = 0.01, s = 5)
                    axes_surface.scatter(all_deltas.flatten(), qth_true_upper.flatten(), color = 'red', alpha = 0.01, s = 5)

                for sep in [min_separation[0], max_separation[0]]:
                    y_bot, y_top = axes_surface.get_ylim()
                    axes_surface.plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes_surface.plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes_surface.set_ylim(y_bot, y_top)
                    axes_surface.set_xlim(-torch.pi, torch.pi)

                axes_surface.set_xlabel(deltas_label[0])
            
            elif D == 2:

                axes_surface = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+4,projection='3d')
                axes_slice_0 = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+5)
                axes_slice_1 = fig_surfaces.add_subplot(num_rows,num_cols,q*num_cols+6)

                axes_surface.set_xlabel('cued: ' + deltas_label[0])
                axes_surface.set_ylabel('estimated: ' + deltas_label[1])
                axes_slice_0.set_xlabel('cued: ' + deltas_label[0])
                axes_slice_1.set_xlabel('estimated: ' + deltas_label[1])

                qth_grid_points = all_grid_points[q,:,1,:]            # [Q, grid_points, 2, D = 2] -> [grid_points, D = 2]
                qth_grid_points_x, qth_grid_points_y = qth_grid_points.reshape(grid_count, grid_count, 2).transpose(2, 0, 1)    # Each [sqrt grid_points, sqrt grid_points]
                structured_qth_mean_surface = qth_mean_surface.reshape(grid_count, grid_count)

                axes_surface.plot_surface(qth_grid_points_x, qth_grid_points_y, structured_qth_mean_surface, color='blue')
                axes_surface.plot_surface(qth_grid_points_x, qth_grid_points_y, qth_lower_surface.reshape(grid_count, grid_count), color='blue', alpha = 0.5)
                axes_surface.plot_surface(qth_grid_points_x, qth_grid_points_y, qth_upper_surface.reshape(grid_count, grid_count), color='blue', alpha = 0.5)

                axes_surface.scatter(*all_inducing_points[q].T, all_inducing_points_means[q], color = 'black', marker = 'o', s = 20)
                axes_surface.scatter([0.0], [0.0], [pi_u_tildes[q].item()], color = 'blue', marker='x')
                axes_surface.scatter([0.0], [0.0], [pi_1_tildes[q].item()], color = 'blue', marker='o')

                if true_mean_surface is not None:
                    try:
                        axes_surface.scatter(all_deltas.flatten(), qth_true_mean.flatten(), color = 'red', alpha = 0.4, s = 5)
                        axes_surface.scatter(all_deltas.flatten(), qth_true_lower.flatten(), color = 'red', alpha = 0.01, s = 5)
                        axes_surface.scatter(all_deltas.flatten(), qth_true_upper.flatten(), color = 'red', alpha = 0.01, s = 5)
                    except ValueError:
                        print("One time fix REMOVED for the sake of non_parametric_model/commands/fc_single_subject_summaries/one_off_lobe_check_on_mcmaster2022_e2_dircue_lowC.sh !!! Requires a permanent solution")
                        # axes_surface.scatter(all_deltas[...,0].flatten(), all_deltas[...,0].flatten()*0, qth_true_mean.flatten(), color = 'red', alpha = 0.4, s = 5)
                        # axes_surface.scatter(all_deltas[...,0].flatten(), all_deltas[...,0].flatten()*0, qth_true_lower.flatten(), color = 'red', alpha = 0.01, s = 5)
                        # axes_surface.scatter(all_deltas[...,0].flatten(), all_deltas[...,0].flatten()*0, qth_true_upper.flatten(), color = 'red', alpha = 0.01, s = 5)


                for g in range(grid_count):
                    slice_color = scalarMap.to_rgba(one_dimensional_grid[g])
                    mean_x_slice, mean_y_slice = structured_qth_mean_surface[:,g], structured_qth_mean_surface[g,:]
                    axes_slice_0.plot(one_dimensional_grid, mean_x_slice, color = slice_color)
                    axes_slice_1.plot(one_dimensional_grid, mean_y_slice, color = slice_color)
                
                for ax_idx, ax in enumerate([axes_slice_0, axes_slice_1]):
                    for sep in [min_separation[ax_idx], max_separation[ax_idx]]:
                        y_bot, y_top = ax.get_ylim()
                        ax.plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
                        ax.plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
                        ax.set_ylim(y_bot, y_top)
                        ax.set_xlim(-torch.pi, torch.pi)
                    ax.plot([-torch.pi, torch.pi], [pi_u_tildes[q].item(), pi_u_tildes[q].item()], 'blue', linestyle= '-.', linewidth = 3)
                    ax.plot([-torch.pi, torch.pi], [pi_1_tildes[q].item(), pi_1_tildes[q].item()], 'blue', linewidth = 3)

        return fig_surfaces, num_rows, num_cols

    def visualise_pdf_for_example(self, deltas_batch: _T, zeta_targets_batch: _T, theta_count = 360, pi_vectors: Optional[_T] = None, *_, override_error_emissions: Optional[ErrorsEmissionsBase] = None, error_emissions_key: Optional[Any] = None):
        """
        deltas_batch of shape [Q, 1, N, D]
        zeta_targets_batch of shape [1, N, 1]
        pi_vectors of shape [Q, 1, N+1] if given
        """
        if override_error_emissions is None:
            override_error_emissions = self.generative_model.error_emissions

        assert zeta_targets_batch.shape[0] == 1
        
        theta_axis = torch.linspace(-torch.pi, torch.pi, theta_count + 1, device=deltas_batch.device)[:-1]
        theta_errors = rectify_angles(theta_axis.unsqueeze(1) - zeta_targets_batch[:,:,0]).unsqueeze(0).repeat(self.num_models, 1, 1)   # [Q, 360, N]
        ss = deltas_batch.shape[-2]

        with torch.no_grad():

            if pi_vectors is None:
                inference_info = self.get_elbo_terms(deltas_batch, None, 'none', 0, False, {}) if error_emissions_key is None else self.get_elbo_terms(error_emissions_key, deltas_batch, None, 'none', 0, False, {})
                pi_vectors = inference_info['priors']       # [Q, 1, N+1]
            else:
                assert tuple(pi_vectors.shape) == (self.num_models, 1, ss + 1)

            individual_component_likelihoods = override_error_emissions.individual_component_likelihoods_from_estimate_deviations(ss, theta_errors) # [Q, 360, N+1]
            pdf_grid = individual_component_likelihoods * pi_vectors.repeat(1, theta_count, 1)        # [Q, 360, N+1]
            component_sums = pdf_grid.sum(-1)                                                               # [Q, 360] 
        
        return theta_axis, component_sums

    def refined_likelihood_estimate(self, all_errors: _T, all_deltas: _T, training_indices: _T, max_variational_batch_size = 0, **kwargs):
        """
        Please see LaTeX documentation for all of this info!

        Inputs:
            all_errors of shape [Q, M^* + M, N]
            all_deltas of shape [Q, M^* + M, N, D] i.e. both the training dataset and the new dataset combined!
            training_indices of shape [Q, M] is just indices of where in all_deltas there is training data
        
        Returns:
            all llh estimates of shape [Q, M^* + M] - can separate this as you wish downstream
        """
        raise Exception('Dont use this method anymore')
        
        Q, Mtotal, N, D = all_deltas.shape
        assert tuple(all_errors.shape) == (Q, Mtotal, N)
        M_train = training_indices.shape[1]
        assert tuple(training_indices.shape) == (Q, M_train)
        variational_model = self.get_variational_model(N)
        assert Q == self.num_models and D == variational_model.num_features
        K = self.num_importance_sampling_samples
        R = variational_model.R

        with torch.no_grad():

            #### First term - the importance adjusted estimate of joint (numerator of posterior) of shape [Q, M]

            # u^k ~ variational "prior"
            variational_u_samples_info = variational_model.sample_from_variational_prior(K)
            variational_u_samples = variational_u_samples_info['samples']                                               # [Q, K, R]
            variational_u_sample_llhs = variational_u_samples_info['sample_log_likelihoods'].mean(-1, keepdim=True)     # [Q, 1]

            # f^k ~ variational posterior
            # We don't get_elbo_terms becausewe don't want to do self.generative_model.get_marginalised_log_likelihood yet - we do aggregation after the log etc, not before
            variational_inference_terms = self.minibatched_inference(all_deltas, max_variational_batch_size=max_variational_batch_size, take_samples=True, conditioned_on_variational_samples=variational_u_samples)
            variational_generated_component_prior_info = self.generative_model.swap_function.generate_pi_vectors(set_size = N, model_evaulations = variational_inference_terms.f_samples, mc_average = False) 
            variational_generated_component_priors: _T = variational_generated_component_prior_info['pis']

            # log p(y | Z, f^i, phi)
            # because we ask for the datasets to be 'combined' we will:
                # a) automatically calculate this llh estimate on all the data (training data or not)
                # b) account for two terms in the derivation here
            individual_component_downstream_likelihoods: _T = self.generative_model.error_emissions.individual_component_likelihoods_from_estimate_deviations(      # [Q, Mtot, N+1] - p(y[m] | beta[n], Z[m])
                set_size = N, estimation_deviations = all_errors
            )
            joint_component_and_error_likelihood_variational = individual_component_downstream_likelihoods.unsqueeze(1) * variational_generated_component_priors    # [Q, K, Mtot, N+1] - p(y[m] | beta[n], Z[m]) * p(beta[n]| f[k], Z[m]) = p(y[m], beta[n] | f[k], Z[m])
            variational_conditioned_log_likelihood_per_datapoint = joint_component_and_error_likelihood_variational.sum(-1).log()                                   # [Q, K, Mtot] - log p(y[m], | f[k], Z[m])

            # Importance weighting term
            u_samples_llh_under_prior = self.generative_model.swap_function.sample_from_prior(N, variational_model.Z, self.num_importance_sampling_samples)['sample_log_likelihoods'].mean(-1, keepdim=True) # [Q, 1]
            log_importance_weighting = u_samples_llh_under_prior - variational_u_sample_llhs

            # Finally (for this term...), construct the first term of the estimate, which is all an MC estimate
            training_indices = training_indices.unsqueeze(1).repeat(1, K, 1).to(variational_generated_component_priors.device)
            variational_log_likelihood_on_train_set: _T = variational_conditioned_log_likelihood_per_datapoint.gather(-1, training_indices)                 # [Q, K, Mtot] -> [Q, K, M_train]
            assert tuple(variational_log_likelihood_on_train_set.shape) == (Q, K, M_train)
            variational_log_likelihood_on_train_set_total = variational_log_likelihood_on_train_set.sum(-1, keepdim=True).mean(1)                                   # [Q, 1]
            
            adjusted_variational_conditioned_log_likelihood_per_datapoint = (                   # [Q, Mtot]
                variational_conditioned_log_likelihood_per_datapoint.mean(1)        # [Q, Mtot]
                + log_importance_weighting                                          # [Q, 1] 
                - variational_log_likelihood_on_train_set_total                     # [Q, 1]
            )

            #### Second term - the "partition function" of size [Q, 1]
            # g^j ~ GP prior
            zero_mus = [torch.zeros(K_dd.shape[0], K_dd.shape[1], dtype = K_dd.dtype, device = K_dd.device) for K_dd in variational_inference_terms.K_dds]
            prior_f_samples_batched = [
                variational_model.reparameterised_sample(num_samples = K, mu = zmu, sigma_chol = torch.linalg.cholesky(K_dd), M = M, N = N)
                for zmu, K_dd, M in zip(zero_mus, variational_inference_terms.K_dds, variational_inference_terms.M_minis)
            ]                                                                                                                                               # Each of shape [Q, K, ~batchsize, N]
            prior_conditioned_component_priors = self.generative_model.swap_function.generate_pi_vectors(                                                         # [Q, K, Mtot, N+1]
                set_size = N, model_evaulations = torch.concat(prior_f_samples_batched, 2), mc_average = False
            )['pis']

            # Finally, aggregate to the full model log-evidence under the prior
            joint_component_and_error_likelihood_prior = individual_component_downstream_likelihoods.unsqueeze(1) * prior_conditioned_component_priors      # [Q, K, Mtot, N+1] - p(y[m] | beta[n], Z[m]) * p(beta[n]| f[k], Z[m]) = p(y[m], beta[n] | f[k], Z[m])
            prior_conditioned_likelihood_per_datapoint: _T = joint_component_and_error_likelihood_prior.sum(-1)                                             # [Q, K, Mtot]
            parition_function = prior_conditioned_likelihood_per_datapoint.mean(1).log().sum(1, keepdim=True)                                               # [Q, K, Mtot] -> [Q, Mtot] -> [Q, 1]


        return {
            "parition_function": parition_function,
            "adjusted_variational_conditioned_log_likelihood_per_datapoint": adjusted_variational_conditioned_log_likelihood_per_datapoint,
            "importance_sampled_log_likelihoods": adjusted_variational_conditioned_log_likelihood_per_datapoint - parition_function
        }
        
    def check_Kuu_stability(self, ):
        for N, variational_model in self.variational_models.items():
            try:
                K_uu = self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z)
                K_uu_inv = torch.linalg.inv(K_uu)
                K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
                K_uu_chol = torch.linalg.cholesky(K_uu)
            except LinAlgError as e:
                print(f'check_Kuu_stability: {e}')
                return False
        return True



class MultipleErrorEmissionsWorkingMemoryFullSwapModel(WorkingMemoryFullSwapModel):

    generative_model: MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel

    def refined_likelihood_estimate(self, *args, **kwargs) -> _T:
        raise NotImplementedError

    def visualise_pdf_for_example(self, error_emissions_key: Any, deltas_batch: _T, zeta_targets_batch: _T, theta_count=360, pi_vectors: Optional[_T] = None):
        override_error_emissions = self.generative_model.get_error_emissions(error_emissions_key)
        return super().visualise_pdf_for_example(deltas_batch, zeta_targets_batch, theta_count, pi_vectors, override_error_emissions=override_error_emissions, error_emissions_key = error_emissions_key)

    def _inference_from_neural_samples_inner(self, error_emissions_key: Any, single_priors: _T, single_sample_errors: _T) -> _T:
        kwargs_for_get_marginalised_log_likelihood = {"error_emissions_key": error_emissions_key}
        return super()._inference_from_neural_samples_inner(single_priors, single_sample_errors, kwargs_for_get_marginalised_log_likelihood)

    def get_elbo_terms(self, error_emissions_key: Any, deltas: _T, data: Optional[_T], training_method: str = 'error', max_variational_batch_size=0, return_kl=True, kwargs_for_individual_component_likelihoods={}) -> Dict[str, Optional[_T]]:
        kwargs_for_get_marginalised_log_likelihood = {'error_emissions_key': error_emissions_key}
        return super().get_elbo_terms(deltas, data, training_method, max_variational_batch_size, return_kl, kwargs_for_individual_component_likelihoods, kwargs_for_get_marginalised_log_likelihood)

    def drop_error_emissions(self, error_emissions_key):
        self.generative_model.drop_error_emissions(error_emissions_key)



class HierarchicalWorkingMemoryFullSwapModel(WorkingMemoryFullSwapModel):

    def __init__(
        self, 
        generative_model: HierarchicalNonParametricSwapErrorsGenerativeModelWrapper,
        variational_models: Dict[int, HierarchicalNonParametricSwapErrorsVariationalModelWrapper],
        num_variational_samples: int, num_importance_sampling_samples: int
    ) -> None:
        super(WorkingMemoryFullSwapModel, self).__init__()

        self.generative_model = generative_model

        # Still indexing over set size here!
        if 0 in variational_models.keys():
            assert len(variational_models) == 1
            self.shared_variational_model = True
        else:
            self.shared_variational_model = False
        self.variational_models: Dict[int, HierarchicalNonParametricSwapErrorsVariationalModelWrapper] = ModuleDict(variational_models)

        self.num_variational_samples = num_variational_samples  # I
        self.num_importance_sampling_samples = num_importance_sampling_samples  # K

        self.num_models = generative_model.num_models
        assert all([self.num_models == vm.num_models for vm in variational_models.values()])

        assert all(set(generative_model.submodel_keys) == set(v.submodel_keys) for v in variational_models.values())
        self.submodel_keys = generative_model.submodel_keys.copy()
        assert None not in self.submodel_keys
    
    def drop_submodel(self, submodel_key):
        self.generative_model.drop_submodel(submodel_key)
        self.submodel_keys.remove(submodel_key)
        for variational_models in self.variational_models.values():
            variational_models.drop_submodel(submodel_key)

    def get_variational_model(self, N: int) -> HierarchicalNonParametricSwapErrorsVariationalModelWrapper:
        "Just changing typehinting here"
        return super().get_variational_model(N)

    def get_variational_submodel(self, N: int, submodel_key: Any) -> NonParametricSwapErrorsVariationalModelWithNonZeroMean:
        return self.get_variational_model(N).submodels[str(submodel_key)]

    def get_generative_submodel(self, submodel_key: Any) -> NonParametricSwapErrorsGenerativeModel:
        "Just changing typehinting here"
        return self.generative_model.submodel_generative_models[str(submodel_key)]

    def minibatched_inference(self, *args, **kwargs):
        raise TypeError('Use minibatched_inference_primary and minibatched_inference_submodel instead!')

    def minibatched_inference_primary(self, deltas: _T, max_variational_batch_size=0, take_samples=True, conditioned_on_primary_variational_samples: Optional[_T] = None):
        return super(HierarchicalWorkingMemoryFullSwapModel, self).minibatched_inference(
            deltas=deltas, max_variational_batch_size=max_variational_batch_size, 
            take_samples=take_samples, conditioned_on_variational_samples=conditioned_on_primary_variational_samples,
            override_generative_model=self.generative_model.primary_generative_model
        )
    
    @return_as_obj
    def minibatched_inference_submodel(
        self, submodel_key: Any, deltas: _T, primary_mus: List[_T], primary_sigmas: List[_T], primary_k_uds: List[_T], primary_K_uu_inv: _T,
        max_variational_batch_size = 0, take_samples = True, 
        conditioned_on_sub_variational_samples: Optional[_T] = None
    ):
        """
        primary_mnus, primary_sigmas are from minibatched_inference_primary
        Samples are drawn from total posterior, not conditioned on any samples from before!!
        """
        Q, M_all, N, D = deltas.shape
        variational_model = self.get_variational_model(N)

        I = self.num_variational_samples
        assert Q == variational_model.num_models
        assert D == variational_model.num_features

        all_deduplicated_deltas, M_minis = variational_model.deduplicate_deltas(deltas, max_variational_batch_size)  # "~M/batch_size length list of entries of shape [Q, ~batch*N, D]"

        # Use sub-level kernel all here
        inducing_locations = variational_model.Z
        submodel_K_dds = [self.generative_model.submodel_generative_models[str(submodel_key)].swap_function.evaluate_kernel(N, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]                         # each [Q, ~batch*N, ~batch*N]
        submodel_K_uu = self.generative_model.submodel_generative_models[str(submodel_key)].swap_function.evaluate_kernel(N, inducing_locations)                                                                             # [Q, R, R]
        submodel_k_uds = [self.generative_model.submodel_generative_models[str(submodel_key)].swap_function.evaluate_kernel(N, inducing_locations, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]    # each [Q, R, ~batch*N]

        # Inverse isn't always symmetric!!
        submodel_K_uu_inv = torch.linalg.inv(submodel_K_uu)
        if not torch.isclose(submodel_K_uu_inv, submodel_K_uu_inv.transpose(1, 2)).all():
            submodel_K_uu_inv_chol = torch.linalg.cholesky(submodel_K_uu_inv)
            submodel_K_uu_inv = torch.bmm(submodel_K_uu_inv_chol, submodel_K_uu_inv_chol.transpose(1, 2))
        
        submodel_mus, submodel_sigmas = [], []
        cross_sigmas = []
        complete_mus, complete_sigmas, complete_sigma_chols = [], [], []
        for primary_mu, primary_sigma, primary_k_ud, k_ud, K_dd in zip(primary_mus, primary_sigmas, primary_k_uds, submodel_k_uds, submodel_K_dds):
            if conditioned_on_sub_variational_samples is None:
                _sub_mu, _sub_sigma, _cross_sigma, _tot_mu, _tot_sigma, _tot_sigma_chol = variational_model.variational_gp_inference_submodel(
                    submodel_key=submodel_key, primary_q_mu=primary_mu, primary_q_covar=primary_sigma,
                    primary_k_ud=primary_k_ud, primary_K_uu_inv=primary_K_uu_inv,
                    sub_k_ud=k_ud, sub_K_dd=K_dd, sub_K_uu_inv=submodel_K_uu_inv
                )
            else:
                _sub_mu, _sub_sigma, _cross_sigma, _tot_mu, _tot_sigma, _tot_sigma_chol = variational_model.variational_gp_inference_conditioned_on_inducing_points_function_submodel(
                    submodel_key=submodel_key, primary_q_mu=primary_mu, primary_q_covar=primary_sigma,
                    primary_k_ud=primary_k_ud, primary_K_uu_inv=primary_K_uu_inv,
                    sub_k_ud=k_ud, sub_K_dd=K_dd, sub_K_uu_inv=submodel_K_uu_inv, u_s=conditioned_on_sub_variational_samples
                )
            submodel_mus.append(_sub_mu.detach())
            submodel_sigmas.append(_sub_sigma.detach())
            cross_sigmas.append(_cross_sigma.detach())
            complete_mus.append(_tot_mu)
            complete_sigmas.append(_tot_sigma.detach())
            complete_sigma_chols.append(_tot_sigma_chol)
        

        with torch.no_grad():
            mean_submodel_surface = torch.concat([
                variational_model.reinclude_model_evals(mu.unsqueeze(1), M_batch, N, 1).squeeze(1)
                for mu, M_batch in zip(submodel_mus, M_minis)
            ], dim = 1)    # [Q, M, N]

            std_submodel_surface = torch.concat([
                variational_model.reinclude_model_evals(torch.diagonal(sigma, dim1=-1, dim2=-2).unsqueeze(1), M_batch, N, 1).squeeze(1).sqrt()
                for sigma, M_batch in zip(submodel_sigmas, M_minis)
            ], dim = 1)    # [Q, M, N]

            mean_total_surface = torch.concat([
                variational_model.reinclude_model_evals(mu.unsqueeze(1), M_batch, N, 1).squeeze(1)
                for mu, M_batch in zip(complete_mus, M_minis)
            ], dim = 1)    # [Q, M, N]

            std_total_surface = torch.concat([
                variational_model.reinclude_model_evals(torch.diagonal(sigma, dim1=-1, dim2=-2).unsqueeze(1), M_batch, N, 1).squeeze(1).sqrt()
                for sigma, M_batch in zip(complete_sigmas, M_minis)
            ], dim = 1)    # [Q, M, N]
        
        submodel_f_samples = None

        if take_samples:
            # Get the samples of f_s evaluated at the deltas
            if conditioned_on_sub_variational_samples is None:
                all_f_samples = [
                    variational_model.reparameterised_sample(num_samples = I, mu = mu, sigma_chol = sigma_chol, M = M, N = N)
                    for mu, sigma_chol, M in zip(complete_mus, complete_sigmas, M_minis)
                ]   # Each of shape [Q, I, ~batchsize, N]
            else:
                all_f_samples = [
                    variational_model.reparameterised_sample(num_samples = 1, mu = mu, sigma_chol = sigma_chol, M = M, N = N, unsqueeze_mu=False)
                    for mu, sigma_chol, M in zip(complete_mus, complete_sigmas, M_minis)
                ]   # Each of shape [Q, K, ~batchsize, N]                

            # Shouldn't be any numerical problems after this
            submodel_f_samples = torch.concat(all_f_samples, 2)  # [Q, I or K, M, N]
        
        return {
            "complete_mus": complete_mus, 
            "complete_sigmas": complete_sigmas, 
            "complete_sigma_chols": complete_sigma_chols, 
            "cross_sigmas": cross_sigmas,
            "submodel_mus": submodel_mus, 
            "submodel_sigmas": submodel_sigmas, 

            "submodel_f_samples": submodel_f_samples,

            "submodel_K_uu": submodel_K_uu, 
            "submodel_K_uu_inv": submodel_K_uu_inv, 
            "submodel_K_dds": submodel_K_dds,

            "mean_submodel_surface": mean_submodel_surface,
            "std_submodel_surface": std_submodel_surface,
            "mean_total_surface": mean_total_surface,
            "std_total_surface": std_total_surface,

            "M_minis": M_minis,
        }

    def _inference_from_neural_samples_inner(self, *args, **kwargs) -> _T:
        raise NotImplementedError

    def get_elbo_terms(self, submodel_key: Any, deltas: _T, data: Optional[_T], training_method: str = 'error', max_variational_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}) -> Dict[str, Optional[_T]]:
        
        assert training_method in ['error', 'none'], f"Have not (and will not) implemented {training_method} training methods for hierarchical model!"

        Q, M_all, N, D = deltas.shape
        variational_model = self.get_variational_model(N)

        minibatched_primary_inference_info = self.minibatched_inference_primary(
            deltas, max_variational_batch_size, False
        )
        minibatched_inference_submodel_info = self.minibatched_inference_submodel(
            submodel_key, deltas, minibatched_primary_inference_info.mus, minibatched_primary_inference_info.sigmas,
            minibatched_primary_inference_info.k_uds, minibatched_primary_inference_info.K_uu_inv,
            max_variational_batch_size, True
        )

        if return_kl:

            primary_kl_term = variational_model.kl_loss(                # [Q]
                K_uu = minibatched_primary_inference_info.K_uu, 
                K_uu_inv=minibatched_primary_inference_info.K_uu_inv
            )

            if variational_model.tied_inducing_locations:
                primary_f_samples_at_submodel_inducing_points = variational_model.sample_from_variational_prior(self.num_variational_samples)['samples']

                submodel_kl_term = variational_model.kl_loss_submodel(
                    submodel_key=submodel_key,
                    Ks_uu=minibatched_inference_submodel_info.submodel_K_uu,
                    Ks_uu_inv=minibatched_inference_submodel_info.submodel_K_uu_inv,
                    mu_prior = primary_f_samples_at_submodel_inducing_points
                )
                
            else: 
                raise NotImplementedError("""
                #### XXX This conditioning will likely cause numerical errors as \\tilde{u} and u_s are very similar
                #### May need to force inducing point sharing between all models in the hierarchy!
                primary_f_at_submodel_inducing_points_moments = self.inference(
                    deltas = variational_model.submodels[str(submodel_key)].Z.unsqueeze(2),  # [Q, R, D] -> [Q, R, 1, D]
                    N_override = N
                )
                pfasipm_mu, _, pfasipm_sigma_chol = primary_f_at_submodel_inducing_points_moments
                eps = torch.randn(self.num_models, self.num_variational_samples, pfasipm_mu.shape[-1], dtype = pfasipm_mu.dtype, device = pfasipm_mu.device) # [Q, I, R]
                primary_f_samples_at_submodel_inducing_points = pfasipm_mu + torch.bmm(eps, pfasipm_sigma_chol.transpose(-1, -2))   # [Q, I, R]

                submodel_kl_term = sum([variational_model.kl_loss_submodel(
                    submodel_key=submodel_key,
                    Ks_uu=minibatched_inference_submodel_info.submodel_K_uu,
                    Ks_uu_inv=minibatched_inference_submodel_info.submodel_K_uu_inv,
                    mu_prior = primary_f_sample_at_submodel_inducing_points
                ) for primary_f_sample_at_submodel_inducing_points in primary_f_samples_at_submodel_inducing_points.permute(1, 0, 2)]
                ) / len(primary_f_samples_at_submodel_inducing_points)
                #### XXX
                """)

        else:
            primary_kl_term = torch.ones(Q) * torch.nan # Won't plot!
            submodel_kl_term = torch.ones(Q) * torch.nan # Won't plot!
            primary_f_samples_at_submodel_inducing_points = None

        prior_info = self.generative_model.submodel_generative_models[str(submodel_key)].swap_function.generate_pi_vectors(
            set_size = N, model_evaulations = minibatched_inference_submodel_info.submodel_f_samples
        )
        priors = prior_info['pis']

        # sm_self = self.generative_model.submodel_generative_models[str(submodel_key)].swap_function

        if training_method == 'error':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = self.generative_model.get_marginalised_log_likelihood(
                submodel_key = submodel_key, estimation_deviations = data, pi_vectors = priors,
                kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
            )
        elif training_method == 'none':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = None, None, None

        return {
            'total_log_likelihood': total_log_likelihood, 
            'likelihood_per_datapoint': likelihood_per_datapoint, 
            'posterior': posterior_vectors, 
            'priors': priors, 
            'submodel_f_samples': minibatched_inference_submodel_info.submodel_f_samples,            # [Q, I, M, N]
            'primary_kl_term': primary_kl_term,
            'submodel_kl_term': submodel_kl_term,
            'distance_loss': 0.0,
            "primary_f_samples_at_submodel_inducing_points": primary_f_samples_at_submodel_inducing_points
        }

    def inference(self, submodel_key: Any, deltas: _T, N_override: Optional[int] = None):
        
        primary_inference_info = super().inference(
            deltas, N_override, override_generative_model=self.generative_model.primary_generative_model
        )
        primary_q_mu, primary_q_covar, primary_q_covar_chol = primary_inference_info['mu'], primary_inference_info['sigma'], primary_inference_info['sigma_chol']

        primary_k_ud, primary_K_uu_inv = primary_inference_info['k_ud'], primary_inference_info['K_uu_inv']

        if submodel_key == None:
            return primary_inference_info

        Q, M, N, D = deltas.shape
        if N_override is not None:
            N = N_override
        assert Q == self.num_models
        variational_model = self.get_variational_model(N)

        sub_generative_model = self.get_generative_submodel(submodel_key)
        deduplicated_deltas = variational_model.deduplicate_deltas(deltas, 0)[0][0]

        with torch.no_grad():

            R = variational_model.R
            inducing_locations = variational_model.Z
            sub_K_dd = sub_generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas)
            sub_K_uu = sub_generative_model.swap_function.evaluate_kernel(N, inducing_locations)
            sub_k_ud = sub_generative_model.swap_function.evaluate_kernel(N, inducing_locations, deduplicated_deltas)
            sub_K_uu_inv = torch.linalg.inv(sub_K_uu)
            # K_dd_inv = torch.linalg.inv(K_dd)

            K_uu_inv = torch.linalg.inv(sub_K_uu)
            # assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
            assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

            # Make variational inferences for q(f)
            submodel_mu, submodel_covar, crossterm_covar, total_mu, total_sigma, total_sigma_chol = variational_model.variational_gp_inference_submodel(
                submodel_key=submodel_key, primary_q_mu=primary_q_mu, primary_q_covar=primary_q_covar,
                primary_k_ud=primary_k_ud, primary_K_uu_inv=primary_K_uu_inv,
                sub_k_ud=sub_k_ud, sub_K_dd=sub_K_dd, sub_K_uu_inv=sub_K_uu_inv
            )

        return {
            'primary_q_mu': primary_q_mu, 
            'primary_q_covar': primary_q_covar, 
            'submodel_mu': submodel_mu, 
            'submodel_covar': submodel_covar, 
            'crossterm_covar': crossterm_covar, 
            'total_mu': total_mu, 
            'total_sigma': total_sigma, 
            'total_sigma_chol': total_sigma_chol,
        }

    def inference_on_grid(self, submodel_key: Any, set_size: int, grid_count: int, device: str = 'cuda') -> Dict[str, Optional[_A]]:
        """
        Set up a suitably shaped grid and perform inference over it

        TODO: shapes
        """
        variational_model = self.get_variational_model(set_size)
        D = variational_model.num_features
        if D > 2:
            raise NotImplementedError

        full_grid = torch.linspace(-torch.pi, +torch.pi, grid_count)
        grid_locs = torch.tensor([0.0]) if D == 1 else full_grid.clone()
        grid_cols = full_grid.clone()
        grid_x, grid_y = torch.meshgrid(grid_locs, grid_cols, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y], -1).reshape(len(grid_cols) * len(grid_locs), 2).to(device)
        grid_points = grid_points[...,[1]] if D == 1 else grid_points

        grid_points = grid_points.unsqueeze(0).unsqueeze(2).repeat(self.num_models, 1, 2, 1)
        grid_points[:,:,0,:] = 0.0  # Doesn't change anything but nice to see when debugging

        if not variational_model.fix_non_swap:
            raise NotImplementedError
        
        primary_inference_info = self.inference(submodel_key, grid_points, N_override=set_size)
        if submodel_key is None:
            flat_mu_est, flat_sigma_est, sigma_chol = primary_inference_info['mu'], primary_inference_info['sigma'], primary_inference_info['sigma_chol']
            sub_std_est = None
            flat_submodel_mu_est = None
        else:
            flat_mu_est, flat_sigma_est, sigma_chol = primary_inference_info['total_mu'], primary_inference_info['total_sigma'], primary_inference_info['total_sigma_chol']
            flat_submodel_mu_est = primary_inference_info['submodel_mu']
            flat_submodel_sigma_est = primary_inference_info['submodel_covar']
            sub_std_est = torch.diagonal(flat_submodel_sigma_est, dim1=-1, dim2=-2).sqrt().cpu().numpy()
            flat_submodel_mu_est = flat_submodel_mu_est.cpu().numpy()

        std_est = torch.diagonal(flat_sigma_est, dim1=-1, dim2=-2).sqrt()
        eps = torch.randn(self.generative_model.num_models, 3, flat_mu_est.shape[1], dtype = flat_mu_est.dtype, device = flat_mu_est.device) # [Q, 3, MN (dedup size)]
        grid_f_samples = flat_mu_est.unsqueeze(1) + torch.bmm(eps, sigma_chol.transpose(-1, -2))   # [Q, 3, MN]

        return {
            'one_dimensional_grid': full_grid.cpu().numpy(),            # [Q, grid_points]
            'all_grid_points': grid_points.cpu().numpy(),               # [Q, grid_points, 2, D]
            'mean_surface': flat_mu_est.cpu().numpy(),                  # [Q, grid_points...]
            'std_surface': std_est.cpu().numpy(),                       # [Q, grid_points...]
            'sub_mean_surface': flat_submodel_mu_est,                   # [Q, grid_points...]
            'sub_std_surface': sub_std_est,                             # [Q, grid_points...]
            'function_samples_on_grid': grid_f_samples.cpu().numpy()    # [Q, 3, grid_points...]
        }
    
    def visualise_variational_approximation(
        self, set_size: int, grid_count: int, pi_u_tildes: _A, pi_1_tildes: _A,
        recent_component_priors: Optional[Dict[Any, _A]], 
        true_primary_mean_surface: Optional[_A],
        true_primary_std_surface: Optional[_A],
        true_submodel_mean_surface: Optional[_A],
        true_submodel_std_surface: Optional[_A],
        true_total_mean_surface: Optional[_A],
        true_total_std_surface: Optional[_A],
        min_separation: float, max_separation: float,
        all_deltas: _A, indices_per_submodel: Dict[Any, List[int]],
        deltas_label: str,
        repeat_index: Optional[int] = 0
    ):
        # assert self.num_models == 1, "Currently cannot do HierarchicalWorkingMemoryFullSwapModel.visualise_variational_approximation for >1 model"
        if repeat_index != 0 and self.num_models != 1:
            raise Exception

        variational_model = self.get_variational_model(set_size)
        S = len(variational_model.submodels)
        D = variational_model.num_features
        num_cols = 4 if D == 1 else 6
        num_rows = S + 1            # Including primary model, but not histogram at bottom
        figsize = 8
        fig_surfaces = plt.figure(figsize = (figsize * num_cols, figsize * num_rows))

        assert len(min_separation) == len(max_separation) == len(deltas_label) == D

        if D > 2:
            raise NotImplementedError

        ### Start by plotting primary swap function surface
        primary_inducing_points = variational_model.Z.detach().cpu().squeeze(-1).numpy()        # [Q, R, D]
        primary_inducing_points_means = variational_model.m_u.detach().cpu().numpy()            # [Q, R]
        if variational_model.inducing_point_variational_parameterisation == 'gaussian':
            primary_inducing_points_covars = variational_model.S_uu.detach().cpu().numpy()      # [Q, R, R]

        primary_inference_info = self.inference_on_grid(submodel_key = None, set_size=set_size, grid_count=grid_count)
        primary_one_dimensional_grid = primary_inference_info['one_dimensional_grid']
        primary_all_grid_points = primary_inference_info['all_grid_points']
        primary_mean_surface = primary_inference_info['mean_surface'][0]
        primary_std_surface = primary_inference_info['std_surface'][0]
        primary_function_samples_on_grid = primary_inference_info['function_samples_on_grid']


        primary_lower_surface, primary_upper_surface = primary_mean_surface - 2 * primary_std_surface, primary_mean_surface + 2 * primary_std_surface

        axes_Suu = fig_surfaces.add_subplot(num_rows,num_cols,3)

        if true_primary_mean_surface is not None:
            primary_true_mean = true_primary_mean_surface[0]    # Q = 1
            if true_primary_std_surface is not None:
                primary_true_std = true_primary_std_surface[0]
                primary_true_lower, primary_true_upper = primary_true_mean - 2 * primary_true_std, primary_true_mean + 2 * primary_true_std
            else:
                primary_true_lower, primary_true_upper = float('nan') * primary_true_mean, float('nan') * primary_true_mean

        if true_total_mean_surface is not None:
            total_true_mean = true_total_mean_surface[0]    # Q = 1
            if true_total_std_surface is not None:
                total_true_std = true_total_std_surface[0]
                total_true_lower, total_true_upper = total_true_mean - 2 * total_true_std, total_true_mean + 2 * total_true_std
            else:
                total_true_lower, total_true_upper = float('nan') * total_true_mean, float('nan') * total_true_mean

        if true_submodel_mean_surface is not None:
            submodel_true_mean = true_submodel_mean_surface[0]    # Q = 1
            if true_submodel_std_surface is not None:
                submodel_true_std = true_submodel_std_surface[0]
                submodel_true_lower, submodel_true_upper = submodel_true_mean - 2 * submodel_true_std, submodel_true_mean + 2 * submodel_true_std
            else:
                submodel_true_lower, submodel_true_upper = float('nan') * submodel_true_mean, float('nan') * submodel_true_mean

        if variational_model.inducing_point_variational_parameterisation == 'gaussian':
            axes_Suu.imshow(primary_inducing_points_covars[0], cmap = 'gray')
        
        if D == 1:
            
            axes_surface = fig_surfaces.add_subplot(num_rows,num_cols,4)

            surface_color = axes_surface.plot(primary_one_dimensional_grid, primary_mean_surface, color = 'blue')[0].get_color()
            axes_surface.fill_between(primary_one_dimensional_grid, primary_lower_surface, primary_upper_surface, color = surface_color, alpha = 0.2)

            sample_colour = lighten_color(surface_color, 1.6)
            for sample_on_grid in primary_function_samples_on_grid[0]:
                axes_surface.plot(primary_one_dimensional_grid, sample_on_grid, color = sample_colour, alpha = 0.4)

            axes_surface.scatter(primary_inducing_points[0], primary_inducing_points_means[0], color = 'black', marker = 'o', s = 20)

            if true_primary_mean_surface is not None:
                axes_surface.scatter(all_deltas.flatten(), primary_true_mean.flatten(), color = 'red', alpha = 0.4, s = 5)
                axes_surface.scatter(all_deltas.flatten(), primary_true_lower.flatten(), color = 'red', alpha = 0.01, s = 5)
                axes_surface.scatter(all_deltas.flatten(), primary_true_upper.flatten(), color = 'red', alpha = 0.01, s = 5)

            for sep in [min_separation[0], max_separation[0]]:
                y_bot, y_top = axes_surface.get_ylim()
                axes_surface.plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
                axes_surface.plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
                axes_surface.set_ylim(y_bot, y_top)
                axes_surface.set_xlim(-torch.pi, torch.pi)

            axes_surface.set_xlabel(deltas_label[0])

        elif D == 2:

            axes_surface = fig_surfaces.add_subplot(num_rows,num_cols,4,projection='3d')
            axes_slice_0 = fig_surfaces.add_subplot(num_rows,num_cols,5)
            axes_slice_1 = fig_surfaces.add_subplot(num_rows,num_cols,6)

            axes_surface.set_xlabel('cued: ' + deltas_label[0])
            axes_surface.set_ylabel('estimated: ' + deltas_label[1])
            axes_slice_0.set_xlabel('cued: ' + deltas_label[0])
            axes_slice_1.set_xlabel('estimated: ' + deltas_label[1])

            primary_grid_points = primary_all_grid_points[0,:,1,:]            # [Q, grid_points, 2, D = 2] -> [grid_points, D = 2]
            primary_grid_points_x, primary_grid_points_y = primary_grid_points.reshape(grid_count, grid_count, 2).transpose(2, 0, 1)    # Each [sqrt grid_points, sqrt grid_points]
            structured_primary_mean_surface = primary_mean_surface.reshape(grid_count, grid_count)

            axes_surface.plot_surface(primary_grid_points_x, primary_grid_points_y, structured_primary_mean_surface, color='blue')
            axes_surface.plot_surface(primary_grid_points_x, primary_grid_points_y, primary_lower_surface.reshape(grid_count, grid_count), color='blue', alpha = 0.5)
            axes_surface.plot_surface(primary_grid_points_x, primary_grid_points_y, primary_upper_surface.reshape(grid_count, grid_count), color='blue', alpha = 0.5)

            axes_surface.scatter(*primary_inducing_points[0].T, primary_inducing_points_means[0], color = 'black', marker = 'o', s = 20)


        ### Loop over submodels!
        for s, smk in enumerate(self.submodel_keys):
            
            var_submodel = self.get_variational_submodel(set_size, smk)
            smdl_inducing_points = var_submodel.Z.detach().cpu().squeeze(-1).numpy()        # [Q, R, D]
            smdl_inducing_points_means = var_submodel.m_u.detach().cpu().numpy()            # [Q, R]
            if var_submodel.inducing_point_variational_parameterisation == 'gaussian':
                smdl_inducing_points_covars = var_submodel.S_uu.detach().cpu().numpy()      # [Q, R, R]

            smdl_inference_info = self.inference_on_grid(submodel_key = smk, set_size=set_size, grid_count=grid_count)
            smdl_one_dimensional_grid = smdl_inference_info['one_dimensional_grid']
            smdl_all_grid_points = smdl_inference_info['all_grid_points']
            smdl_total_mean_surface = smdl_inference_info['mean_surface'][0]
            smdl_total_std_surface = smdl_inference_info['std_surface'][0]
            smdl_sub_mean_surface = smdl_inference_info['sub_mean_surface'][0]
            smdl_sub_std_surface = smdl_inference_info['sub_std_surface'][0]
            smdl_function_samples_on_grid = smdl_inference_info['function_samples_on_grid']

            smdl_total_lower_surface, smdl_total_upper_surface = smdl_total_mean_surface - 2 * smdl_total_std_surface, smdl_total_mean_surface + 2 * smdl_total_std_surface
            smdl_sub_lower_surface, smdl_sub_upper_surface = smdl_sub_mean_surface - 2 * smdl_sub_std_surface, smdl_sub_mean_surface + 2 * smdl_sub_std_surface

            # axes1D_exponentiated = fig_surfaces.add_subplot(2,3,2)
            axes_simplex = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+1)
            axes_simplex_no_u = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+2)
            axes_Suu = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+3)

            if var_submodel.inducing_point_variational_parameterisation == 'gaussian':
                axes_Suu.imshow(smdl_inducing_points_covars[0], cmap = 'gray')
            
            if recent_component_priors is not None:
                standard_swap_model_simplex_plots(recent_component_priors[set_size][smk][0], axes_simplex, ax_no_u = axes_simplex_no_u)
                legend_without_repeats(axes_simplex)
                legend_without_repeats(axes_simplex_no_u)

            if D == 1:
                
                axes_surface = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+4)

                surface_color = axes_surface.plot(smdl_one_dimensional_grid, smdl_sub_mean_surface, color = 'blue')[0].get_color()
                axes_surface.fill_between(smdl_one_dimensional_grid, smdl_sub_lower_surface, smdl_sub_upper_surface, color = surface_color, alpha = 0.2)
                axes_surface.scatter(smdl_inducing_points[0], smdl_inducing_points_means[0], color = 'black', marker = 'o', s = 20)

                surface_color = axes_surface.plot(smdl_one_dimensional_grid, smdl_total_mean_surface, color = 'orange')[0].get_color()
                axes_surface.fill_between(smdl_one_dimensional_grid, smdl_total_lower_surface, smdl_total_upper_surface, color = surface_color, alpha = 0.2)
                axes_surface.scatter(smdl_inducing_points[0], primary_inducing_points_means[0] + smdl_inducing_points_means[0], color = 'orange', marker = 'o', s = 20)

                sample_colour = lighten_color(surface_color, 1.6)
                for sample_on_grid in smdl_function_samples_on_grid[0]:
                    axes_surface.plot(smdl_one_dimensional_grid, sample_on_grid, color = sample_colour, alpha = 0.4)

                assert (smdl_inducing_points == primary_inducing_points).all()
                axes_surface.plot([-torch.pi, torch.pi], [pi_u_tildes[s].item(), pi_u_tildes[s].item()], surface_color, linestyle= '-.', linewidth = 3)
                axes_surface.plot([-torch.pi, torch.pi], [pi_1_tildes[s].item(), pi_1_tildes[s].item()], surface_color, linewidth = 3)

                for sep in [min_separation[0], max_separation[0]]:
                    y_bot, y_top = axes_surface.get_ylim()
                    axes_surface.plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes_surface.plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes_surface.set_ylim(y_bot, y_top)
                    axes_surface.set_xlim(-torch.pi, torch.pi)
                
                if true_primary_mean_surface is not None:
                    axes_surface.scatter(all_deltas[indices_per_submodel[smk]].flatten(), submodel_true_mean[[indices_per_submodel[smk]]].flatten(), color = 'red', alpha = 0.4, s = 5)
                    axes_surface.scatter(all_deltas[indices_per_submodel[smk]].flatten(), submodel_true_lower[[indices_per_submodel[smk]]].flatten(), color = 'red', alpha = 0.1, s = 5)
                    axes_surface.scatter(all_deltas[indices_per_submodel[smk]].flatten(), submodel_true_upper[[indices_per_submodel[smk]]].flatten(), color = 'red', alpha = 0.1, s = 5)
                    axes_surface.scatter(all_deltas[indices_per_submodel[smk]].flatten(), total_true_mean[[indices_per_submodel[smk]]].flatten(), color = 'purple', alpha = 0.4, s = 5)
                    axes_surface.scatter(all_deltas[indices_per_submodel[smk]].flatten(), total_true_lower[[indices_per_submodel[smk]]].flatten(), color = 'purple', alpha = 0.1, s = 5)
                    axes_surface.scatter(all_deltas[indices_per_submodel[smk]].flatten(), total_true_upper[[indices_per_submodel[smk]]].flatten(), color = 'purple', alpha = 0.1, s = 5)

                axes_surface.set_xlabel(deltas_label[0])

            elif D == 2:

                axes_surface = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+4,projection='3d')
                axes_slice_0 = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+5)
                axes_slice_1 = fig_surfaces.add_subplot(num_rows,num_cols,(s+1)*num_cols+6)

                axes_surface.set_xlabel('cued: ' + deltas_label[0])
                axes_surface.set_ylabel('estimated: ' + deltas_label[1])
                axes_slice_0.set_xlabel('cued: ' + deltas_label[0])
                axes_slice_1.set_xlabel('estimated: ' + deltas_label[1])

                smdl_grid_points = smdl_all_grid_points[0,:,1,:]            # [Q, grid_points, 2, D = 2] -> [grid_points, D = 2]
                smdl_grid_points_x, smdl_grid_points_y = smdl_grid_points.reshape(grid_count, grid_count, 2).transpose(2, 0, 1)    # Each [sqrt grid_points, sqrt grid_points]
                
                structured_smdl_total_mean_surface = smdl_total_mean_surface.reshape(grid_count, grid_count)
                structured_smdl_sub_mean_surface = smdl_sub_mean_surface.reshape(grid_count, grid_count)

                axes_surface.plot_surface(smdl_grid_points_x, smdl_grid_points_y, structured_smdl_sub_mean_surface, color='blue')
                axes_surface.plot_surface(smdl_grid_points_x, smdl_grid_points_y, smdl_sub_lower_surface.reshape(grid_count, grid_count), color='blue', alpha = 0.5)
                axes_surface.plot_surface(smdl_grid_points_x, smdl_grid_points_y, smdl_sub_upper_surface.reshape(grid_count, grid_count), color='blue', alpha = 0.5)
                axes_surface.scatter(*smdl_inducing_points[0].T, smdl_inducing_points_means[0], color = 'black', marker = 'o', s = 20)

                axes_surface.plot_surface(smdl_grid_points_x, smdl_grid_points_y, structured_smdl_total_mean_surface, color='orange')
                axes_surface.plot_surface(smdl_grid_points_x, smdl_grid_points_y, smdl_total_lower_surface.reshape(grid_count, grid_count), color='orange', alpha = 0.5)
                axes_surface.plot_surface(smdl_grid_points_x, smdl_grid_points_y, smdl_total_upper_surface.reshape(grid_count, grid_count), color='orange', alpha = 0.5)
                axes_surface.scatter(*smdl_inducing_points[0].T, primary_inducing_points_means[0] + smdl_inducing_points_means[0], color = 'orange', marker = 'o', s = 20)

                axes_surface.scatter([0.0], [0.0], [pi_u_tildes[s].item()], color = 'blue', marker='x')
                axes_surface.scatter([0.0], [0.0], [pi_1_tildes[s].item()], color = 'blue', marker='o')

        return fig_surfaces, num_rows, num_cols


    def visualise_pdf_for_example(self, submodel_key: Any, deltas_batch: _T, zeta_targets_batch: _T, theta_count=360, pi_vectors: Optional[_T] = None):
        """
        deltas_batch of shape [Q, 1, N, D]
        zeta_targets_batch of shape [1, N, 1]
        pi_vectors of shape [Q, 1, N+1] if given
        """
        assert zeta_targets_batch.shape[0] == 1
        
        theta_axis = torch.linspace(-torch.pi, torch.pi, theta_count + 1, device=deltas_batch.device)[:-1]
        theta_errors = rectify_angles(theta_axis.unsqueeze(1) - zeta_targets_batch[:,:,0]).unsqueeze(0).repeat(self.num_models, 1, 1)   # [Q, 360, N]
        ss = deltas_batch.shape[-2]

        with torch.no_grad():

            if pi_vectors is None:
                inference_info = self.get_elbo_terms(submodel_key, deltas = deltas_batch, data = None, training_method = 'none', max_variational_batch_size = 0, return_kl = False, kwargs_for_individual_component_likelihoods = {})
                pi_vectors = inference_info['priors']       # [Q, 1, N+1]
            else:
                assert tuple(pi_vectors.shape) == (self.num_models, 1, ss + 1)

            individual_component_likelihoods = self.get_generative_submodel(submodel_key).error_emissions.individual_component_likelihoods_from_estimate_deviations(ss, theta_errors) # [Q, 360, N+1]
            pdf_grid = individual_component_likelihoods * pi_vectors.repeat(1, theta_count, 1)        # [Q, 360, N+1]
            component_sums = pdf_grid.sum(-1)                                                               # [Q, 360] 
        
        return theta_axis, component_sums
        
    def refined_likelihood_estimate(self, *args, **kwargs) -> _T:
        raise NotImplementedError

    def check_Kuu_stability(self, ):
        for N, variational_model in self.variational_models.items():
            try:
                K_uu = self.generative_model.primary_generative_model.swap_function.evaluate_kernel(N, variational_model.Z)
                K_uu_inv = torch.linalg.inv(K_uu)
                K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
                K_uu_chol = torch.linalg.cholesky(K_uu)
            except LinAlgError as e:
                print(f'check_Kuu_stability: {e}')
                return False

            for submodel_key in self.submodel_keys:
                try:
                    sub_K_uu = self.generative_model.submodel_generative_models[str(submodel_key)].swap_function.evaluate_kernel(
                        N, variational_model.Z
                    )
                    sub_K_uu_inv = torch.linalg.inv(sub_K_uu)
                    sub_K_uu_inv_chol = torch.linalg.cholesky(sub_K_uu_inv)
                    sub_K_uu_chol = torch.linalg.cholesky(sub_K_uu)
                except LinAlgError as e:
                    print(f'check_Kuu_stability: {e}')
                    return False
        return True

    def load_primary_state_dict(self, state_dict: Dict[str, _T]):

        assert self.num_models == 1
        
        # Load state dict for primary generative model, excluding error emissions
        generative_model_state_dict = {
            k.removeprefix('generative_model.'): v[[0]] for k, v in state_dict.items() 
            if k.startswith('generative_model.') and not k.removeprefix('generative_model.').startswith('error_emissions')
        }
        assert set(generative_model_state_dict.keys()) == set(self.generative_model.primary_generative_model.state_dict().keys())
        self.generative_model.primary_generative_model.load_state_dict(generative_model_state_dict)

        # Load state dict for primary variational model, excluding error emissions
        variational_model_state_dict = {k.removeprefix('variational_models.'): v[[0]] for k, v in state_dict.items() if k.startswith('variational_models.')}
        assert set(variational_model_state_dict.keys()) - set(self.variational_models.state_dict().keys()) == set()
        self.variational_models.load_state_dict(variational_model_state_dict, strict = False)

        # Load emission parameters for all generative submodels
        error_emissions_dict = {k.removeprefix('generative_model.error_emissions.'): v[[0]] for k, v in state_dict.items() if k.startswith('generative_model.error_emissions.')}
        for gen_submodel in self.generative_model.submodel_generative_models.values():
            assert set(error_emissions_dict.keys()) == set(gen_submodel.error_emissions.state_dict().keys())
            gen_submodel.error_emissions.load_state_dict(error_emissions_dict)

    def load_submodel_state_dict(self, submodel_key: Any, state_dict: Dict[str, _T]):
        
        # Load state dict for generative submodel
        generative_model_state_dict = {k.removeprefix('generative_model.'): v[[0]] for k, v in state_dict.items() if k.startswith('generative_model.')}
        self.get_generative_submodel(submodel_key).load_state_dict(generative_model_state_dict)

        # Load state dict for generative submodel, ignoring inducing points
        for set_size, variational_model in self.variational_models.items():
            set_size_variational_model_state_dict = {k.removeprefix(f'variational_models.{set_size}.'): v[[0]] for k, v in state_dict.items() if k.startswith(f'variational_models.{set_size}')}
            assert variational_model.tied_inducing_locations
            set_size_variational_model_state_dict.pop('inducing_points_tilde')
            self.get_variational_submodel(set_size, submodel_key).load_state_dict(set_size_variational_model_state_dict, strict = False)

        pass



class WorkingMemorySimpleSwapModel(WorkingMemoryFullSwapModel):
    
    def __init__(self, generative_model: NonParametricSwapErrorsGenerativeModel) -> None:
        super(WorkingMemoryFullSwapModel, self).__init__()

        self.num_models = generative_model.num_models
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

    def get_elbo_terms(self, deltas: _T, data: Optional[_T], training_method: str = 'error', max_variational_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}) -> Dict[str, Optional[_T]]:
        if data is not None:
            Q, M, N = data.shape
        else:
            Q, M, N, D = deltas.shape
        return self.get_elbo_terms_easier(data, M, N, training_method, kwargs_for_individual_component_likelihoods)
    
    def visualise_variationald_approximation(self, *args, **kwargs):
        "TODO: implement this, but just for the simplex plots!"
        raise NotImplementedError
    