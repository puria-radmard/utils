import torch
from torch import Tensor as _T
from torch.nn import Module, ModuleDict

from numpy import ndarray as _A

from typing import Dict, Optional, List

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from math import log as mathlog

from purias_utils.util.api import return_as_obj
from purias_utils.util.plotting import standard_swap_model_simplex_plots, legend_without_repeats, lighten_color
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from torch.linalg import LinAlgError

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

    def get_variational_model(self, N) -> NonParametricSwapErrorsVariationalModel:
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
    def minibatched_inference(self, deltas: _T, max_variational_batch_size = 0, take_samples = True):
        Q, M_all, N, D = deltas.shape
        variational_model = self.get_variational_model(N)

        I = self.num_variational_samples
        assert Q == variational_model.num_models
        assert D == variational_model.num_features
        
        R = variational_model.R

        all_deduplicated_deltas, M_minis = variational_model.deduplicate_deltas(deltas, max_variational_batch_size)  # "~M/batch_size length list of entries of shape [Q, ~batch*N, D]"

        # Use kernel all here:
        K_dds = [self.generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]                         # each [Q, ~batch*N, ~batch*N]
        K_uu = self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z)                                                                             # [Q, R, R]
        k_uds = [self.generative_model.swap_function.evaluate_kernel(N, variational_model.Z, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]    # each [Q, R, ~batch*N]

        # Inverse isn't always symmetric!!
        K_uu_inv = torch.linalg.inv(K_uu)
        if not torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all():
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))


        # Make variational inferences for q(f)
        mus, sigmas, sigma_chols = [], [], []
        for k_ud, K_dd in zip(k_uds, K_dds):
            # [Q, ~batch*N], [Q, ~batch*N, ~batch*N], [Q, ~batch*N, ~batch*N]
            _mu, _sigma, _sigma_chol = variational_model.variational_gp_inference(k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv)
            mus.append(_mu)
            sigma_chols.append(_sigma_chol)
            sigmas.append(_sigma.detach())
        
        # For fmse usage
        with torch.no_grad():
            mean_surface = torch.concat([
                variational_model.reinclude_model_evals(mu.unsqueeze(1), M_batch, N, 1).squeeze(1)
                for mu, M_batch in zip(mus, M_minis)
            ], dim = 1)    # [Q, M, N]

            std_surface = torch.concat([
                variational_model.reinclude_model_evals(torch.diagonal(sigma, dim1=-1, dim2=-2).unsqueeze(1), M_batch, N, 1).squeeze(1)
                for sigma, M_batch in zip(sigmas, M_minis)
            ], dim = 1)    # [Q, M, N]

        if take_samples:
            # Get the samples of f evaluated at the deltas
            all_f_samples = [
                variational_model.reparameterised_sample(num_samples = I, mu = mu, sigma_chol = sigma_chol, M = M, N = N)
                for mu, sigma_chol, M in zip(mus, sigma_chols, M_minis)
            ]   # Each of shape [Q, I, ~batchsize, N]

            # Shouldn't be any numerical problems after this
            f_samples = torch.concat(all_f_samples, 2)  # [Q, I, M, N]

        else:
            f_samples = None

        return {
            "mus": mus, "sigmas": sigmas, "sigma_chols": sigma_chols, "f_samples": f_samples,
            "K_uu": K_uu, "K_uu_inv": K_uu_inv, "K_dds": K_dds, "M_minis": M_minis,
            "mean_surface": mean_surface, "std_surface": std_surface
        }


    def get_elbo_terms(self, deltas: _T, data: Optional[_T], training_method: str = 'error', max_variational_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}) -> Dict[str, Optional[_T]]:

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
                estimation_deviations = data, pi_vectors = priors,
                kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
            )
        elif training_method == 'beta':
            raise NotImplementedError
            llh_term = self.generative_model.get_component_log_likelihood(
                selected_components = data, pi_vectors = pi_vectors
            )
            posterior, unaggregated_lh = None, None
        elif training_method == 'none':
            total_log_likelihood, likelihood_per_datapoint, posterior_vectors = None, None, None

        distance_loss = minibatched_inference_info.K_uu.tril(-1).max() / minibatched_inference_info.K_uu.max()

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

    def inference(self, deltas: _T, N_override: Optional[int] = None):
        
        Q, M, N, D = deltas.shape
        if N_override is not None:
            N = N_override
        assert Q == self.num_models
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
            # assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
            K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
            K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
            assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

            # Make variational inferences for q(f)
            mu, sigma, sigma_chol = variational_model.variational_gp_inference(
                k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
            )

        return mu, sigma, sigma_chol

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
            

    def inference_on_grid(self, set_size: int, grid_count: int, device: str = 'cuda') -> Dict[str, _A]:
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
        flat_mu_est, flat_sigma_est, sigma_chol = self.inference(grid_points, N_override=set_size)
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


    def visualise_variational_approximation(
        self, set_size: int, grid_count: int, pi_u_tildes: _A, pi_1_tildes: _A,
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

        variational_model = self.get_variational_model(set_size)
        Q = self.num_models
        D = variational_model.num_features
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

            individual_component_likelihoods = self.generative_model.error_emissions.individual_component_likelihoods_from_estimate_deviations(ss, theta_errors) # [Q, 360, N+1]
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

            # f^k ~ variational posterior
            # We don't get_elbo_terms becausewe don't want to do self.generative_model.get_marginalised_log_likelihood yet - we do aggregation after the log etc, not before
            variational_inference_terms = self.minibatched_inference(all_deltas, max_variational_batch_size=max_variational_batch_size, take_samples = True)
            variational_generated_component_prior_info = self.generative_model.swap_function.generate_pi_vectors(set_size = N, model_evaulations = variational_inference_terms.f_samples, mc_average = False) 
            variational_generated_component_priors: _T = variational_generated_component_prior_info['pis']                                                  # [Q, K, Mtotal, N+1]

            # log p(y | Z, f^i, phi)
            # because we ask for the datasets to be 'combined' we will:
                # a) automatically calculate this llh estimate on all the data (training data or not)
                # b) account for two terms in the derivation here
            individual_component_downstream_likelihoods: _T = self.generative_model.error_emissions.individual_component_likelihoods_from_estimate_deviations(
                set_size = N, estimation_deviations = all_errors
            )                                                                                                                                                       # [Q, Mtotal, N+1] - p(y[m] | beta[n], Z[m])
            joint_component_and_error_likelihood_variational = individual_component_downstream_likelihoods.unsqueeze(1) * variational_generated_component_priors    # [Q, K, Mtot, N+1] - p(y[m] | beta[n], Z[m]) * p(beta[n]| f[k], Z[m]) = p(y[m], beta[n] | f[k], Z[m])
            variational_conditioned_log_likelihood_per_datapoint = joint_component_and_error_likelihood_variational.sum(-1).log()                                   # [Q, K, Mtot] - log p(y[m], | f[k], Z[m])

            # Finally (for this term...), construct the first term of the estimate, which is all an MC estimate
            training_indices = training_indices.unsqueeze(1).repeat(1, K, 1).to(variational_generated_component_priors.device)
            variational_log_likelihood_on_train_set: _T = variational_conditioned_log_likelihood_per_datapoint.gather(-1, training_indices)                 # [Q, K, Mtot] -> [Q, K, M_train]
            assert tuple(variational_log_likelihood_on_train_set.shape) == (Q, K, M_train)
            variational_log_likelihood_on_train_set_total = variational_log_likelihood_on_train_set.sum(-1, keepdim=True).mean(1)                                   # [Q, 1]
            adjusted_variational_conditioned_log_likelihood_per_datapoint = (variational_conditioned_log_likelihood_per_datapoint.mean(1) + variational_log_likelihood_on_train_set_total)      # [Q, Mtot]


            #### Second term - the "partition function" of size [Q, 1]

            # g^j ~ GP prior
            zero_mus = [torch.zeros(K_dd.shape[0], K_dd.shape[1], dtype = K_dd.dtype, device = K_dd.device) for K_dd in variational_inference_terms.K_dds]
            prior_f_samples_batched = [
                variational_model.reparameterised_sample(num_samples = K, mu = zmu, sigma_chol = torch.linalg.cholesky(K_dd), M = M, N = N)
                for zmu, K_dd, M in zip(zero_mus, variational_inference_terms.K_dds, variational_inference_terms.M_minis)
            ]                                                                                                                                               # Each of shape [Q, K, ~batchsize, N]
            prior_f_samples = torch.concat(prior_f_samples_batched, 2)                                                                                      # [Q, K, Mtot, N]
            prior_conditioned_prior_info = self.generative_model.swap_function.generate_pi_vectors(                                                         # [Q, K, Mtot, N+1]
                set_size = N, model_evaulations = prior_f_samples, mc_average = False
            )
            prior_conditioned_component_priors = prior_conditioned_prior_info['pis']

            # Finally, aggregate to the full model log-evidence under the prior
            joint_component_and_error_likelihood_prior = individual_component_downstream_likelihoods.unsqueeze(1) * prior_conditioned_component_priors      # [Q, K, Mtot, N+1] - p(y[m] | beta[n], Z[m]) * p(beta[n]| f[k], Z[m]) = p(y[m], beta[n] | f[k], Z[m])
            prior_conditioned_likelihood_per_datapoint: _T = joint_component_and_error_likelihood_prior.sum(-1)                                             # [Q, K, Mtot]
            parition_function = prior_conditioned_likelihood_per_datapoint.mean(1).log().sum(1, keepdim=True)                                               # [Q, K, Mtot] -> [Q, Mtot] -> [Q, 1]

        return {
            "parition_function": parition_function,
            "adjusted_variational_conditioned_log_likelihood_per_datapoint": adjusted_variational_conditioned_log_likelihood_per_datapoint,
            "importance_sampled_log_likelihoods": adjusted_variational_conditioned_log_likelihood_per_datapoint - parition_function
        }
        






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
    
    def visualise_variational_approximation(self, *args, **kwargs):
        "TODO: implement this, but just for the simplex plots!"
        raise NotImplementedError
