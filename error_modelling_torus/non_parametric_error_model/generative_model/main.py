
import torch
from torch import nn
from torch import Tensor as _T
from torch.distributions import Dirichlet

from typing import Optional, Dict, List


from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.emissions import ErrorsEmissionsBase
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.swap_function import SwapFunctionBase


class NonParametricSwapErrorsGenerativeModel(nn.Module):

    def __init__(self, swap_function: SwapFunctionBase, error_emissions: ErrorsEmissionsBase) -> None:
        super().__init__()

        self.swap_function = swap_function
        self.error_emissions = error_emissions

        if (self.swap_function.function_set_sizes is not None) and (self.error_emissions.emissions_set_sizes is not None):
            assert self.swap_function.function_set_sizes == self.error_emissions.emissions_set_sizes

    def get_marginalised_log_likelihood(self, estimation_deviations: _T, pi_vectors: _T):
        """
        This is the first term of the ELBO when training on the estimates

        estimation_deviations ([M, N])  = rectify(estimates - zeta_c)
        pi_vectors of shape [I, M, N+1]

        Generate a grid of shape [1, M, N+1] giving the individual components of the llh term inside the log
        Exp these and multiply these with the I pi vector samples
        Then sum over the last dimension for both, giving a size of [I, M]

        Then log again and and sum over the M dimension, 
            and finally mean over the I dimension
        """
        set_size = estimation_deviations.shape[1]
        individual_component_likelihoods = self.error_emissions.individual_component_likelihoods_from_estimate_deviations(
            set_size, estimation_deviations
        )   # [1, M, N+1] - p(y[m] | beta[n], Z[m])

        pdf_grid = individual_component_likelihoods * pi_vectors                  # [I, M, N+1] - p(y[m] | beta[n], Z[m]) * p(beta[n]| Z[m], f[i])
        total_log_likelihood = pdf_grid.sum(-1).mean(0).log().sum(0)                        # [I, M, N+1] -> [I, M] -> [M] -> [M] -> scalar

        posterior_vectors = pdf_grid.sum(0) # [M, N+1]
        posterior_vectors = posterior_vectors / posterior_vectors.sum(-1, keepdim = True)    # [M, N+1]

        return total_log_likelihood, posterior_vectors, pdf_grid

    def get_component_log_likelihood(self, selected_components: _T, pi_vectors: _T):
        """
        This is the first term of the ELBO when training on the selected components/betas (i.e. synthetic data only)

        Very simple loglikelihood here

        selected_components [M] = betas from (synthetic) generative model
        pi_vectors of shape [I, M, N+1], i.e. still generated as before by variational model
        """
        I, M, N_plus_1 = pi_vectors.shape
        selected_pis = torch.zeros(I, M).to(pi_vectors.device).to(pi_vectors.dtype)
        for m in range(M):
            b = selected_components[m]
            selected_pis[:,m] = pi_vectors[:,m,b]
        return selected_pis.mean(0).log().sum()

    def full_data_generation(self, set_size: int, vm_means: _T, kwargs_for_generate_pi_vectors: dict = {}, kwargs_for_sample_betas: dict = {}, kwargs_for_sample_from_components: dict = {}):
        """
        vm_means is basically zeta_recalled: [M, N]
        """
        with torch.no_grad():
            pi_vectors, exp_grid = self.swap_function.generate_pi_vectors(set_size=set_size, return_exp_grid=True, **kwargs_for_generate_pi_vectors)
            betas = self.swap_function.sample_betas(pi_vectors, **kwargs_for_sample_betas)
            samples = self.error_emissions.sample_from_components(set_size, betas, vm_means, **kwargs_for_sample_from_components)
        return {'exp_grid': exp_grid, 'pi_vectors': pi_vectors, 'betas': betas, 'samples': samples}

    def empirical_residual_distribution_weights(self, posterior_vectors: _T, estimation_deviations: _T):
        """
        posterior_vectors: [M, N+1] - p(beta[n] | y[m], Z[m])
        estimation_deviations: [M, N] - rectify(estimates - zeta_c)
        """
        particle_weights_non_uniform = posterior_vectors[:,1:].detach() # posterior_vectors [M, N]
        set_size = estimation_deviations.shape[-1]
        with torch.no_grad():
            error_lhs: _T = self.error_emissions.individual_component_likelihoods_from_estimate_deviations_inner(set_size, estimation_deviations)
            particle_weights_uniform = (posterior_vectors[:,[0]] * error_lhs)   # [M, N]
        return {
            "particle_weights_non_uniform": particle_weights_non_uniform.detach(),
            "particle_weights_uniform": particle_weights_uniform.detach(),
            "particle_weights_total": set_size * (particle_weights_non_uniform + particle_weights_uniform).detach(),
        }


