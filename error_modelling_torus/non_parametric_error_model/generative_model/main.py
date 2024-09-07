
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

    def get_marginalised_log_likelihood(self, estimation_deviations: _T, pi_vectors: _T, return_posterior: bool):
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
        individual_component_log_likelihoods = self.error_emissions.individual_component_log_likelihoods_from_estimate_deviations(
            set_size, estimation_deviations
        )
        pdf_grid = individual_component_log_likelihoods.exp() * pi_vectors  # [I, M, N]
        total_log_likelihood = pdf_grid.sum(-1).log().sum(-1).mean(0)                         # [I, M, N] -> [I, M] -> [I, M] -> [I] -> scalar

        if return_posterior:
            with torch.no_grad():
                posterior_vectors = pdf_grid * pi_vectors           
                posterior_vectors = posterior_vectors / posterior_vectors.sum(-1, keepdim = True)   # [I, M, N]
                posterior_vectors = posterior_vectors.mean(0)

        else:
            posterior_vectors = None

        return total_log_likelihood, posterior_vectors, pdf_grid

    def get_component_likelihood(self, selected_components: _T, pi_vectors: _T):
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
        return selected_pis.log().sum(-1).mean(0)

    def get_categorical_likelihood(self, real_pi_vectors: _T, exp_f_evals: _T):
        """
        This is the first term of the ELBO when training on the pis (i.e. synthetic data only)

        Dirichlet loglikelihood

        real_pi_vectors [M, N] - i.e. from (synthetic) generative model
        exp_f_evals of shape [I, M, N+1] -
            exp_f_evals[:,:,0] are e^pi_u_tildes
            exp_f_evals[:,:,n>0] are e^f evaluations at each point
        """
        dirichlets = Dirichlet(concentration = exp_f_evals)
        log_probs = dirichlets.log_prob(real_pi_vectors)
        return log_probs.sum(-1).mean(0)

    def full_data_generation(self, set_size: int, vm_means: _T, **kwargs_for_swap_function):
        """
        vm_means is basically zeta_col
        vm_means: [M, N]
        """
        with torch.no_grad():
            pi_vectors, exp_grid = self.swap_function.generate_pi_vectors(set_size=set_size, return_exp_grid=True, **kwargs_for_swap_function)
            betas = self.swap_function.sample_betas(pi_vectors)
            samples = self.error_emissions.sample_from_components(set_size, betas, vm_means)
        return {'exp_grid': exp_grid, 'pi_vectors': pi_vectors, 'betas': betas, 'samples': samples}

    def inference(self, set_size: int, estimation_deviations, **kwargs_for_swap_function):
        """
        Full generate posteriors from priors and component values

        estimation_deviations ([M, N])  = rectify(estimates - zeta_c)
        """
        with torch.no_grad():
            pi_vectors = self.swap_function.generate_pi_vectors(set_size=set_size, return_exp_grid=False, **kwargs_for_swap_function)
            post_hoc_lhs = self.error_emissions.individual_component_log_likelihoods_from_estimate_deviations(set_size, estimation_deviations).exp()
            posterior_vector = pi_vectors * post_hoc_lhs
            posterior_vector = posterior_vector / posterior_vector.sum(-1, keepdim = True)
        return {
            'prior': pi_vectors,
            'likelihood': post_hoc_lhs,
            'posterior': posterior_vector
        }
