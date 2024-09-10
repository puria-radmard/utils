import math

import torch
from torch import nn
from torch import Tensor as _T

from typing import Optional

import numpy as np

from torch.distributions import Dirichlet, VonMises, Cauchy, Uniform
from torch.distributions.von_mises import _log_modified_bessel_fn

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.helpers import *

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


VALID_EMISSION_TYPES = ['von_mises', 'wrapped_cauchy', 'wrapped_stable', 'uniform']


class ErrorsEmissionsBase(nn.Module):

    def __init__(self, emissions_set_sizes: list) -> None:
        super().__init__()

        self.emissions_set_sizes = emissions_set_sizes

    def generate_samples(self, comp_means, set_size):
        raise NotImplementedError

    @staticmethod
    def fill_in_uniform_samples_and_begin_sampling(selected_components: _T, vm_means: _T):
        sample_set = torch.zeros_like(selected_components).double()
        unif_sampl = 2 * torch.pi * torch.rand(sample_set[selected_components == 0].shape) - torch.pi
        sample_set[selected_components == 0] = unif_sampl.double().to(selected_components.device)
        reshaped_vm = vm_means.unsqueeze(0).repeat(selected_components.shape[0], 1, 1)
        return sample_set, reshaped_vm

    def sample_from_components(self, set_size: int, selected_components: _T, vm_means: _T):
        """
        selected_components of shape [I, M]
        vm_means of shape [M, N] around circle - **NB name is a misnomer, as the emission probability might not be a von Mises**
        """
        sample_set, reshaped_vm = self.fill_in_uniform_samples_and_begin_sampling(selected_components, vm_means)

        with torch.no_grad():
            for n in range(selected_components.max().item()):
                n_sel = (selected_components == (n+1))
                comp_means = reshaped_vm[n_sel][:,n]
                if comp_means.numel() > 0:
                    assert sample_set[n_sel].unique().item() == 0.0
                    samples = self.generate_samples(comp_means, set_size)
                    sample_set[n_sel] = samples
        return sample_set       # [I, M]

    def evaluate_emissions_pdf_on_circle(self, set_size: int, num_test_points = 360, device = 'cuda', **llh_kwargs):
        theta_axis = torch.linspace(-torch.pi, torch.pi, num_test_points+1)[:-1].to(device)
        with torch.no_grad():
            likelihood = self.individual_component_likelihoods_from_estimate_deviations(
                set_size=set_size, estimation_deviations=theta_axis.unsqueeze(-1), **llh_kwargs
            )[0,:,1]
        return theta_axis.cpu().numpy(), likelihood.cpu().numpy()

    def individual_component_likelihoods_from_estimate_deviations(self, set_size: int, estimation_deviations: _T) -> _T:
        raise NotImplementedError



class ParametricErrorsEmissionsBase(ErrorsEmissionsBase):

    def emission_parameter(self, set_size):
        raise NotImplementedError

    def individual_component_likelihoods_from_estimate_deviations_inner(self, set_size: int, estimation_deviations: _T):
        raise NotImplementedError

    def individual_component_likelihoods_from_estimate_deviations(self, set_size: int, estimation_deviations: _T):
        """
        estimation_deviations ([M, N])  = rectify(estimates - zeta_c)
        
        output is of size [1, M, N+1], where output[:,0] is likelihood of the uniform component (always 1/2pi)
            Hanging 0th dimenion is there for self.get_marginalised_log_likelihood downstream...

        However, estimation_deviations can come in any shape [..., N] 
            - this is used for example in NonParametricModelDrivenMultipleOrientationDelayedSingleEstimationTask where there is a batch and a trial axis at the front
        """
        emission_component_probs = self.individual_component_likelihoods_from_estimate_deviations_inner(set_size, estimation_deviations)
        Mdims = estimation_deviations.shape[:-1]
        uniform_component_probs = torch.ones(*Mdims, 1).to(estimation_deviations.device) / (2 * torch.pi)
        output = torch.concat([uniform_component_probs, emission_component_probs], -1)
        return output.unsqueeze(0)  # [1, (M...), N + 1]



class VonMisesParametricErrorsEmissions(ParametricErrorsEmissionsBase):

    def __init__(self, emissions_set_sizes: list) -> None:
        super().__init__(emissions_set_sizes)

        self.concentration_holder = (
            ConcentrationParameterHolder() if emissions_set_sizes is None
            else nn.ModuleDict({str(N): ConcentrationParameterHolder() for N in emissions_set_sizes})
        )
    
    def emission_parameter(self, set_size):
        return self.concentration_holder[str(set_size)].concentration

    def generate_samples(self, comp_means, set_size):
        return VonMises(loc = comp_means, concentration = self.emission_parameter(set_size)).sample()

    def individual_component_likelihoods_from_estimate_deviations_inner(self, set_size: int, estimation_deviations: _T):
        von_mises_concentration = self.emission_parameter(set_size)
        log_prob = (von_mises_concentration * torch.cos(estimation_deviations)) - math.log(2 * math.pi) - _log_modified_bessel_fn(von_mises_concentration, order=0)     # Always zero mean
        return log_prob.exp()



class WrappedStableParametricErrorsEmissions(ParametricErrorsEmissionsBase):

    p_cut_off = 100

    def __init__(self, emissions_set_sizes: list) -> None:
        super().__init__(emissions_set_sizes)

        self.alpha_stability_holder = (
            StableAlphaHolder() if emissions_set_sizes is None
            else nn.ModuleDict({str(N): StableAlphaHolder() for N in emissions_set_sizes})
        )

        self.gamma_scale_holder = (
            StableGammaHolder() if emissions_set_sizes is None
            else nn.ModuleDict({str(N): StableGammaHolder() for N in emissions_set_sizes})
        )

    def emission_parameter(self, set_size):
        return torch.concat([self.alpha_stability_holder[str(set_size)].alpha, self.gamma_scale_holder[str(set_size)].gamma])

    def generate_samples(self, comp_means, set_size, alpha: Optional[float] = None, gamma: Optional[float] = None):
        "Method taken from Wikipedia!"

        if alpha is None:
            alpha = self.alpha_stability_holder[str(set_size)].alpha
        if gamma is None:
            gamma = self.gamma_scale_holder[str(set_size)].gamma

        sample_shape = comp_means.shape

        u_samples = (torch.rand(sample_shape).to(comp_means.device) * torch.pi) - (torch.pi / 2)
        w_samples = torch.ones(sample_shape).to(comp_means.device)
        w_samples.exponential_(lambd=1).to(comp_means.device)

        sin_term = (alpha * u_samples).sin() / torch.pow(u_samples.cos(), 1. / alpha)
        exp_term = torch.pow((u_samples * (1. - alpha)).cos() / w_samples, ((1. - alpha) / alpha))
        x_samples = sin_term * exp_term

        return rectify_angles((gamma * x_samples) + comp_means)

    def individual_component_likelihoods_from_estimate_deviations_inner(self, set_size: int, estimation_deviations: _T, alpha: Optional[float] = None, gamma: Optional[float] = None):
        "Method taken from Arthur Pewsey, 2008"
        
        if alpha is None:
            alpha = self.alpha_stability_holder[str(set_size)].alpha
        if gamma is None:
            gamma = self.gamma_scale_holder[str(set_size)].gamma

        gamma_to_the_alpha = torch.pow(gamma, alpha)
        
        result = torch.ones_like(estimation_deviations).to(estimation_deviations.device) / (2 * torch.pi)
        rho_p = torch.ones_like(gamma_to_the_alpha).to(estimation_deviations.device) # rho_0

        for p in range(1, self.p_cut_off + 1):
            p_minus_1_tensor = torch.tensor(p - 1.0).to(estimation_deviations.device)
            log_r_p_minus_1 = gamma_to_the_alpha * (torch.pow(p_minus_1_tensor, alpha) - torch.pow(p_minus_1_tensor + 1.0, alpha))
            rho_p = rho_p * log_r_p_minus_1.exp()
            pth_term = (rho_p * (p * estimation_deviations).cos()) / torch.pi
            result = result + pth_term
        
        return result


    

class UniformParametricErrorsEmissions(ParametricErrorsEmissionsBase):

    def __init__(self, emissions_set_sizes: list) -> None:
        super().__init__(emissions_set_sizes)

        self.uniform_halfwidth_holder = (
            UniformHalfWidthHolder() if emissions_set_sizes is None
            else nn.ModuleDict({str(N): UniformHalfWidthHolder() for N in emissions_set_sizes})
        )
    
    def emission_parameter(self, set_size):
        return self.uniform_halfwidth_holder[str(set_size)].halfwidth
    
    def generate_samples(self, comp_means, set_size):
        halfwidth = self.emission_parameter(set_size)
        assert 0 <= halfwidth <= torch.pi
        return Uniform(low = comp_means - halfwidth, high = comp_means + halfwidth).sample()

    def individual_component_likelihoods_from_estimate_deviations_inner(self, set_size: int, estimation_deviations: _T):
        in_bump = estimation_deviations.abs() > self.emission_parameter(set_size)
        import pdb; pdb.set_trace(header = 'finish this! neginf part might cause problems')
        return None



class SmoothedWeightedDeltasErrorsEmissions(ErrorsEmissionsBase):

    """
    delta_smoother_kappa = concentration of the von Mises distribution with which deltas are approximated
    """

    def __init__(self, emissions_set_sizes: list, delta_smoother_kappa: float, initial_distribution_kappa: float) -> None:
        super().__init__(emissions_set_sizes)

        self.delta_smoother_kappa = delta_smoother_kappa
        
        assert len(emissions_set_sizes) == 1, "SmoothedWeightedDeltasErrorsEmissions update in training.py not implemented for multiple set sizes!"

        self.delta_train_holder = (
            DeltaTrainHolder(initial_distribution_kappa) if emissions_set_sizes is None
            else nn.ModuleDict({str(N): DeltaTrainHolder(initial_distribution_kappa) for N in emissions_set_sizes})
        )

    def load_new_distribution(self, set_size: int, locations: _T, weights: _T):
        self.delta_train_holder[str(set_size)].load_delta_locations(locations)
        self.delta_train_holder[str(set_size)].load_delta_weights(weights)

    def get_current_distribution(self, set_size: int):
        return (
            self.delta_train_holder[str(set_size)].delta_locations,
            self.delta_train_holder[str(set_size)].delta_weights
        )

    def generate_residuals_samples(self, shape, set_size):
        "Hierarchical sampling - first sample locations then sample from tight von mises from that"
        weights = self.delta_train_holder[str(set_size)].delta_weights
        locations = self.delta_train_holder[str(set_size)].delta_locations
        selected_locations_idx = torch.tensor(np.random.choice(len(weights), tuple(shape), p=weights.cpu().numpy()))
        selected_locations = locations[selected_locations_idx]
        samples = VonMises(selected_locations, self.delta_smoother_kappa).sample()
        
        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.hist(locations.cpu().numpy(), 50, weights = weights.cpu().numpy(), alpha = 0.5, density = True)
        # plt.hist(samples.cpu().numpy(), 50, alpha = 0.5, density = True)
        # plt.savefig('samples.png')

        return samples
    
    def generate_samples(self, comp_means, set_size):
        return rectify_angles(comp_means + self.generate_residuals_samples(comp_means.shape, set_size))

    def individual_component_likelihoods_from_estimate_deviations_inner(self, set_size: int, estimation_deviations: _T, locations: _T = None, weights: _T = None):
        if weights is None:
            weights = self.delta_train_holder[str(set_size)].delta_weights
        if locations is None:
            locations = self.delta_train_holder[str(set_size)].delta_locations
        von_mises = VonMises(locations.unsqueeze(-1).unsqueeze(-1), self.delta_smoother_kappa)
        emissions_prob = (weights.unsqueeze(-1).unsqueeze(-1) * von_mises.log_prob(estimation_deviations).exp()).sum(0)
        Mdims = estimation_deviations.shape[:-1]
        uniform_component = torch.ones(*Mdims, 1).to(estimation_deviations.device) / (2 * torch.pi)
        output = torch.concat([uniform_component, emissions_prob], -1)
        return output.unsqueeze(0)  # [1, (M...), N + 1]
