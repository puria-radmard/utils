from __future__ import annotations

import torch
from torch import nn

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from typing import Union


def reduce_to_single_model(module: Union[HolderBase, nn.ModuleDict], model_index: int = 0) -> None:
    """
    Downsteam models have helpers registered either as single modules, or module dicts.
    e.g.:

    self.concentration_holder = (
        ConcentrationParameterHolder(num_models) if emissions_set_sizes is None
        else nn.ModuleDict({str(N): ConcentrationParameterHolder(num_models) for N in emissions_set_sizes})
    )

    This module takes in either type and results after running self.reduce_to_single_model appropriately
    """
    if isinstance(module, HolderBase):
        module.reduce_to_single_model(model_index)
    else:
        for v in module.values():
            v: HolderBase
            v.reduce_to_single_model(model_index)


class HolderBase(nn.Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.num_models = num_models

    def __getitem__(self, idx):
        "In case set size is accessed here, see logic below, e.g. NonParametricSwapErrorsGenerativeModel.inverse_ells!"
        assert isinstance(idx, str)
        return self
    
    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.num_models = 1
        for name, param in self.named_parameters():
            self.register_parameter(name, nn.Parameter(param[[model_index]], requires_grad = True))


class KernelParameterHolder(HolderBase):
    def __init__(self, num_models, num_features, trainable_kernel_delta):

        super().__init__(num_models)

        self.num_features = num_features

        log_scaler_raw = (1.0 + torch.relu(2.0 + (0.6 * torch.randn([num_models])))).log().to(torch.float64)
        log_inverse_ells_raw = (1.0 + torch.relu(5.0 + (1.5 * torch.randn([num_models, num_features])))).log().to(torch.float64)
        
        self.register_parameter('log_inverse_ells', nn.Parameter(log_inverse_ells_raw, requires_grad = True))
        self.register_parameter('log_scaler', nn.Parameter(log_scaler_raw, requires_grad = True))
        
        if trainable_kernel_delta:
            log_kernel_noise_sigma_raw = (torch.tensor(0.00001)).log().to(torch.float64)
            self.register_parameter('log_kernel_noise_sigma', nn.Parameter(log_kernel_noise_sigma_raw, requires_grad = True))
            self.minimal_kernel_sigma = 0.00001
        else:
            self.log_kernel_noise_sigma = (torch.tensor(0.0001)).log().to(torch.float64)
            # self.log_kernel_noise_sigma = -(torch.tensor(float('inf'))).log().to(torch.float64)
            self.minimal_kernel_sigma = 0.0

    @property
    def inverse_ells(self):
        return self.log_inverse_ells.exp().reshape(self.num_models, 1, 1, self.num_features)

    @property
    def scaler(self):
        return self.log_scaler.exp().reshape(self.num_models, 1, 1)

    @property
    def kernel_noise_sigma(self):
        return self.log_kernel_noise_sigma.exp() + self.minimal_kernel_sigma



class UniformHalfWidthHolder(HolderBase):
    def __init__(self, num_models: int):
        super().__init__(num_models)
        halfwidth_unscaled_raw = (0.25 * torch.randn(num_models)).to(torch.float64)
        self.register_parameter('halfwidth_unscaled', nn.Parameter(halfwidth_unscaled_raw, requires_grad = True))

    @property
    def halfwidth(self):
        return torch.pi * self.halfwidth_unscaled.sigmoid()


class ConcentrationParameterHolder(HolderBase):
    def __init__(self, num_models: int):
        super().__init__(num_models)
        log_concentration_raw = (10 + (0.3 * torch.randn(num_models)).exp()).log().to(torch.float64)
        self.register_parameter('log_concentration', nn.Parameter(log_concentration_raw, requires_grad = True))

    @property
    def concentration(self):
        return self.log_concentration.exp()


class DoubleConcentrationParameterHolder(HolderBase):
    """
    This is a bit of a weird case where we want two concentrations but want one to be always not greater than the other
    """
    def __init__(self, num_models: int):
        super().__init__(num_models)
        log_larger_concentration_raw = (8 + (0.3 * torch.randn(num_models)).exp()).log().to(torch.float64)
        log_smaller_concentration_raw = (12 + (0.3 * torch.randn(num_models)).exp()).log().to(torch.float64)
        # smaller_concentration_ratio_raw = (0.2 * torch.randn(num_models)).to(torch.float64)
        self.register_parameter('log_larger_concentration', nn.Parameter(log_larger_concentration_raw, requires_grad = True))
        self.register_parameter('log_smaller_concentration', nn.Parameter(log_smaller_concentration_raw, requires_grad = True))
        # self.register_parameter('smaller_concentration_ratio_raw', nn.Parameter(smaller_concentration_ratio_raw, requires_grad = True))

    @property
    def concentrations(self):
        larger_concentration = self.log_larger_concentration.exp()
        smaller_concentration = self.log_smaller_concentration.exp()
        # smaller_concentration_as_ratio = self.smaller_concentration_ratio_raw.sigmoid()
        # smaller_concentration = smaller_concentration_as_ratio * larger_concentration
        return larger_concentration, smaller_concentration


class StableAlphaHolder(HolderBase):
    def __init__(self, num_models: int):
        super().__init__(num_models)

        init_alpha_upper = torch.tensor(1.99)
        init_alpha_raw_upper = torch.arctanh(init_alpha_upper - 1.0).item()
        init_alpha_lower = torch.tensor(0.90)
        init_alpha_raw_lower = torch.arctanh(init_alpha_lower - 1.0).item()

        alpha_raw = (init_alpha_raw_lower + (init_alpha_raw_upper-init_alpha_raw_lower) * torch.rand(num_models)).to(torch.float64)
        self.register_parameter('alpha_raw', nn.Parameter(alpha_raw, requires_grad = True))

    @property
    def alpha(self):
        return self.alpha_raw.tanh() + 1.



class StableGammaHolder(HolderBase):
    def __init__(self, num_models: int):
        super().__init__(num_models)
        gamma_raw = (0.5 * torch.randn(num_models)).to(torch.float64)
        self.register_parameter('gamma_raw', nn.Parameter(gamma_raw, requires_grad = True))

    @property
    def gamma(self):
        return self.gamma_raw.exp()


class DeltaTrainHolder(HolderBase):
    """
    Should have no state dict parameters in current form, since we just load in the values of the weights and locations
    from the inference calculations with the generative model
    """
    def __init__(self, initial_distribution_kappa: float, num_initial_locations: int = 500, device = 'cuda') -> None:
        raise Exception('Needs updating!')
        super().__init__()
        self.delta_locations = (torch.rand(num_initial_locations) * 2 * (torch.pi - 1e-4) - torch.pi).to(device)
        weights = torch.randn_like(self.delta_locations).abs() * (self.delta_locations.cos() * initial_distribution_kappa).exp()
        self.delta_weights = (weights / weights.sum()).to(device)

    def load_delta_locations(self, new_locs):
        assert len(new_locs.shape) == 1
        self.delta_locations = rectify_angles(new_locs)

    def load_delta_weights(self, new_weights):
        assert len(new_weights.shape) == 1
        comp = torch.tensor(1.)
        assert torch.isclose(new_weights.sum().to(comp.dtype), comp)
        self.delta_weights = rectify_angles(new_weights)


class PiTildeHolder(HolderBase):
    def __init__(self, mean_init: float, num_models: int) -> None:
        super().__init__(num_models)
        pi_tilde_raw = ((0.20 * torch.randn(num_models)) + mean_init).to(torch.float64)
        self.register_parameter('pi_tilde', nn.Parameter(pi_tilde_raw, requires_grad = True))
