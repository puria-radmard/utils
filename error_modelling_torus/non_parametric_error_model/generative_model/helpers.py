import torch
from torch import nn

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles



class HolderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, idx):
        "In case set size is accessed here, see logic below, e.g. NonParametricSwapErrorsGenerativeModel.inverse_ells!"
        assert isinstance(idx, str)
        return self


class KernelParameterHolder(HolderBase):
    def __init__(self, num_features, trainable_kernel_delta):
        
        super().__init__()

        self.num_features = num_features

        log_scaler_raw = (1.0 + torch.relu(2.0 + (0.6 * torch.randn([])))).log().to(torch.float64)
        log_inverse_ells_raw = (1.0 + torch.relu(5.0 + (1.5 * torch.randn([])))).log().to(torch.float64)
        
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
        return self.log_inverse_ells.exp().reshape(1, 1, self.num_features)

    @property
    def scaler(self):
        return self.log_scaler.exp()

    @property
    def kernel_noise_sigma(self):
        return self.log_kernel_noise_sigma.exp() + self.minimal_kernel_sigma



class UniformHalfWidthHolder(HolderBase):
    def __init__(self):
        super().__init__()
        halfwidth_unscaled_raw = (0.25 * torch.randn(1)).to(torch.float64)
        self.register_parameter('halfwidth_unscaled', nn.Parameter(halfwidth_unscaled_raw, requires_grad = True))

    @property
    def halfwidth(self):
        return torch.pi * self.halfwidth_unscaled.sigmoid()


class ConcentrationParameterHolder(HolderBase):
    def __init__(self):
        super().__init__()
        log_concentration_raw = (10 + (0.3 * torch.randn(1)).exp()).log().to(torch.float64)
        self.register_parameter('log_concentration', nn.Parameter(log_concentration_raw, requires_grad = True))

    @property
    def concentration(self):
        return self.log_concentration.exp()

class StableAlphaHolder(HolderBase):
    def __init__(self):
        super().__init__()

        init_alpha_upper = torch.tensor(1.99)
        init_alpha_raw_upper = torch.arctanh(init_alpha_upper - 1.0).item()
        init_alpha_lower = torch.tensor(0.90)
        init_alpha_raw_lower = torch.arctanh(init_alpha_lower - 1.0).item()

        alpha_raw = (init_alpha_raw_lower + (init_alpha_raw_upper-init_alpha_raw_lower) * torch.rand(1)).to(torch.float64)
        self.register_parameter('alpha_raw', nn.Parameter(alpha_raw, requires_grad = True))

    @property
    def alpha(self):
        return self.alpha_raw.tanh() + 1



class StableGammaHolder(HolderBase):
    def __init__(self):
        super().__init__()
        gamma_raw = (0.1 + 0.5 * torch.randn(1)).to(torch.float64)
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
    def __init__(self, mean_init) -> None:
        super().__init__()
        pi_tilde_raw = ((0.20 * torch.randn(1)) + mean_init).to(torch.float64)
        self.register_parameter('pi_tilde', nn.Parameter(pi_tilde_raw, requires_grad = True))
    
    def logit_vector(self):
        return self.pi_tilde
