import torch
from torch import Tensor as T

from torch.nn.functional import softplus

from typing import Union

from exponential_family_models.base import ExponentialFamilyModelLayerBase


class ZeroCenteredLinearSpreadLaplaceModelLayer(ExponentialFamilyModelLayerBase):
    """
        p(x | y) = Laplace(x | 0, SoftPlus(B * y))
            where B is self.spread_multiplier

        XXX: For now, everything is independent along the axes! 
            This makes the generate_sufficient_statistic and mean_sufficient_statistic calculation easy!
    """

    def __init__(self, output_dim: int):
        super().__init__(output_dim=output_dim)
        self.register_parameter(name='spread_multiplier', param=torch.nn.Parameter(torch.randn(output_dim)))

    def generate_natural_parameter_from_raw_parameters(self, raw_params, z_prev):
        """
        -1/scale, independent dims
        """
        mult = raw_params.unsqueeze(0).repeat(z_prev.shape[0], 1)
        return -1 / softplus(mult * z_prev)

    def generate_sufficient_statistic(self, z: T):
        return z.abs()
    
    def mean(self, z_prev: Union[None, T]):
        "Regardless of spread, also its marginal is also zero I'm pretty sure"
        return torch.zeros(z_prev.shape[0], self.output_dim)
    
    def raw_parameter_values(self):
        return self.spread_multiplier

    def replace_raw_parameters(self, new_parameters: T):
        assert tuple(new_parameters.shape) == (self.output_dim,)
        self.spread_multiplier.data = new_parameters.data
    
    def mean_sufficient_statistic(self, z_prev: Union[None, T]):
        """
            S(z) = |z|, which is distributed Exponential(1 / scale)
            The mean of this is scale
            Independent dims allows (mult * ...) rather than (mult @ ...)
        """
        mult = self.spread_multiplier.unsqueeze(0).repeat(z_prev.shape[0], 1)
        if len(z_prev.shape) == 1:
            z_prev = z_prev.unsqueeze(1)
        return softplus(mult * z_prev)
        
    def sample_conditional_from_natural_parameter(self, natural_parameter: Union[T, None]):
        "Sample from zero centered Laplace"
        scale = -1 / natural_parameter
        shape = natural_parameter.shape
        finfo = torch.finfo(scale.dtype)
        dtype = natural_parameter.dtype
        device = natural_parameter.device
        u = torch.rand(shape, dtype=dtype, device=device) * 2 - 1
        return - scale * u.sign() * torch.log1p(-u.abs().clamp(min=finfo.tiny))
