import torch
from torch import nn
from torch import Tensor as T

from typing import List, Callable

from purias_utils.exponential_family_models.base import *

class MixtureOfGaussiansPriorModelLayer(ExponentialFamilyPriorBase):
    """
        Only learns the Gaussian statistics, not the mixing proportions

        COME BACK TO THIS WHEN YOU'RE READY
    """

    def __init__(self, output_dim: int, mixture_priors: List[float]):
        super().__init__(output_dim)

        self.mixture_priors = torch.tensor(mixture_priors)
        self.cdf = torch.cumsum(self.mixture_priors)
        self.K = len(mixture_priors)

        # TODO: when putting this on mirzakhani, replace this with a ModelLayer object!
        self.register_parameter(name='cholesky_sigma', param=torch.nn.Parameter(torch.randn(self.K, output_dim, output_dim)))
        self.register_parameter(name='mu', param=torch.nn.Parameter(torch.randn(self.K, output_dim)))
        
    def sample_conditional_from_natural_parameter(self, natural_parameter):
        "Very inefficient way of doing this..."
        raise NotImplementedError

        normal_rand = torch.randn_like(natural_parameter)                   # batch x dim
        cholesky_sigmas = [torch.tril(cs) for cs in self.cholesky_sigma]    # K x [dim x dim]

        comp_rands = torch.rand(natural_parameter.shape[0])         # batch
        comp_rands = comp_rands.unsqueeze(-1).repeat(self.K)        # batch, K
        selected_components = (comp_rands < self.cdf).argmax(-1)    # batch (ints)

        output = torch.zeros_like(natural_parameter)                # batch x dim
        for b in range(natural_parameter.shape[0]):                 # TODO: group this up by k!
            k = selected_components[b]
            output[b] = normal_rand[b] @ cholesky_sigmas[k] + self.mu[k]

        return output




class MixtureOfScalarGaussiansModelLayer(ExponentialFamilyModelLayerBase):
    """
        This instead takes in a z_prev, transforms it, and uses it as its mean.
        Standard deviation is taken as a raw parameter.

        If no transform is given, just use a {0,1} -> {-1,1} times mu transformer

        TODO: maybe generalise this to multivariate?
    """
    def __init__(self, conditional_transform: Union[Callable[[T, T], T], None] = None):
        super().__init__(output_dim=1)

        self.register_parameter(name='mu', param=torch.nn.Parameter(torch.randn([]).abs()))
        self.register_parameter(name='sigma', param=torch.nn.Parameter(torch.randn([]).abs()))

        if conditional_transform is not None:
            self.rf = conditional_transform
        else:
            self.rf = lambda bz, m: (2 * bz - 1) * m

    def generate_natural_parameter_from_raw_parameters(self, raw_params: T, z_prev: T):
        "Scalar so output shape is [batch, 2]"
        mu = raw_params[0]
        sigma = raw_params[1]
        mean = self.rf(z_prev, mu)
        minus_half = torch.ones_like(mean) * -0.5
        return torch.cat([mean, minus_half], dim = 1) / (sigma ** 2)

    def generate_sufficient_statistic(self, z: T):
        return torch.cat([z, z**2], axis = -1)
    
    def sample_conditional_from_natural_parameter(self, natural_parameter: T):
        min_half_over_sigma_squared = natural_parameter[:,1].unsqueeze(1)
        sigma = (-0.5 / min_half_over_sigma_squared) ** 0.5
        mean = natural_parameter[:,0].unsqueeze(1) * (sigma ** 2)
        return mean + (torch.randn(natural_parameter.shape[0], 1) * sigma)
    
    def mean(self, z_prev: Union[None, T]):
        "Conditional mean is easy, marginal mean I'm not sure!"
        if z_prev is None:
            raise NotImplementedError
        else:
            return self.rf(z_prev, self.mu)

    def mean_sufficient_statistic(self, z_prev: Union[None, T]):
        "Same comment as above! Expected value, then expected squared value"
        if z_prev is None:
            raise NotImplementedError
        else:
            conditional_mean = self.mean(z_prev)
            conditional_power = conditional_mean**2 + self.sigma**2
            return torch.hstack([conditional_mean, conditional_power])
    
    def raw_parameter_values(self):
        "Return as a 2-tensor"
        return torch.stack([self.mu, self.sigma])

    def replace_raw_parameters(self, new_parameters: T):
        "Follow the above!"
        assert tuple(new_parameters.shape) == (2,)
        self.mu.data = new_parameters.data[0]
        self.sigma.data = new_parameters.data[1]


def batch_diag(bm: T):
    assert (len(bm.shape) == 3) and bm.shape[1] == bm.shape[2]
    return torch.stack([torch.diag(m) for m in bm], dim=0)



class DiagonalConditionalGaussianModelLayer(ExponentialFamilyModelLayerBase):
    """
        Covariance is learned as a log, kept strictly diagonal
        Mean is some parameter matrix mean_multiplier @ z_prev (vector)
    """

    def __init__(self, output_dim: int, input_dim: int):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.register_parameter(name='log_diagonal_sigma', param=torch.nn.Parameter(torch.randn(output_dim)))
        self.register_parameter(name='mean_multiplier', param=torch.nn.Parameter(torch.randn(input_dim, output_dim)))

    def generate_natural_parameter_from_raw_parameters(self, raw_params: T, z_prev: T, flatten = True):
        N_out, N_in = self.output_dim, self.input_dim
        mean_multiplier = raw_params[:N_out * N_in].reshape(N_in, N_out)
        diagonal_sigma = raw_params[-N_out:]
        inverse_cov = torch.diag(1 / diagonal_sigma.exp())
        mean = z_prev @ mean_multiplier
        first_natural_parameter = (mean @ inverse_cov).unsqueeze(1)
        second_natural_parameter = -0.5*inverse_cov.unsqueeze(0).repeat(first_natural_parameter.shape[0], 1, 1)
        result = torch.cat([first_natural_parameter, second_natural_parameter], dim = 1)
        return result.reshape(z_prev.shape[0], -1) if flatten else result
    
    def generate_sufficient_statistic(self, z: T, flatten = True):
        z_squared = z.unsqueeze(1) * z.unsqueeze(2)
        result = torch.cat([z.unsqueeze(1), z_squared], dim = 1)
        return result.reshape(z.shape[0], -1) if flatten else result

    def sample_conditional_from_natural_parameter(self, natural_parameter: Union[T, None]):
        "This will only work for diagonal inverse cov provided!! No asserts in place here!"
        if len(natural_parameter.shape) == 3:   # generated with flatten = False
            assert natural_parameter.shape[1:] == (self.output_dim + 1, self.output_dim)
        elif len(natural_parameter.shape) == 2: # generated with flatten = True
            natural_parameter = natural_parameter.reshape(-1, self.output_dim + 1, self.output_dim)
        minus_over_half_inverse_cov = natural_parameter[:,1:]
        cov_matrix = torch.diag_embed(- 0.5 / batch_diag(minus_over_half_inverse_cov))
        mean_vector = torch.einsum('bnn, bn -> bn', cov_matrix, natural_parameter[:,0])
        cov_cholesky = cov_matrix**0.5
        rand_seed = torch.randn(natural_parameter.shape[0], self.output_dim)
        result = mean_vector + torch.einsum('bnn, bn -> bn', cov_cholesky, rand_seed)
        return result

    def mean(self, z_prev: Union[None, T]):
        "Conditional is easy, but marginal I'm not sure!"
        if z_prev is None:
            raise NotImplementedError
        else:
            if self.input_dim > 1:
                assert z_prev.shape[-1] == self.input_dim
            else:
                assert len(z_prev.shape) == 1
                z_prev = z_prev.reshape(-1, 1)
            return z_prev @ self.mean_multiplier

    def mean_sufficient_statistic(self, z_prev: Union[None, T], flatten = True):
        "Same comment as above! Expected value, then expected squared value"
        if z_prev is None:
            raise NotImplementedError
        else:
            conditional_mean = self.mean(z_prev)
            mu_outer = conditional_mean.unsqueeze(2) @ conditional_mean.unsqueeze(1)
            conditional_power = torch.diag(self.log_diagonal_sigma.exp()) + mu_outer
            result = torch.cat([conditional_mean.unsqueeze(1), conditional_power], dim = 1)
            return result.reshape(z_prev.shape[0], -1) if flatten else result
    
    def raw_parameter_values(self):
        """
            This one is a bit tricky.
            Have to return the mean multiplier (N_out x N_int) and the covariance diagonal (N_out)
                as one vector.
        """
        return torch.cat([
            self.mean_multiplier.reshape(-1),
            self.log_diagonal_sigma
        ], dim = 0)
    
    def replace_raw_parameters(self, new_parameters: T):
        N_out, N_in = self.output_dim, self.input_dim
        assert tuple(new_parameters.shape) == (N_out * N_in + N_out,)
        self.mean_multiplier.data = new_parameters[:N_out * N_in].data.reshape(N_in, N_out)
        self.log_diagonal_sigma.data = new_parameters[-N_out:].data



