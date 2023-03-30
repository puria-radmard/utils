import torch
from torch import nn
from torch import Tensor as T

from torch import sigmoid

from exponential_family_models.base import ExponentialFamilyPriorBase, ExponentialFamilyModelLayerBase

class LinearBernoulliPriorModelLayer(ExponentialFamilyPriorBase):
    """
        This is the prior over latents in the Helmholtz Mode described in Dayan and Abbott, chapter 10

        We however use the natural parameter of the Bernoulli, which is logit(p), where p is the heads probability
    """
    def __init__(self, output_dim: int):
        super().__init__(output_dim)

        self.register_parameter(name='natural_prior_param', param=torch.nn.Parameter(torch.randn(output_dim)))

    def sample_conditional_from_natural_parameter(self, natural_parameter):
        assert natural_parameter.shape[-1] == self.output_dim
        return torch.bernoulli(sigmoid(natural_parameter))

    def generate_natural_parameter_from_raw_parameters(self, raw_params, batch_size):
        return raw_params.repeat(batch_size, 1)
    
    def generate_sufficient_statistic(self, z: T):
        "No difference!"
        return z

    def mean(self):
        return sigmoid(self.natural_prior_param)
    
    def mean_sufficient_statistic(self):
        "No difference to mean!"
        return sigmoid(self.natural_prior_param)
    
    def log_partition_function_gradient(self):
        "Only one parameter, so can just output that"
        exp = self.natural_prior_param.exp()
        return exp / (1 + exp)
    
    def raw_parameter_values(self):
        "Raw parameters are natural parameters!"
        return self.natural_prior_param

    def replace_raw_parameters(self, new_parameters: T):
        "Only get one vector in, so not hard to redistribute!"
        self.natural_prior_param.data = new_parameters.data


class LinearBernoulliModelLayer(ExponentialFamilyModelLayerBase):
    """
        TODO: come back to this and see how it will work in new framework!

        This is the generative model G in the Helmholtz Mode described in Dayan and Abbott, chapter 10
        It is also the recongitional model W that approximates the inverse

        We however use the natural parameter of the Bernoulli, which is logit(p), where p is the heads probability
    """
    def __init__(self, prev_dim: int, output_dim: int, use_bias: bool = True):

        super().__init__(output_dim)

        self.conditional_parameter_generator = nn.Linear(prev_dim, output_dim, bias = use_bias)
        self.prev_dim = prev_dim

    def sample_conditional_from_natural_parameter(self, natural_parameter):
        assert natural_parameter.shape[-1] == self.output_dim
        return torch.bernoulli(sigmoid(natural_parameter))

    def generate_natural_parameter_from_raw_parameters(self, raw_params, z_prev):
        raise NotImplementedError
        assert z_prev.shape[-1] == self.prev_dim
        return self.conditional_parameter_generator(z_prev)
