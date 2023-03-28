import math
import numpy as np
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import _standard_gamma
from torch.distributions.constraints import Constraint
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

from scipy.special import beta as beta_function # sigh

def sample_from_categorical(pmf):
    cdf = np.cumsum(pmf)
    return (np.random.uniform() < cdf).argmax()

def generate_epsilon_greedy_policy(q_values, epsilon):
    num_a = len(q_values)
    policy_pmf = np.ones(num_a) * epsilon / num_a
    policy_pmf[q_values.argmax()] += 1 - epsilon
    return policy_pmf



class _Half2DPositive(Constraint):
    """
    Strictly 2d vector, with first entry real and second entry positive
    """
    def __init__(self):
        super().__init__()

    def check(self, value):
        return (
            value.shape[-1] == 2 and
            value[0] == value[0] and
            0 < value[1]
        )

    def __repr__(self):
        return "funky support constraint for normal-gamma distribution"


class ExtendedDirichlet(Dirichlet):

    def predictive_entropy(self):
        # TODO: check shape for batching here!
        # TODO: notation in pdf is weird here!
        mean_vector: torch.Tensor = self.mean
        return - (mean_vector * mean_vector.log()).sum(0)

    def average_entropy(self):
        # TODO: check shape for batching here!
        conc_sum = self.concentration.sum(0)
        return (
            torch.digamma(conc_sum + 1) - 
            (self.concentration * torch.digamma(self.concentration + 1)).sum() / conc_sum
        )

    def mean_array(self, array: torch.Tensor):
        # TODO: check shape for batching here!
        return (self.concentration * array).sum(0) / self.concentration.sum()

    def entropy(self):
        raise Exception('Dont use ExtendedDirichlet until maths sorted out')


class NormalGamma(ExponentialFamily):
    r"""
    TODO: do an example!
    """
    arg_constraints = {
        'normal_loc': constraints.real, 
        'normal_precision': constraints.positive,
        'gamma_concentration': constraints.positive, 
        'gamma_rate': constraints.positive
    }
    support = _Half2DPositive
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, normal_loc, normal_precision, gamma_concentration, gamma_rate, validate_args=None):
        stats = [normal_loc, normal_precision, gamma_concentration, gamma_rate]
        self.normal_loc, self.normal_precision, self.gamma_concentration, self.gamma_rate = broadcast_all(*stats)
        if all([isinstance(stat, Number) for stat in stats]):
            batch_shape = torch.Size()
        else:
            batch_shape = self.normal_loc.size()
        super(NormalGamma, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NormalGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.normal_loc = self.normal_loc.expand(batch_shape)
        new.normal_precision = self.normal_precision.expand(batch_shape)
        new.gamma_concentration = self.gamma_concentration.expand(batch_shape)
        new.gamma_rate = self.gamma_rate.expand(batch_shape)
        super(NormalGamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        """
        Sequentially sample from gamma then normal. Return mean and precision of sampled gaussians as a tuple
        """
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            _prec = _standard_gamma(self.gamma_concentration.expand(shape)) / self.gamma_rate.expand(shape)
            _mean = torch.normal(
                mean = self.normal_loc.expand(shape),
                std = (self.normal_precision * _prec) ** -2 # TODO: check this! should it be squared??
            )
            return (_prec, _mean)
        
    def predictive_entropy(self):
        # TODO: this can 100% be parallelised across different NormalGamma instances!

        k = self.gamma_concentration
        nu = self.gamma_rate
        beta = self.normal_precision

        return 0.5 * (
            torch.log((2 * nu * beta_function(k, 0.5)) / (beta + 1))
            + (2 * k + 1) * torch.digamma(0.5*(2 * k + 1) + torch.digamma(k))
        )

    def average_entropy(self):
        # TODO: double check this one! Also typo in the pdf 
        return 0.5 * (
            torch.digamma(self.gamma_concentration) -
            torch.log(2 * math.pi * math.e * self.gamma_rate)
        )
        
