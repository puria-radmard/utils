from torch.autograd.functional import jacobian
from torch import nn
from torch import Tensor as _T

from typing import Union
from abc import ABC, abstractmethod


class ExponentialFamilyModelLayerBase(nn.Module, ABC):
    """
        One layer in Vertes and Sahani 2018, section 3.
        Overwrite with required functions for g, S, and theta.
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def sample_conditional_from_previous_latent(self, z_prev: Union[_T, None]):
        """
            This would sample from e^{g(z_prev, params) . S(z_own)} / Z(g),
            but we just use torch.distributions to make this easier to implement
        """
        natural_parameter = self.generate_natural_parameter(z_prev)
        return self.sample_conditional_from_natural_parameter(natural_parameter)

    @abstractmethod
    def sample_conditional_from_natural_parameter(self, natural_parameter: Union[_T, None]):
        "The same as sample_conditional_from_previous_latent when combined with generate_natural_parameter"
        raise NotImplementedError
    
    @abstractmethod
    def generate_natural_parameter_from_raw_parameters(self, raw_params: _T, z_prev: _T):
        """
        Needed so we can use torch autograd to implement nabla_natural_parameter
        """
        raise NotImplementedError

    def generate_natural_parameter(self, z_prev: _T):
        params = self.raw_parameter_values()
        return self.generate_natural_parameter_from_raw_parameters(params, z_prev)

    @abstractmethod
    def generate_sufficient_statistic(self, z: _T):
        "S in the paper"
        raise NotImplementedError
    
    @abstractmethod
    def mean(self, z_prev: Union[None, T]):
        "Return mean sufficient statistic wrt own distribution. If z_prev is provided, it's a conditional mean"
        raise NotImplementedError

    @abstractmethod
    def raw_parameter_values(self):
        "Thetas, not natural parameters"
        raise NotImplementedError

    @abstractmethod
    def replace_raw_parameters(self, new_parameters: _T):
        """
            Will receive a single replacement term tensor, 
                and should reshape/distribute them to parameters accordingly
            Should correspond to raw_parameter_values's output
        """
        raise NotImplementedError

    @abstractmethod
    def mean_sufficient_statistic(self, z_prev: Union[None, T]):
        """
            mu in Vertes and Sahani 2018, see equation 10, if a z_prev is provided
            else a total mean

            NB: not just sufficient statistic of mean!
                Consider zero-centered laplace case, where sufficient statistic is |x|
        """
        raise NotImplementedError

    def nabla_natural_parameter(self, z_prev: Union[_T, None]):
        """
            Derivative wrt parameter of generate_natural_parameter
            Still seperated by batch remember!
        """
        inputs = self.raw_parameter_values()
        func = lambda x: self.generate_natural_parameter_from_raw_parameters(raw_params=x, z_prev=z_prev)
        jac = jacobian(func=func, inputs=inputs)
        return jac



class ExponentialFamilyPriorBase(ExponentialFamilyModelLayerBase):
    """
        Not conditioned on anything, just needs a batch size when sampling
    """
    def sample_conditional_from_previous_latent(self, batch_size: int):
        natural_parameter = self.generate_natural_parameter(batch_size)
        return self.sample_conditional_from_natural_parameter(natural_parameter)

    @abstractmethod
    def generate_natural_parameter_from_raw_parameters(self, raw_params: _T, batch_size: int):
        """
        Needed so we can use torch autograd to implement nabla_natural_parameter
        """
        raise NotImplementedError

    def nabla_natural_parameter(self):
        """
            Derivative wrt parameter of generate_natural_parameter
            Still seperated by batch remember!
        """
        inputs = self.raw_parameter_values()
        func = lambda x: self.generate_natural_parameter_from_raw_parameters(raw_params=x, batch_size=1)
        jac = jacobian(func=func, inputs=inputs)
        return jac

    def generate_natural_parameter(self, batch_size: int):
        params = self.raw_parameter_values()
        return self.generate_natural_parameter_from_raw_parameters(params, batch_size)

    @abstractmethod
    def log_partition_function_gradient(self):
        """
            log_partition_function is phi as in p = exp( ... - phi)
            s.t. in natural domain it = ... / exp(phi)
            This is the gradient of phi wrt the 'root' parameter,
                NOT (nec.) the natural parameter
            Careful with negatives!

            Should only depend on 'root' parameter

            Should have consistent output as adjust_natural_parameters has input!
        """
        raise NotImplementedError
    
    @abstractmethod
    def mean(self):
        "Cannot have conditional mean"
        raise NotImplementedError

    @abstractmethod
    def mean_sufficient_statistic(self):
        "Cannot have conditional mean"
        raise NotImplementedError
