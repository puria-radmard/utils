from typing import Callable, Union

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor as T

class GaussianProcessPrior(nn.Module, ABC):
    """
        Main class for a Gaussian process prior.
        Overwrite this with:
            a kernel function, k
            any relevant parameters for training
        
        Use this as the class to implement any kernel function/loglikelihood
            approximations in the future!
        
        TODO: this assumes zero mean right now!

        TODO: outputs are all scalar right now!
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def k(self, X1: T, X2: T):
        raise NotImplementedError
    
    @abstractmethod
    def reset_state_dict(self):
        raise NotImplementedError

    def correlation_matrix(self, X1: T, X2: T = None):
        """
            X1 of shape [b1, d]
            X2 of shape [b2, d]
            returns correlation matrix of shape [b1, b2]
            where d = self.input_dim
        """
        if X2 is None:
            X2 = X1
        for feature_array in [X1, X2]:
            assert (feature_array.shape[1] == self.input_dim)
            assert (len(feature_array.shape) == 2)
        return self.k(X1, X2)

    def cov_cholesky(self, X: T):
        "TODO: maybe do a low rank approximation? Functions are wiggly otherwise!"
        try:
            return torch.linalg.cholesky(self.correlation_matrix(X, X))
        except torch._C._LinAlgError:
            return torch.linalg.cholesky(self.correlation_matrix(X, X) + 1e-5 * torch.eye(X.shape[0]))
    
    def sample(self, X: T, return_seed = False, seed = None):
        """
            Samples from prior distribution, given input feature array
        """
        batch = X.shape[0]
        if seed is None:
            seed = torch.randn(batch, device=X.device)
        else:
            assert tuple(seed.shape) == (batch, )
        mean = torch.zeros(batch, device=X.device)
        cov_chol = self.cov_cholesky(X)
        if return_seed:
            return (cov_chol @ seed) + mean, seed
        else:
            return (cov_chol @ seed) + mean


class ARDGaussianProcessPrior(GaussianProcessPrior):
    "SE when input_dim = 1"
    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim)

        self.register_parameter(
            name='log_dimension_lengths', 
            param=torch.nn.Parameter(torch.randn(input_dim))
        )

        self.register_parameter(
            name='log_primary_length', 
            param=torch.nn.Parameter(torch.tensor(0.))
        )

    def reset_state_dict(self):
        self.log_dimension_lengths.data = torch.randn(self.input_dim)
        self.log_dimension_lengths.grad = None
        self.log_primary_length.data = torch.tensor(0.)
        self.log_primary_length.grad = None
    
    def k(self, X1: T, X2: T):
        # iterate over dimsions
        dimension_contributions = []

        for d in range(self.input_dim):
            X1d = X1[:,d]
            X2d = X2[:,d].unsqueeze(1)
            dimension_contributions.append(
                (X1d - X2d).square()
            )
        
        squared_differences = torch.stack(dimension_contributions, -1)
        dimension_scales = self.log_dimension_lengths.exp().square()
        scaled_squared_differences = squared_differences / (2 * dimension_scales)

        log_K_matrix = (2 * self.log_primary_length) - scaled_squared_differences.sum(-1)
            
        return log_K_matrix.exp()   

