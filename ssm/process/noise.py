import torch
from torch import Tensor as _T

from purias_utils.ssm.process.base import ProcessBase

from purias_utils.maths.matmul import multiply_2d_state_safely


class OrnsteinUhlenbeckProcess(ProcessBase):

    def __init__(self, N_matrix, output_magnitude, eps1, eps2, device):
        self.N_matrix = N_matrix.to(device)
        self.output_magnitude = output_magnitude
        self.eps1 = eps1
        self.eps2 = eps2
        self.device = device

    def __call__(self, *args, **kwargs):
        new_rand = torch.randn_like(self.previous_value).to(self.device) @ self.N_matrix 
        self.previous_value = (
            (self.previous_value * self.eps1) + (new_rand * self.eps2)
        )
        return self.output_magnitude * self.previous_value 

    def to(self, device):
        self.device = device
        self.N_matrix = self.N_matrix.to(device)
        return self


class MatrixStateOrnsteinUhlenbeckProcess(OrnsteinUhlenbeckProcess):

    def __call__(self, *args, **kwargs):
        new_rand = multiply_2d_state_safely(lambda x: self.N_matrix @ x, torch.randn_like(self.previous_value).to(self.device))
        self.previous_value = (
            (self.previous_value * self.eps1) + (new_rand * self.eps2)
        )
        return self.output_magnitude * self.previous_value


class WhiteNoise(ProcessBase):

    def __init__(self, N_matrix, output_magnitude, device):
        self.N_matrix: _T = N_matrix.to(device)
        self.output_magnitude = output_magnitude
        self.device = device
    
    @property
    def covariance(self):
        return self.N_matrix @ self.N_matrix.T

    def __call__(self, *args, **kwargs):
        new_rand = (self.N_matrix @ torch.randn_like(self.previous_value).to(self.device))
        self.previous_value = new_rand
        return self.output_magnitude * self.previous_value 

    def to(self, device):
        self.device = device
        self.N_matrix = self.N_matrix.to(device)
        return self
