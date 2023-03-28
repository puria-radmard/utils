import torch
from purias_utils.noise.base import NoiseProcess


class OrnsteinUhlenbeckProcess(NoiseProcess):

    def __init__(self, N_matrix, magnitude, eps1, eps2, device):
        self.N_matrix = N_matrix.to(device)
        self.magnitude = magnitude
        self.eps1 = eps1
        self.eps2 = eps2
        self.device = device

    def __call__(self, *args, **kwargs):
        new_rand = (self.N_matrix @ torch.randn_like(self.previous_value).to(self.device))
        self.previous_value = ((self.previous_value * self.eps1) + (new_rand * self.eps2))
        return self.magnitude * self.previous_value
