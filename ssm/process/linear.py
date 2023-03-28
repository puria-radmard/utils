import torch
from torch import Tensor as T
from purias_utils.ssm.process.base import ProcessBase


class LinearStateSpaceModel(ProcessBase):
    """
        Input process is B@u, rather than just u itself
    """
    def __init__(
        self, 
        dynamics_matrix_process: ProcessBase,
        noise_process: ProcessBase,
        input_process: ProcessBase,
        ) -> None:
        self.dynamics_matrix_process = dynamics_matrix_process
        self.input_process = input_process
        self.noise_process = noise_process
        
    def __call__(self, *args, **kwargs):
        self.previous_value = (
            self.dynamics_matrix_process.previous_value @ 
            self.previous_value
        ) + self.input_process.previous_value + self.noise_process.previous_value
        return self.previous_value






