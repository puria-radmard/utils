import torch
from torch import Tensor as T
from purias_utils.ssm.process.base import ProcessBase, FlatProcess

from typing import Union

class LinearStateSpaceModel(ProcessBase):
    """
        This is an AR(1) process with an input
        Input process is B@u, rather than just u itself
    """
    def __init__(
        self, 
        dynamics_matrix_process: ProcessBase,
        noise_process: ProcessBase,
        input_process: ProcessBase,
        value_offset_process: Union[ProcessBase, None],
        ) -> None:

        if value_offset_process is None:
            value_offset_process = FlatProcess(value = 0.0)

        self.dynamics_matrix_process = dynamics_matrix_process
        self.input_process = input_process
        self.noise_process = noise_process
        self.value_offset_process = value_offset_process

        self.add_parent(dynamics_matrix_process)
        self.add_parent(input_process)
        self.add_parent(noise_process)
        self.add_parent(value_offset_process)
        
    def __call__(self, *args, **kwargs):
        value_term = (self.previous_value - self.value_offset_process.previous_value)
        ar_term = self.dynamics_matrix_process.previous_value @ value_term
        input_term = self.input_process.previous_value
        noise_term = self.noise_process.previous_value
        self.previous_value = ar_term + input_term + noise_term
        return self.previous_value






