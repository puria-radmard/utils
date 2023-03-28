import torch
from torch import Tensor as T
from abc import ABC, abstractmethod

class NoiseProcess(ABC):

    previous_value: T

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reinitialise(self, new_value: T):
        self.previous_value = new_value

    def generate(self, num_steps, *args, **kwargs):
        return torch.stack([self() for _ in range(num_steps)])


class NoiselessProcess(NoiseProcess):

    def __init__(self):
        self.previous_value = 0.0

    def __call__(self, *args, **kwargs):
        return 0.0

    def reinitialise(self, new_value: T):
        self.previous_value = new_value

