from __future__ import annotations

import torch
from torch import Tensor as T

from typing import List
from abc import ABC, abstractmethod

class ProcessBase(ABC):

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

    def __add__(self, other: ProcessBase) -> ProcessFromFunction:
        return ProcessFromFunction(lambda *a, **k: self(*a, **k) + other(*a, **k))

    def __mult__(self, other: ProcessBase) -> ProcessFromFunction:
        return ProcessFromFunction(lambda *a, **k: self(*a, **k) * other(*a, **k))

    def __matmul__(self, other: ProcessBase) -> ProcessFromFunction:
        return ProcessFromFunction(lambda *a, **k: self(*a, **k) @ other(*a, **k))


class FlatProcess(ProcessBase):

    def __init__(self, value = 0.0):
        self.previous_value = value

    def __call__(self, *args, **kwargs):
        return self.previous_value



class ProcessFromFunction(ProcessBase):

    def __init__(self, function):
        self.function = function
    
    def __call__(self, *args, **kwargs):
        self.previous_value = self.function(*args, **kwargs)
        return self.previous_value


class ProcessGroup:

    def __init__(self, ordered_processes: List[ProcessBase]):
        self.ordered_processes = ordered_processes

    def __call__(self, list_of_args_and_kwargs = None):
        output = []
        if list_of_args_and_kwargs is None:
            list_of_args_and_kwargs = [((), {}) for _ in self.ordered_processes]
        for (args_and_kwargs, process) in zip(list_of_args_and_kwargs, self.ordered_processes):
            args, kwargs = args_and_kwargs
            process(*args, **kwargs)
            output.append(process.previous_value)
        return output

    @property
    def previous_values(self):
        return [process.previous_value for process in self.ordered_processes]
