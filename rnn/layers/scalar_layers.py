import torch
from torch import nn
from torch import Tensor as _T
from typing import Type, Union


class ScalarLayer(nn.Module):

    def __init__(self, initial_value = 1.0, bias = True, bias_initial_value = 0.0) -> None:
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.tensor(initial_value), requires_grad=True))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.tensor(bias_initial_value), requires_grad=True))
        else:
            self.bias = torch.tensor(0.0)

    def forward(self, input: _T) -> _T:
        return self.weight * input + self.bias


class FixedScalarLayer(nn.Module):

    def __init__(self, value) -> None:
        super().__init__()
        self.weight = value

    def forward(self, input: _T) -> _T:
        return self.weight * input
