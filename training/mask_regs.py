from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as T

from maths import communicability_reg


import itertools

class RegBase:
    """
    Regaularisers that aggregate some param * mask.
    """
    def __init__(self, mask_func: callable) -> None: 
        self.mask_func = mask_func
        super(RegBase, self).__init__()

    @property
    def mask(self) -> T: 
        return self.mask_func()

    # Basic arithmetic
    def __add__(self, other: RegBase):
        new_reg = RegBase()

        @property
        def new_mask(new_self):
            return self.mask + other.mask

        new_reg.mask = new_mask

    def __mul__(self, other: RegBase):
        def new_mask():
            return self.mask * other.mask
        new_reg = RegBase(new_mask)
        return new_reg



class L1Reg(RegBase):

    def __init__(self, param: T) -> None:
        self.param = param

    @property
    def mask(self):
        return torch.sign(self.param)



class DistanceReg(RegBase):

    def __init__(self, locations: T):
        self.locations = locations.float()
        _locs = self.locations[None, :, :]
        self.distances = torch.cdist(_locs, _locs)[0]

    @property
    def mask(self):
        # Assume no changes
        return self.distances.abs()


class GridDistanceReg(DistanceReg):

    def __init__(self, counts: T, lengths: T):
        dim_locations = []
        for count, length in zip(counts, lengths):
            dim_locations.append(torch.linspace(0., length, count))
        locations = torch.tensor(list(itertools.product(*dim_locations)))
        super(GridDistanceReg, self).__init__(locations = locations)


class CommunicabilityReg(RegBase):
    """
    This uses Croft and Higham, 2009, as in seRNN manuscript
    """

    def __init__(self, param: T) -> None:
        self.param = param
        self.device = param.device

    @property
    def mask(self):
        detached_param = self.param.detach()
        return communicability_reg(detached_param).abs()
