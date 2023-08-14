from typing import Optional

import torch
from torch import Tensor as T
from torch.nn import Module, CrossEntropyLoss


class WeightedSequenceCEL(CrossEntropyLoss):

    def __init__(self, weight: Optional[T] = None) -> None:
        super().__init__(weight, reduction='none')

    def forward(self, input: T, target: T, seq_weight: T) -> T:
        input = input.permute(0, 2, 1)
        # [batch, time]
        unweighted = super(WeightedSequenceCEL, self).forward(input=input, target=target)
        return (seq_weight * unweighted).mean()


class AngleLoss(Module):

    def __init__(self, fixation: bool = True, magnitude_regulariser_weight: float = 0.01) -> None:
        super().__init__()
        self.fixation = fixation
        self.magnitude_regulariser_weight = magnitude_regulariser_weight

    def forward(self, input: T, target: T) -> T:
        assert input.shape == (target.shape[0], 3 if self.fixation else 2)
        assert target.shape == (input.shape[0], 2 if self.fixation else 1)
        
        xy_output = input[:,1:] if self.fixation else input
        angles = torch.arctan2(*xy_output.T) # MIGHT CAUSE X-Y SWAPPED WHEN PLOTTED
        angle_target = target[:,1] if self.fixation else target[:,0]
        loss_grid = 1 - (angles - angle_target).cos()

        if self.magnitude_regulariser_weight > 0:
            mags = (xy_output[:,0]**2 +  xy_output[:,1]**2)**0.5
            mag_loss = self.magnitude_regulariser_weight * torch.mean((mags - 1.0)**2)
        else:
            mag_loss = 0.0

        if self.fixation:
            fix_loss_grid = torch.binary_cross_entropy_with_logits(input[:,0], target[:,0])
            loss_grid = torch.stack([fix_loss_grid, loss_grid], axis = 1)

        return loss_grid, mag_loss
