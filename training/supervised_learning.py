from typing import Optional
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

