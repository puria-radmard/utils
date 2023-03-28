from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as T
from typing import Type, Union

import copy


from purias_utils.rnn.augmentation.mitosis import recurrent_mitosis, feedforward_mitosis


class WeightLayer(nn.Module):
    """
    Stackable augmentations to weight matrix!

    self.base_matrix propagates all the way back to the raw matrix
    self.base_matrix gives the masked
    """

    def __init__(self, base_matrix: Union[T, Type[WeightLayer]]) -> None:
        super().__init__()

        if isinstance(base_matrix, T):
            base_matrix = WeightLayerBase(base_matrix)
        self._base_matrix: WeightLayer = base_matrix

    @property
    def masked_weight(self) -> T:
        "Specific to subclass"
        raise NotImplementedError

    @property
    def shape(self):
        return self.masked_weight.shape

    @property
    def num_neurons(self):
        """NB: must adhere! See mitosis functions. Might have to rewrite this in cases where neurons are added by mask"""
        return self.raw_matrix.shape[0]

    @property
    def raw_matrix(self):
        """
        This propagates back to the WeightLayerBase layer!
        """
        return self._base_matrix.raw_matrix

    @raw_matrix.setter
    def raw_matrix(self, new_raw_matrix):
        self._base_matrix.raw_matrix = new_raw_matrix

    @property
    def base_matrix(self):
        return self._base_matrix.masked_weight

    def recurrent_mitosis(self, i):
        new_W = recurrent_mitosis(self.raw_matrix.data.clone(), i)
        new = copy.deepcopy(self)
        new.raw_matrix = new_W
        return new

    def feedforward_mitosis(self, i):
        new_W = feedforward_mitosis(self.raw_matrix.data.clone(), i)
        new = copy.deepcopy(self)
        new.raw_matrix = new_W
        return new

    def forward(self, x: T):
        return self.masked_weight @ x



class WeightLayerBase(WeightLayer):

    def __init__(self, base_matrix: T):
        super(WeightLayer, self).__init__()
        self.register_parameter(name='_base_matrix', param=torch.nn.Parameter(base_matrix))

    @property
    def masked_weight(self):
        "Base case"
        return self._base_matrix

    @property
    def raw_matrix(self):
        "Base case"
        return self._base_matrix

    @raw_matrix.setter
    def raw_matrix(self, new_raw_matrix):
        if not isinstance(new_raw_matrix, torch.nn.Parameter):
            new_raw_matrix = torch.nn.Parameter(new_raw_matrix)
        self._base_matrix = new_raw_matrix

    @property
    def base_matrix(self):
        "Base case"
        return self._base_matrix



class AbsWeightLayer(WeightLayer):

    @property
    def masked_weight(self):
        return self.base_matrix.abs()


class ConstantWeightLayer(WeightLayer):
    """
    No parameter assignment
    """

    def __init__(self, base_matrix: T) -> None:
        super(WeightLayer, self).__init__()
        self._base_matrix = base_matrix

    @property
    def masked_weight(self):
        return self._base_matrix
    

class NoAutapse(WeightLayer):

    @property
    def masked_weight(self):
        return self._base_matrix.masked_weight.fill_diagonal_(0.0)
