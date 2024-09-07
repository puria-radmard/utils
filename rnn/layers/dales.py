from typing import Iterable

import torch
from torch import Tensor as _T
from typing import Type, Union, List

from purias_utils.rnn.layers.base import WeightLayer


class BinaryMaskRecurrent(WeightLayer):

    """
    NB: if doing Dale's law for recurrent matrix, should use this to wrap an AbsWeightLayer
    """

    def __init__(self, base_matrix: Union[_T, Type[WeightLayer]], exc_indexes: Iterable, exc_mask = 1.0, inh_mask = -1.0):
        super().__init__(base_matrix)
        self.exc_indexes = set(exc_indexes)
        self.inh_indexes = set([i for i in range(self.num_neurons) if i not in exc_indexes])
        self.exc_mask = exc_mask
        self.inh_mask = inh_mask

    @property
    def masked_weight(self):
        mask = self.inh_mask * torch.ones(self.base_matrix.shape)
        mask[:,list(self.exc_indexes)] = self.exc_mask
        return self.base_matrix * mask.to(self.raw_matrix.device)

    @property
    def num_inh(self):
        return len(self.inh_indexes)

    @property
    def num_exc(self):
        return len(self.exc_indexes)

    def is_exc(self, i):
        return i in self.exc_indexes

    def recurrent_mitosis(self, i):
        """code not DRY!!"""
        new = super().recurrent_mitosis(i)
        new_exc_indexes = [j for j in self.exc_indexes if j <= i]
        new_exc_indexes += [j + 1 for j in self.exc_indexes if j >= i]
        new.exc_indexes = set(new_exc_indexes)
        new.inh_indexes = set([i for i in range(new.num_neurons) if i not in new_exc_indexes])
        return new



class MatrixShapedBinaryMaskRecurrent(BinaryMaskRecurrent):

    "Used in mixer architectures. Input comes in as shape [... P, C, Ch] and weight is of shape [P, C, C]"

    def __init__(self, base_matrix: Union[_T, Type[WeightLayer]], exc_indexes: Iterable, exc_mask=1, inh_mask=-1):
        super().__init__(base_matrix, exc_indexes, exc_mask, inh_mask)
        if (inh_mask != 1.0) or (exc_indexes != []):
            raise NotImplementedError

    def forward(self, x: _T):
        return self.masked_weight @ x



class BinaryMaskForward(BinaryMaskRecurrent):

    def __init__(self, base_matrix: Union[_T, Type[WeightLayer]], exc_indexes: Iterable, exc_mask=1, inh_mask=0, exempt_indices: List[int]=[0]):
        super().__init__(base_matrix, exc_indexes, exc_mask, inh_mask)
        self.exempt_indices = exempt_indices

    @property
    def masked_weight(self):
        mask = self.inh_mask * torch.ones_like(self.base_matrix)
        mask[list(self.exc_indexes),:] = self.exc_mask
        for ei in self.exempt_indices:
            mask[:,ei] = 1.0                # i.e. no E or I for fixation/extra information

        return self.base_matrix * mask
