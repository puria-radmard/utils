import torch
from torch import Tensor as _T
from torch.nn import Parameter as P


def lesion_from_mask(parameter: P, mask: _T, lesion_value: float = 0):
    assert parameter.shape == mask.shape
    assert len(mask.shape) == 2
    parameter.data[mask] = lesion_value


def lesion_from_callable(parameter: P, key: callable, lesion_value: float = 0):
    lesion_from_mask(parameter, key(parameter), lesion_value)


def lesion_from_threshold(
    parameter: P, 
    threshold: float, 
    upper_bound: bool = False, 
    lesion_value: float = 0
):
    mask = (parameter > threshold) if upper_bound else (parameter <= threshold)
    lesion_from_mask(parameter, mask, lesion_value)
