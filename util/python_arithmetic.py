from typing import List
import torch


def assert_all_shapes_same(a: List[torch.Tensor], b: List[torch.Tensor]):
    assert all([_a.shape == _b.shape for _a, _b in zip(a, b)])


def subtract_list_of_tensors(a: List[torch.Tensor], b: List[torch.Tensor]):
    assert_all_shapes_same(a, b)
    return [_a - _b for _a, _b in zip(a, b)]


def multiply_list_of_tensors(a: List[torch.Tensor], b: List[torch.Tensor]):
    assert_all_shapes_same(a, b)
    return [_a * _b for _a, _b in zip(a, b)]


def add_list_of_tensors(a: List[torch.Tensor], b: List[torch.Tensor]):
    assert_all_shapes_same(a, b)
    return [_a + _b for _a, _b in zip(a, b)]


def divide_list_of_tensors(a: List[torch.Tensor], b: List[torch.Tensor]):
    assert_all_shapes_same(a, b)
    return [_a / _b for _a, _b in zip(a, b)]


def sort_dict_by_master_value(_dict, _key):
    idx_map = list(range(len(_dict[_key])))
    idx_map = sorted(idx_map, key = lambda _i: _dict[_key][_i])
    out_dict = {k: v[idx_map] for k, v in _dict.items()}
    return out_dict
