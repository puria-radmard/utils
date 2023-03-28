import torch
from torch import Tensor as T
from purias_utils.rnn.layers.dales import BinaryMaskRecurrent

__all__ = [
    'get_dales_e2e',
    'get_dales_e2i',
    'get_dales_i2e',
    'get_dales_i2i',
    'adjust_dales_e2e',
    'adjust_dales_e2i',
    'adjust_dales_i2e',
    'adjust_dales_i2i',
]


def get_dales_matrix_by_synapse_type(
    dales_matrix: BinaryMaskRecurrent, from_type: str, to_type: str, _abs: bool
    ):
    "Gets Dales form, so uses self.masked_weight"

    assert from_type in {'e', 'i'}
    assert to_type in {'e', 'i'}

    row_indexes = sorted(list(dales_matrix.exc_indexes if to_type == 'e' else dales_matrix.inh_indexes))
    col_indexes = sorted(list(dales_matrix.exc_indexes if from_type == 'e' else dales_matrix.inh_indexes))

    result_columns = []
    for ri in row_indexes:
        result_columns.append(dales_matrix.raw_matrix.data[ri,col_indexes].unsqueeze(0))

    result = torch.cat(result_columns, 0)
    if _abs:
        result = result.abs()

    return result


def get_dales_e2e(dales_matrix: BinaryMaskRecurrent, _abs=True):
    return get_dales_matrix_by_synapse_type(dales_matrix, from_type='e', to_type='e', _abs=_abs)

def get_dales_e2i(dales_matrix: BinaryMaskRecurrent, _abs=True):
    return get_dales_matrix_by_synapse_type(dales_matrix, from_type='e', to_type='i', _abs=_abs)

def get_dales_i2e(dales_matrix: BinaryMaskRecurrent, _abs=True):
    return get_dales_matrix_by_synapse_type(dales_matrix, from_type='i', to_type='e', _abs=_abs)

def get_dales_i2i(dales_matrix: BinaryMaskRecurrent, _abs=True):
    return get_dales_matrix_by_synapse_type(dales_matrix, from_type='i', to_type='i', _abs=_abs)



def adjust_dales_matrix_by_synapse_type(
    dales_matrix: BinaryMaskRecurrent, new_weights: T, from_type: str, to_type: str
    ):

    "Adjusts actual values, so uses self.raw_matrix.data"

    assert from_type in {'e', 'i'}
    assert to_type in {'e', 'i'}

    row_indexes = sorted(list(dales_matrix.exc_indexes if to_type == 'e' else dales_matrix.inh_indexes))
    col_indexes = sorted(list(dales_matrix.exc_indexes if from_type == 'e' else dales_matrix.inh_indexes))

    assert len(row_indexes) == len(new_weights)

    for i, ri in enumerate(row_indexes):
        dales_matrix.raw_matrix.data[ri,col_indexes] = new_weights[i]



def adjust_dales_e2e(dales_matrix: BinaryMaskRecurrent, new_weights: T):
    adjust_dales_matrix_by_synapse_type(dales_matrix, new_weights, from_type='e', to_type='e')

def adjust_dales_e2i(dales_matrix: BinaryMaskRecurrent, new_weights: T):
    adjust_dales_matrix_by_synapse_type(dales_matrix, new_weights, from_type='e', to_type='i')

def adjust_dales_i2e(dales_matrix: BinaryMaskRecurrent, new_weights: T):
    adjust_dales_matrix_by_synapse_type(dales_matrix, new_weights, from_type='i', to_type='e')

def adjust_dales_i2i(dales_matrix: BinaryMaskRecurrent, new_weights: T):
    adjust_dales_matrix_by_synapse_type(dales_matrix, new_weights, from_type='i', to_type='i')


