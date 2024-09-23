raise Exception('Deprecated/need to update')

import torch
from torch import pi
import random
from torch import Tensor as T

def random_in_range(upper, lower):
    return random.random() * (upper - lower) + lower


def construct_kappa_matrix_from_chol_kappa_squared_elements(D, *matrix_coeffs):

    num_coeffs = len(matrix_coeffs)

    if num_coeffs == 1:
        matrix = torch.eye(D).float() * matrix_coeffs[0]

    elif num_coeffs == D:
        matrix = torch.diag(torch.tensor(matrix_coeffs)).float()

    elif num_coeffs == 0.5 * D * (D+1):
        assert D == 2, "Not done the sqrt checks for D!=2"

        starting_idx_matrix = 0
        starting_idx_coeffs = 0
        matrix = torch.zeros(D, D).float()
        for i in range(D):
            matrix[i, starting_idx_matrix:] = torch.tensor(matrix_coeffs[starting_idx_coeffs:starting_idx_coeffs+D-starting_idx_matrix])
            matrix[starting_idx_matrix:, i] = torch.tensor(matrix_coeffs[starting_idx_coeffs:starting_idx_coeffs+D-starting_idx_matrix])
            starting_idx_coeffs = starting_idx_coeffs+D-starting_idx_matrix
            starting_idx_matrix += 1
        
    elif num_coeffs == D * (D+1):
        assert D == 2, "Not done the sqrt checks for D!=2"
        a_1 = random_in_range(matrix_coeffs[0], matrix_coeffs[3])
        a_2 = random_in_range(matrix_coeffs[2], matrix_coeffs[5])
        b_upper = min(a_1 + a_2, matrix_coeffs[4])
        b = random_in_range(matrix_coeffs[1], b_upper)
        return construct_kappa_matrix_from_chol_kappa_squared_elements(D, a_1, b, a_2)

    return matrix @ matrix.T

def scaled_unmasked_log_vm_from_kappa_matrix(deltas: T, Kappa: T):
    "deltas coming in shape [M, N, D], Kappa in shape [D, D]"
    _Kappa = Kappa.to(deltas.device).to(deltas.dtype)
    if len(deltas.shape) == 3:
        M, N, D = deltas.shape
        flattened_cos_x = (deltas).reshape(M*N, D).cos()
        exponent = ((flattened_cos_x @ _Kappa) * flattened_cos_x).sum(-1).reshape(M, N).sqrt()
    elif len(deltas.shape) == 2:
        flattened_cos_x = (deltas).cos()
        exponent = ((flattened_cos_x @ _Kappa) * flattened_cos_x).sum(-1).sqrt()
    return exponent

def normalise_to_interval(thing: T, upper: float = 2.0):
    scale = (thing.max() - thing.min())
    if scale != 0:
        return upper * (thing - thing.min()) / scale 
    else:
        assert 1 in thing.shape
        return upper * thing / thing.max()

def construct_f_function(kappa_squared_matrix):
    # (args.target_function_offset + vm.log_prob(deltas).exp()).sum(-1)
    def f_function(deltas: T):
        log_vm = scaled_unmasked_log_vm_from_kappa_matrix(deltas, kappa_squared_matrix).exp()
        mask = deltas.cos().sum(-1)
        masked_log_vm = mask * log_vm
        return normalise_to_interval(masked_log_vm)
    return f_function

