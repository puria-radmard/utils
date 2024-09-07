import torch
from torch import Tensor as _T

from purias_utils.rnn.layers.dales import BinaryMaskRecurrent
from purias_utils.rnn.synaptic_plasticity.synapse_types import *


def update_excitatory_input_homeostatically(
    dales_matrix: BinaryMaskRecurrent, alpha_e: float, e_set: float, activity: _T,
    average_error_term: bool = False
):
    """
        This updates W_ee and W_ei, i.e. everything inputting to excitatory neurons
    """

    exc_activity = activity[sorted(list(dales_matrix.exc_indexes))]
    inh_activity = activity[sorted(list(dales_matrix.inh_indexes))]
    error_term = e_set - exc_activity
    if average_error_term:
        error_term = error_term.mean()
        exc_error = error_term * exc_activity
        inh_error = error_term * inh_activity
    else:
        exc_error = torch.outer(error_term, exc_activity)
        inh_error = torch.outer(error_term, inh_activity)

    current_e2e = get_dales_e2e(dales_matrix)
    new_e2e = current_e2e + alpha_e * exc_error
    adjust_dales_e2e(dales_matrix, new_e2e.data)

    current_i2e = get_dales_i2e(dales_matrix)
    new_i2e = current_i2e - alpha_e * inh_error
    adjust_dales_i2e(dales_matrix, new_i2e.data)
    



def update_inhibitory_input_homeostatically(
    dales_matrix: BinaryMaskRecurrent, alpha_i: float, i_set: float, activity: _T,
    average_error_term: bool = False
):
    """
        This updates W_ee and W_ei, i.e. everything inputting to inhibitory neurons
    """

    exc_activity = activity[list(dales_matrix.exc_indexes)]
    inh_activity = activity[list(dales_matrix.inh_indexes)]
    error_term = i_set - inh_activity
    if average_error_term:
        error_term = error_term.mean()
        exc_error = error_term * exc_activity
        inh_error = error_term * inh_activity
    else:
        exc_error = torch.outer(error_term, exc_activity)
        inh_error = torch.outer(error_term, inh_activity)

    current_e2i = get_dales_e2i(dales_matrix)
    new_e2i = current_e2i + alpha_i * exc_error
    adjust_dales_e2i(dales_matrix, new_e2i.data)

    current_i2i = get_dales_i2i(dales_matrix)
    new_i2i = current_i2i - alpha_i * inh_error
    adjust_dales_i2i(dales_matrix, new_i2i.data)
    



def update_excitatory_input_cross_homeostatically(
    dales_matrix: BinaryMaskRecurrent, alpha_e: float, i_set: float, activity: _T,
    average_error_term: bool = False
):
    """
        This updates W_ee and W_ei, i.e. everything inputting to excitatory neurons
    """

    exc_activity = activity[list(dales_matrix.exc_indexes)]
    inh_activity = activity[list(dales_matrix.inh_indexes)]
    error_term = i_set - inh_activity
    if average_error_term:
        error_term = error_term.mean()
        exc_error = error_term * exc_activity
        inh_error = error_term * inh_activity
    else:
        exc_error = torch.outer(error_term, exc_activity)
        inh_error = torch.outer(error_term, inh_activity)

    current_e2e = get_dales_e2e(dales_matrix)
    new_e2e = current_e2e + alpha_e * exc_error
    adjust_dales_e2e(dales_matrix, new_e2e.data)

    current_i2e = get_dales_i2e(dales_matrix)
    new_i2e = current_i2e - alpha_e * inh_error
    adjust_dales_i2e(dales_matrix, new_i2e.data)
    


def update_inhibitory_input_cross_homeostatically(
    dales_matrix: BinaryMaskRecurrent, alpha_i: float, e_set: float, activity: _T,
    average_error_term: bool = False
):
    """
        This updates W_ee and W_ei, i.e. everything inputting to inhibitory neurons
    """

    exc_activity = activity[list(dales_matrix.exc_indexes)]
    inh_activity = activity[list(dales_matrix.inh_indexes)]
    error_term = e_set - exc_activity
    if average_error_term:
        error_term = error_term.mean()
        exc_error = error_term * exc_activity
        inh_error = error_term * inh_activity
    else:
        exc_error = torch.outer(error_term, exc_activity)
        inh_error = torch.outer(error_term, inh_activity)

    current_e2i = get_dales_e2i(dales_matrix)
    new_e2i = current_e2i - alpha_i * exc_error
    adjust_dales_e2i(dales_matrix, new_e2i.data)

    current_i2i = get_dales_i2i(dales_matrix)
    new_i2i = current_i2i + alpha_i * inh_error
    adjust_dales_i2i(dales_matrix, new_i2i.data)
