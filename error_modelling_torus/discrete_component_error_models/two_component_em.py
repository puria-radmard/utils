import torch
from torch import pi
from torch import Tensor as _T

from tqdm import tqdm

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


def total_likelihoods(data_squared: _T, p_u: _T, sigma2_e: _T):
    "Expecting data as size [num_datasets, dataset_size], and both parameters as scalars"
    uniform_term = p_u / (2 * pi)
    norm_term = (1 - p_u) / (2 * pi * sigma2_e)**0.5
    exp_term = torch.exp(- data_squared / (2 * sigma2_e))
    return (uniform_term + (norm_term * exp_term)).log().sum()


def e_step(data_squared: _T, previous_p_u, previous_sigma2_e):
    "Expecting data as size [num_datasets, dataset_size], and both parameters as scalars"
    sigma_pu = previous_p_u * (previous_sigma2_e ** 0.5)
    norm_term = ((2 * pi) ** 0.5) * (1 - previous_p_u)
    exp_term = torch.exp(- data_squared / (2 * previous_sigma2_e))
    new_qs = (sigma_pu) / (sigma_pu + (norm_term * exp_term))
    return 1 - new_qs


def m_step_p_u(previous_qs: _T):
    "Expecting q is shape [num_datasets, dataset_size], since each is just a Bernoulli parameter"
    assert torch.all(previous_qs >= 0.0) and torch.all(previous_qs <= 1.0)
    return 1 - previous_qs.mean()


def m_step_sigma2_e(previous_qs: _T, data_squared: _T):
    "Expecting q is shape [num_datasets, dataset_size], since each is just a Bernoulli parameter"
    assert torch.all(previous_qs >= 0.0) and torch.all(previous_qs <= 1.0)
    return (previous_qs * data_squared).sum() / previous_qs.sum()


def full_em_algorithm(dataset: _T, num_iter, use_tqdm):
    "Expecting data as size [num_datasets, dataset_size]"

    print("WARNING! full_em_algorithm STILL USES THE GAUSSIAN APPROXIMATION VERSION OF THE TWO-COMPONENT MODEL")
    
    current_p_u = torch.tensor(0.1)
    current_sigma2_e = torch.tensor(0.3)
    squared_dataset = rectify_angles(dataset).square()
    all_lhs = []

    for i in tqdm(range(num_iter), disable=not use_tqdm):

        current_qs = e_step(squared_dataset, current_p_u, current_sigma2_e)
        current_p_u = m_step_p_u(current_qs)
        current_sigma2_e = m_step_sigma2_e(current_qs, squared_dataset)

        all_lhs.append(total_likelihoods(squared_dataset, current_p_u, current_sigma2_e))

    return current_p_u, current_sigma2_e, all_lhs
