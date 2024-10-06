import torch
from torch import Tensor as _T

from typing import Optional, List, Tuple, Union

import os

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.main import WorkingMemorySimpleSwapModel, WorkingMemoryFullSwapModel


def setup_model_whole(
    *_, 
    num_models: int,
    swap_type: str,
    kernel_type: Optional[str],
    emission_type: str,
    fix_non_swap: bool,
    remove_uniform: bool,
    include_pi_u_tilde: bool,
    include_pi_1_tilde: bool,
    normalisation_inner: str,
    shared_swap_function: bool,
    shared_emission_distribution: bool,
    all_set_sizes: List[int],
    trainable_kernel_delta: bool,
    R_per_dim: int,
    fix_inducing_point_locations: bool,
    all_min_seps: Optional[_T],
    inducing_point_variational_parameterisation_type: str,
    symmetricality_constraint: bool,
    num_variational_samples: int,
    num_importance_sampling_samples: int,
    resume_path: Optional[str],
    device = 'cuda',
    regenerate_kernel_until_stability: bool = True,
) -> Tuple[Union[WorkingMemorySimpleSwapModel, WorkingMemoryFullSwapModel], int, List[int]]:
    variational_models, delta_dimensions, D = NonParametricSwapErrorsVariationalModel.from_typical_args(
        num_models = num_models,
        swap_type = swap_type,
        R_per_dim = R_per_dim,
        fix_non_swap = fix_non_swap,
        fix_inducing_point_locations = fix_inducing_point_locations,
        all_min_seps = all_min_seps,
        inducing_point_variational_parameterisation_type = inducing_point_variational_parameterisation_type,
        symmetricality_constraint = symmetricality_constraint,
        shared_swap_function = shared_swap_function,
        all_set_sizes = all_set_sizes,
        device = device,
    )

    stable_kernel_generated = False
    failed_attempts = 0

    while not stable_kernel_generated:

        print(f'Initialising generative_model after {failed_attempts} failed attempts at stability...')

        generative_model = NonParametricSwapErrorsGenerativeModel.from_typical_args(
            num_models = num_models,
            swap_type = swap_type,
            kernel_type = kernel_type,
            emission_type = emission_type,
            fix_non_swap = fix_non_swap,
            remove_uniform = remove_uniform,
            include_pi_u_tilde = include_pi_u_tilde,
            include_pi_1_tilde = include_pi_1_tilde,
            normalisation_inner = normalisation_inner,
            shared_swap_function = shared_swap_function,
            shared_emission_distribution = shared_emission_distribution,
            all_set_sizes = all_set_sizes,
            trainable_kernel_delta = trainable_kernel_delta,
            num_features = D,
            device = device,
        )

        if swap_type == 'spike_and_slab':
            swap_model = WorkingMemorySimpleSwapModel(generative_model)
            this_kernel_stable = True
        else:
            swap_model = WorkingMemoryFullSwapModel(generative_model, variational_models, num_variational_samples, num_importance_sampling_samples)
            this_kernel_stable = swap_model.check_Kuu_stability()

        stable_kernel_generated = (
            this_kernel_stable or (not regenerate_kernel_until_stability)
        )

    if resume_path is not None:
        map_location=None if device == 'cuda' else torch.device('cpu')
        parameter_load_path = os.path.join(resume_path, '{model}.{exl}')
        swap_model.load_state_dict(torch.load(parameter_load_path.format(model = f'swap_model', ext = 'mdl'), map_location=map_location))

        # if (emission_type == 'residual_deltas'):
        #     emissions_data = torch.load(parameter_load_path.format(model = f'generative_model_emission_histogram', ext = 'data'))
        #     for set_size, load in emissions_data.items():
        #         swap_model.generative_model.error_emissions.load_new_distribution(set_size, load['inference_locations'].to(device), load['inference_weights'].to(device))

    return swap_model, D, delta_dimensions
