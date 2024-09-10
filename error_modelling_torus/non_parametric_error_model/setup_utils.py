import torch
from torch import Tensor as _T

from typing import Optional

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import VALID_EMISSION_TYPES

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.swap_function import NonParametricSwapFunctionExpCos, NonParametricSwapFunctionWeiland, SpikeAndSlabSwapFunction
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.emissions import VonMisesParametricErrorsEmissions, WrappedStableParametricErrorsEmissions, UniformParametricErrorsEmissions, SmoothedWeightedDeltasErrorsEmissions
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel


def setup_model_whole(
    swap_type, swap_kernel, emission_type, all_set_sizes, remove_uniform, include_pi_u_tilde, trainable_kernel_delta,
    R_per_dim, fix_non_swap, include_pi_1_tilde, fix_inducing_point_locations, symmetricality_constraint, inducing_point_variational_parameterisation,
    shared_swap_function = False, shared_emission_distribution = False, min_seps: Optional[_T] = None, device='cuda'
    ):
    """
    If min_seps is provided, expecting it in shape [len(all_set_sizes), D], all > 0
        if shared_swap_function then take min along the first dimension
    Remember that when loading a model, there is no need to provide min_seps - it's just for inducing points initalisation, and loading a state dict with solve this!
    """

    if swap_type == 'full':
        make_variational_model = lambda min_seps: NonParametricSwapErrorsVariationalModel(R_per_dim = R_per_dim, num_features = 2, fix_non_swap = fix_non_swap, fix_inducing_point_locations = fix_inducing_point_locations, symmetricality_constraint = symmetricality_constraint, min_seps = min_seps, inducing_point_variational_parameterisation = inducing_point_variational_parameterisation).to(device)
        D = 2
    elif swap_type in ['cue_dim_only', 'est_dim_only']:
        make_variational_model = lambda min_seps: NonParametricSwapErrorsVariationalModel(R_per_dim = R_per_dim, num_features = 1, fix_non_swap = fix_non_swap, fix_inducing_point_locations = fix_inducing_point_locations, symmetricality_constraint = symmetricality_constraint, min_seps = min_seps, inducing_point_variational_parameterisation = inducing_point_variational_parameterisation).to(device)
        D = 1   # Only case where it has to be changed
    elif swap_type == 'spike_and_slab':
        make_variational_model = lambda *x: torch.nn.Identity() # No parameters to be saved here!
        D = 0 

    kernel_type_classes = {
        'exp_cos': NonParametricSwapFunctionExpCos,
        'weiland': NonParametricSwapFunctionWeiland,
    }

    if swap_type == 'spike_and_slab':
        assert swap_kernel == 'exp_cos' # i.e. default
        make_swap_function = lambda logits_set_sizes: SpikeAndSlabSwapFunction(logits_set_sizes, remove_uniform).to(device)
    else:   # XXX: different arguments not allowed yet!
        make_swap_function = lambda kernel_set_sizes: kernel_type_classes[swap_kernel](D, kernel_set_sizes, trainable_kernel_delta, remove_uniform, include_pi_u_tilde, fix_non_swap, include_pi_1_tilde).to(device)

    if swap_type == 'cue_dim_only':
        delta_dimensions = [0]    # Only locations
    elif swap_type == 'est_dim_only':
        delta_dimensions = [1]    # Only colours
    else:
        delta_dimensions = ...

    if emission_type in VALID_EMISSION_TYPES:
        emission_type_classes = {
            "von_mises": VonMisesParametricErrorsEmissions,
            "wrapped_stable": WrappedStableParametricErrorsEmissions,
            "uniform": UniformParametricErrorsEmissions,
        }
        make_emissions_model = lambda emissions_set_sizes: emission_type_classes[emission_type](emissions_set_sizes)
    #    make_emissions_model = lambda emissions_set_sizes: SmoothedWeightedDeltasErrorsEmissions(emissions_set_sizes, args.delta_smoother_kappa, args.initial_distribution_kappa)

    make_generative_model = lambda func_ss, ems_ss: NonParametricSwapErrorsGenerativeModel(
        swap_function=make_swap_function(func_ss), error_emissions=make_emissions_model(ems_ss)
    )

    variational_model, variational_models = None, None

    if shared_emission_distribution:
        generative_model = make_generative_model(None, None).to(device)
    else:
        generative_model = make_generative_model(all_set_sizes, all_set_sizes).to(device)

    if min_seps is not None:
        assert list(min_seps.shape) == [len(all_set_sizes), 2], f"Not expecting min_seps of shape {list(min_seps.shape)}"
        assert (min_seps >= 0.0).all()  # a subset of dimensions might not have min sep
        assert fix_non_swap, "Should not have a min separation if delta = 0 is in the swap functions domain"
        assert swap_type != 'spike_and_slab', "Specifying min_seps does not make sense for spike_and_slab model!"
        min_seps = min_seps[:,delta_dimensions]
        if shared_swap_function:
            variational_model = make_variational_model(min_seps.min(0))
        else:
            variational_models = {N: make_variational_model(min_sep_set_size) for min_sep_set_size, N in zip(min_seps, all_set_sizes)}
    else:
        if shared_swap_function:
            variational_model = make_variational_model(None)
        else:
            variational_models = {N: make_variational_model(None) for N in all_set_sizes}



    
    return generative_model, variational_models, variational_model, D, delta_dimensions



