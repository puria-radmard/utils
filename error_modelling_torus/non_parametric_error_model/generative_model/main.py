
import torch
from torch import nn
from torch import Tensor as _T
from torch.nn import ModuleDict

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from typing import Optional, Dict, List, Any


from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.emissions import ErrorsEmissionsBase, VonMisesParametricErrorsEmissions, WrappedStableParametricErrorsEmissions, DoubleVonMisesParametricErrorsEmissions
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.swap_function import SwapFunctionBase, NonParametricSwapFunctionExpCos, NonParametricSwapFunctionWeiland, SpikeAndSlabSwapFunction


KERNEL_TYPE_CLASSES = {
    'exp_cos': NonParametricSwapFunctionExpCos,
    'weiland': NonParametricSwapFunctionWeiland,
}

EMISSION_TYPE_CLASSES = {
    "von_mises": VonMisesParametricErrorsEmissions,
    "wrapped_stable": WrappedStableParametricErrorsEmissions,
    "double_von_mises": DoubleVonMisesParametricErrorsEmissions,
    # "uniform": UniformParametricErrorsEmissions,
}


class NonParametricSwapErrorsGenerativeModel(nn.Module):

    def __init__(self, num_models: int, swap_function: SwapFunctionBase, error_emissions: ErrorsEmissionsBase) -> None:
        super().__init__()

        self.num_models = num_models
        self.swap_function = swap_function
        self.error_emissions = error_emissions
        assert num_models == swap_function.num_models == error_emissions.num_models

        if (self.swap_function.function_set_sizes is not None) and (self.error_emissions.emissions_set_sizes is not None):
            assert self.swap_function.function_set_sizes == self.error_emissions.emissions_set_sizes

    @classmethod
    def from_typical_args(
        cls, *_, 
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
        num_features: int,
        device = 'cuda',
        error_emissions_keys: Optional[List[Any]],
        **kwargs
    ):
        """
        This is to replace the old setup_utils.py function logic!
        """
        swap_function_set_sizes = None if shared_swap_function else all_set_sizes
        emission_set_sizes = None if shared_emission_distribution else all_set_sizes
        
        if swap_type == 'spike_and_slab':
            swap_function = SpikeAndSlabSwapFunction(
                num_models = num_models,
                logits_set_sizes = swap_function_set_sizes,
                remove_uniform = remove_uniform,
                include_pi_u_tilde = include_pi_u_tilde,
                include_pi_1_tilde = include_pi_1_tilde,
                normalisation_inner = normalisation_inner
            ).to(device)

        else:
            KernelClass = KERNEL_TYPE_CLASSES[kernel_type]
            swap_function = KernelClass(
                num_models = num_models,
                num_features = num_features,
                kernel_set_sizes = swap_function_set_sizes,
                trainable_kernel_delta = trainable_kernel_delta,
                remove_uniform = remove_uniform,
                include_pi_u_tilde = include_pi_u_tilde,
                fix_non_swap = fix_non_swap,
                include_pi_1_tilde = include_pi_1_tilde,
                normalisation_inner = normalisation_inner
            ).to(device)

        EmissionsClass = EMISSION_TYPE_CLASSES[emission_type]

        if error_emissions_keys is None:
            error_emissions = EmissionsClass(num_models, emission_set_sizes)
        else:
            error_emissions = {smk: EmissionsClass(num_models, emission_set_sizes) for smk in error_emissions_keys}

        return cls(num_models, swap_function, error_emissions)

    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.num_models = 1
        self.swap_function.reduce_to_single_model(model_index)
        try:
            self.error_emissions.reduce_to_single_model(model_index)
        except AttributeError:
            pass

    def get_marginalised_log_likelihood(self, estimation_deviations: _T, pi_vectors: _T, kwargs_for_individual_component_likelihoods: dict = {}, *_, _error_emissions_obj: Optional[ErrorsEmissionsBase] = None):
        """
        This is the first term of the ELBO when training on the estimates

        estimation_deviations ([Q, M, N])  = rectify(estimates - zeta_c)
        pi_vectors of shape [Q, M, N+1] = output of self.swap_function.generate_pi_vectors

        Generate a grid of shape [Q, M, N+1] giving the individual components of the lh term (inside the log)
        Aggregate these also, giving a [Q] vector of overal log-likelihoods

        Also do infernce built in, given that it's cheap -> posteriors shaped [Q, M, N+1]
        """
        if _error_emissions_obj is None:
            _error_emissions_obj = self.error_emissions

        set_size = estimation_deviations.shape[-1]
        individual_component_likelihoods = _error_emissions_obj.individual_component_likelihoods_from_estimate_deviations(
            set_size, estimation_deviations, **kwargs_for_individual_component_likelihoods
        )   # [Q, M, N+1] - p(y[m] | beta[n], Z[m])
        
        #import matplotlib.pyplot as plt
        #plt.scatter(estimation_deviations.flatten().detach().cpu().numpy(), individual_component_likelihoods[0,:,1:].flatten().detach().cpu().numpy())
        #plt.savefig('individual_component_likelihoods.png')
        try:
            joint_component_and_error_likelihood = individual_component_likelihoods * pi_vectors    # [Q, M, N+1] - p(y[m] | beta[n], Z[m]) * p(beta[n]| Z[m]) = p(y[m], beta[n] | Z[m])
        except RuntimeError:
            joint_component_and_error_likelihood = individual_component_likelihoods * pi_vectors.to(device = individual_component_likelihoods.device)
        likelihood_per_datapoint = joint_component_and_error_likelihood.sum(-1).log()           # [Q, M, N+1] -> [Q, M] -> [Q, M] = log p(y[m] | Z[m])
        total_log_likelihood = likelihood_per_datapoint.sum(-1)                                 # [Q]

        # [Q, M, N+1] - p(y[m], beta[n] | Z[m]) / p(y[m] | Z[m]) =  p(beta[n] | y[m], Z[m])
        posterior_vectors = joint_component_and_error_likelihood / joint_component_and_error_likelihood.sum(-1, keepdim = True)

        return total_log_likelihood, likelihood_per_datapoint, posterior_vectors    # [Q], [Q, M], [Q, M, N+1]

    def get_component_log_likelihood(self, selected_components: _T, pi_vectors: _T):
        """
        This is the first term of the ELBO when training on the selected components/betas (i.e. synthetic data only)

        Very simple loglikelihood here

        selected_components [M] = betas from (synthetic) generative model
        pi_vectors of shape [I, M, N+1], i.e. still generated as before by variational model

        TODO: update for pre-meaned pi vectors!
        """
        raise Exception
        I, M, N_plus_1 = pi_vectors.shape
        selected_pis = torch.zeros(I, M).to(pi_vectors.device).to(pi_vectors.dtype)
        for m in range(M):
            b = selected_components[m]
            selected_pis[:,m] = pi_vectors[:,m,b]
        return selected_pis.mean(0).log().sum()
    
    def data_generation_from_component_priors(self, set_size: int, vm_means: _T, component_priors: _T, kwargs_for_sample_betas: dict = {}, kwargs_for_sample_from_components: dict = {}, *_, _error_emissions_obj: Optional[ErrorsEmissionsBase]):
        """
        vm_means is basically zeta_recalled: [M, N]
        component_priors are [Q, M, N+1]
        """
        if _error_emissions_obj is None:
            _error_emissions_obj = self.error_emissions

        betas = self.swap_function.sample_betas(component_priors, **kwargs_for_sample_betas)
        samples = _error_emissions_obj.sample_from_components(set_size, betas, vm_means, **kwargs_for_sample_from_components)
        return {'pi_vectors': component_priors, 'betas': betas, 'samples': samples}

    def full_data_generation(self, set_size: int, vm_means: _T, kwargs_for_generate_pi_vectors: dict = {}, kwargs_for_sample_betas: dict = {}, kwargs_for_sample_from_components: dict = {}, *_, _error_emissions_obj: Optional[ErrorsEmissionsBase]):
        """
        vm_means is basically zeta_recalled: [M, N]
        """
        if _error_emissions_obj is None:
            _error_emissions_obj = self.error_emissions

        with torch.no_grad():
            prior_info = self.swap_function.generate_pi_vectors(set_size=set_size, **kwargs_for_generate_pi_vectors)
            betas = self.swap_function.sample_betas(prior_info['pis'], **kwargs_for_sample_betas)
            samples = _error_emissions_obj.sample_from_components(set_size, betas, vm_means, **kwargs_for_sample_from_components)
        return {'exp_grid': prior_info['exp_grid'], 'pi_vectors': prior_info['pis'], 'betas': betas, 'samples': samples}

    def empirical_residual_distribution_weights(self, posterior_vectors: _T, estimation_deviations: _T, kwargs_for_individual_component_likelihoods: dict = {}, *_, _error_emissions_obj: Optional[ErrorsEmissionsBase]):
        """
        posterior_vectors: [Q, M, N+1] - p(beta[n] | y[m], Z[m])       (f samples already factored out)
        estimation_deviations: [Q, M, N] - rectify(estimates - zeta_c)

        Implementation is a little different to what is said in the latex, but generates effectively the same result...

        All output values of shape [Q, M, N]
        
        XXX -- NB: division by M not done here for some reason... TODO: downstream debug!
        """
        if _error_emissions_obj is None:
            _error_emissions_obj = self.error_emissions

        particle_weights_non_uniform = posterior_vectors[...,1:].detach() # posterior_vectors [Q, M, N]
        set_size = estimation_deviations.shape[-1]
        with torch.no_grad():
            if (posterior_vectors[...,[0]] != 0.0).any():

                dense_grid = torch.linspace(-torch.pi, +torch.pi, 5 * estimation_deviations[0].numel(), device = estimation_deviations.device).unsqueeze(0).repeat(self.num_models, 1) # [Q, many]
                grid_point_distance = dense_grid[0,1] - dense_grid[0,0]

                error_lhs: _T = _error_emissions_obj.individual_component_likelihoods_from_estimate_deviations_inner(set_size, dense_grid, **kwargs_for_individual_component_likelihoods)   # [Q, many]
                error_lhs = error_lhs / (grid_point_distance * error_lhs).sum(-1, keepdim=True)     # basically 1
                error_lhs = grid_point_distance * error_lhs
                
                dense_grid = dense_grid.unsqueeze(1)    # [Q, 1, many]
                reshaped_epsilons = estimation_deviations.reshape(self.num_models, -1, 1)             # [Q, MN, 1] --> reshaped_epsilons.reshape(*estimation_deviations.shape) == estimation_deviations
                distance_of_grid_points_to_epsilons = rectify_angles(dense_grid - reshaped_epsilons).abs()  # [Q, MN, many]
                eval_point_assigment_to_epsilon_index = distance_of_grid_points_to_epsilons.argmin(1)       # [Q, many]

                epsilon_assigment = []
                for q in range(self.num_models):
                    epsilon_assigment.append(
                        torch.stack([error_lhs[q][eval_point_assigment_to_epsilon_index[q]==ii].sum() for ii in range(distance_of_grid_points_to_epsilons.shape[1])])       # [MN] XXX: super inefficient!
                    )
                epsilon_assigment = torch.stack(epsilon_assigment, 0)   # [Q, MN]

                particle_weights_uniform_unscaled = epsilon_assigment.reshape(*estimation_deviations.shape) * estimation_deviations.shape[1]    # [Q, M, N]
                particle_weights_uniform = particle_weights_uniform_unscaled * posterior_vectors[...,[0]].detach().mean(0, keepdim=True)        # [Q, M, N]
                
                # import matplotlib.pyplot as plt
                # plt.clf()
                # plt.scatter(dense_grid.flatten().cpu(), error_lhs.flatten().cpu(), label = 'error_lhs')
                # plt.scatter(reshaped_epsilons.flatten().cpu(), epsilon_assigment.cpu(), label = 'particle_weights_uniform_unscaled')
                # plt.legend()
                # plt.savefig('prior_particles')

            else:
                particle_weights_uniform = torch.zeros_like(
                    particle_weights_non_uniform, 
                    device = particle_weights_non_uniform.device, 
                    dtype = particle_weights_non_uniform.dtype
                )
        return {
            "particle_weights_non_uniform": particle_weights_non_uniform,
            "particle_weights_uniform": particle_weights_uniform,
            "particle_weights_total": set_size * (particle_weights_non_uniform + particle_weights_uniform),
        }




class MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel(NonParametricSwapErrorsGenerativeModel):

    def __init__(self, num_models: int, swap_function: SwapFunctionBase, error_emissions_dict: Dict[Any, ErrorsEmissionsBase]) -> None:
        super(NonParametricSwapErrorsGenerativeModel, self).__init__()

        self.num_models = num_models
        self.swap_function = swap_function
        self.error_emissions: Dict[str, ErrorsEmissionsBase] = ModuleDict(
            {str(k): v for k, v in error_emissions_dict.items()}
        )
        assert all([num_models == swap_function.num_models == ee.num_models for ee in self.error_emissions.values()])

        self.error_emissions_keys = list(error_emissions_dict.keys())

        if (self.swap_function.function_set_sizes is not None) and (self.get_error_emissions(self.error_emissions_keys[0]).emissions_set_sizes is not None):
            for ee in self.error_emissions.values():
                assert self.swap_function.function_set_sizes == ee.emissions_set_sizes

    def get_error_emissions(self, key: Any):
        return self.error_emissions[str(key)]

    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.num_models = 1
        self.swap_function.reduce_to_single_model(model_index)
        try:
            [ee.reduce_to_single_model(model_index) for ee in self.error_emissions.values()]
        except AttributeError:
            pass

    def get_marginalised_log_likelihood(self, error_emissions_key: Any, estimation_deviations: _T, pi_vectors: _T, kwargs_for_individual_component_likelihoods: dict = {}):
        error_emissions = self.get_error_emissions(error_emissions_key)
        return super(MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel, self).get_marginalised_log_likelihood(
            estimation_deviations=estimation_deviations,
            pi_vectors=pi_vectors,
            kwargs_for_individual_component_likelihoods=kwargs_for_individual_component_likelihoods,
            _error_emissions_obj=error_emissions
        )

    def data_generation_from_component_priors(self, error_emissions_key: Any, set_size: int, vm_means: _T, component_priors: _T, kwargs_for_sample_betas: dict = {}, kwargs_for_sample_from_components: dict = {}):
        error_emissions = self.get_error_emissions(error_emissions_key)
        return super(MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel, self).data_generation_from_component_priors(
            set_size = set_size,
            vm_means = vm_means,
            component_priors = component_priors,
            kwargs_for_sample_betas = kwargs_for_sample_betas,
            kwargs_for_sample_from_components = kwargs_for_sample_from_components,
            _error_emissions_obj=error_emissions
        )

    def full_data_generation(self, error_emissions_key: Any, set_size: int, vm_means: _T, kwargs_for_generate_pi_vectors: dict = {}, kwargs_for_sample_betas: dict = {}, kwargs_for_sample_from_components: dict = {}):
        error_emissions = self.get_error_emissions(error_emissions_key)
        return super(MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel, self).full_data_generation(
            set_size = set_size,
            vm_means = vm_means,
            kwargs_for_generate_pi_vectors = kwargs_for_generate_pi_vectors,
            kwargs_for_sample_betas = kwargs_for_sample_betas,
            kwargs_for_sample_from_components = kwargs_for_sample_from_components,
            _error_emissions_obj=error_emissions
        )

    def empirical_residual_distribution_weights(self, error_emissions_key: Any, posterior_vectors: _T, estimation_deviations: _T, kwargs_for_individual_component_likelihoods: dict = {}):
        error_emissions = self.get_error_emissions(error_emissions_key)
        return super(MultipleErrorEmissionsNonParametricSwapErrorsGenerativeModel, self).empirical_residual_distribution_weights(
            posterior_vectors = posterior_vectors,
            estimation_deviations = estimation_deviations,
            kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods,
            _error_emissions_obj=error_emissions
        )

    def drop_error_emissions(self, error_emissions_key):
        self.error_emissions_keys.remove(error_emissions_key)
        self.error_emissions.pop(str(error_emissions_key))



class HierarchicalNonParametricSwapErrorsGenerativeModelWrapper(nn.Module):
    """
    We don't need a separate class for the sub-models (like for the variational) because there is no
    real interaction between top level and sublevel models! 
    Passing on function evaluations as prior means is all handled by HierarchicalWorkingMemoryFullSwapModel
    """
    def __init__(
        self, 
        primary_generative_model: NonParametricSwapErrorsGenerativeModel,
        submodel_generative_models: Dict[Any, NonParametricSwapErrorsGenerativeModel],
    ) -> None:
        super().__init__()
        
        self.num_models = primary_generative_model.num_models
        self.primary_generative_model = primary_generative_model
        self.submodel_generative_models: Dict[str, NonParametricSwapErrorsGenerativeModel] = ModuleDict(
            {str(k): v for k, v in submodel_generative_models.items()}
        )
        self.submodel_keys = list(submodel_generative_models.keys())
        assert all(self.num_models == v.num_models for v in submodel_generative_models.values())

        del self.primary_generative_model.error_emissions   # Should never be used!

    @classmethod
    def from_typical_args(
        cls, *_, 
        submodel_keys: List[Any],
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
        num_features: int,
        device = 'cuda',
        **kwargs
    ):
        """
        This is to replace the old setup_utils.py function logic!
        """
        primary_model = NonParametricSwapErrorsGenerativeModel.from_typical_args(
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
            num_features = num_features,
            device = device,
        )

        submodels = {
            smk: NonParametricSwapErrorsGenerativeModel.from_typical_args(
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
                num_features = num_features,
                device = device,
            ) for smk in submodel_keys
        }

        return cls(primary_model, submodels)
    
    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.primary_generative_model.reduce_to_single_model(model_index)
        [submodel.reduce_to_single_model(model_index) for submodel in self.submodel_generative_models.values()]

    def get_marginalised_log_likelihood(
        self, submodel_key: Any, estimation_deviations: _T, pi_vectors: _T, kwargs_for_individual_component_likelihoods: dict = {}
    ):
        """
        Pass this on to the relevant submodel, which will contain the `downstream` parameters a.k.a. \phi
        """
        return self.submodel_generative_models[str(submodel_key)].get_marginalised_log_likelihood(
            estimation_deviations=estimation_deviations, pi_vectors=pi_vectors,
            kwargs_for_individual_component_likelihoods=kwargs_for_individual_component_likelihoods
        )

    def data_generation_from_component_priors(
        self, submodel_key: Any, set_size: int, vm_means: _T, component_priors: _T, kwargs_for_sample_betas: dict = {}, kwargs_for_sample_from_components: dict = {}
    ):
        """
        Again, defering like in self.get_marginalised_log_likelihood
        """
        return self.submodel_generative_models[str(submodel_key)].data_generation_from_component_priors(
            set_size=set_size, vm_means=vm_means, component_priors=component_priors, 
            kwargs_for_sample_betas=kwargs_for_sample_betas, kwargs_for_sample_from_components=kwargs_for_sample_from_components
        )

    def full_data_generation(self, submodel_key: Any, *args, **kwargs):
        raise NotImplementedError

    def drop_submodel(self, submodel_key):
        self.submodel_keys.remove(submodel_key)
        self.submodel_generative_models.pop(str(submodel_key))




