import torch
from torch import Tensor as _T

from typing import Type

from purias_utils.error_modelling_torus.non_parametric_attraction_model.main import WrappedStableAttractionErrorModel, VonMisesAttractionErrorModel
from purias_utils.error_modelling_torus.non_parametric_attraction_model.variational_approximation import InferenceEvaluationInfo, NoInputInferenceEvaluationInfo



def wrap_up_zero_variance_synthetic_eval_info(model: WrappedStableAttractionErrorModel, inference_info_class: Type[InferenceEvaluationInfo], device: str, **synthetic_params):
    
    num_datapoints = list(synthetic_params.values())[0].shape[-1]
    zero_variance_sigma = torch.zeros(1, num_datapoints, num_datapoints).to(device)
    zero_variance_sigma_chol = torch.zeros(1, num_datapoints, num_datapoints).to(device)

    synthetic_eval_info = {
        param_name: inference_info_class(
            mu = model.get_inverse_link_function(param_name)(synthetic_param),
            sigma = zero_variance_sigma, sigma_chol = zero_variance_sigma_chol
        )
        for param_name, synthetic_param in synthetic_params.items()
    }

    return synthetic_eval_info



def no_input_von_mises_generate_synthetic_eval_info(synthetic_target_responses: _T, model: VonMisesAttractionErrorModel, device = 'cuda'):

    assert len(synthetic_target_responses.shape) == 1

    # synthetic_target_responses ignored
    target_mu = 0.4 * torch.ones_like(synthetic_target_responses.unsqueeze(0))
    target_kappa = 8.0 * torch.ones_like(synthetic_target_responses.unsqueeze(0))

    return wrap_up_zero_variance_synthetic_eval_info(
        model = model, inference_info_class = NoInputInferenceEvaluationInfo, device = device,
        mu = target_mu, kappa = target_kappa
    )



def no_input_prior_wrapped_stable_generate_synthetic_eval_info(synthetic_target_responses: _T, model: WrappedStableAttractionErrorModel, device = 'cuda'):

    assert len(synthetic_target_responses.shape) == 1

    # synthetic_target_responses ignored
    target_alpha = 1.9 * torch.ones_like(synthetic_target_responses.unsqueeze(0))
    target_beta = 0.7 * torch.ones_like(synthetic_target_responses.unsqueeze(0))
    target_gamma = 0.35 + 0.1 * torch.ones_like(synthetic_target_responses.unsqueeze(0))
    target_delta = 0.5 * torch.ones_like(synthetic_target_responses.unsqueeze(0))

    return wrap_up_zero_variance_synthetic_eval_info(
        model = model, inference_info_class = NoInputInferenceEvaluationInfo, device = device,
        alpha = target_alpha, beta = target_beta, gamma = target_gamma, delta = target_delta
    )



def gp_prior_wrapped_stable_generate_synthetic_eval_info(synthetic_target_responses: _T, model: WrappedStableAttractionErrorModel, device = 'cuda'):

    assert len(synthetic_target_responses.shape) == 1

    # Synthetic information for each one parameter 
    # For now, this is just the mean varying!
    target_alpha = 1.9 * torch.ones_like(synthetic_target_responses.unsqueeze(0))
    target_beta = 0.7 * torch.cos(2 * synthetic_target_responses.unsqueeze(0))
    target_gamma = 0.35 + 0.1 * torch.sin(2 * synthetic_target_responses.unsqueeze(0))
    target_delta = 0.5 * torch.sin(synthetic_target_responses.unsqueeze(0))

    return wrap_up_zero_variance_synthetic_eval_info(
        model = model, inference_info_class = InferenceEvaluationInfo, device = device,
        alpha = target_alpha, beta = target_beta, gamma = target_gamma, delta = target_delta
    )


def gp_prior_von_mises_generate_synthetic_eval_info(synthetic_target_responses: _T, model: WrappedStableAttractionErrorModel, device = 'cuda'):

    assert len(synthetic_target_responses.shape) == 1

    # target_mu = 0.5 * torch.ones_like(synthetic_target_responses.unsqueeze(0))
    target_mu = 0.5 * torch.sin(synthetic_target_responses.unsqueeze(0))
    target_kappa = 2.0 + 3.0 * (2.0 * synthetic_target_responses).unsqueeze(0).cos().exp()
    # target_kappa = 8.0 + 5.0 * torch.cos(synthetic_target_responses.unsqueeze(0))
    # target_kappa = 10.0 * torch.ones_like(synthetic_target_responses.unsqueeze(0))

    return wrap_up_zero_variance_synthetic_eval_info(
        model = model, inference_info_class = InferenceEvaluationInfo, device = device,
        mu = target_mu, kappa = target_kappa
    )
