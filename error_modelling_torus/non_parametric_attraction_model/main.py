from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

import warnings
from typing import List, Dict, Tuple, Optional, Callable

from matplotlib.pyplot import Axes

from purias_utils.error_modelling_torus.non_parametric_attraction_model.variational_approximation import SVGPApproximation, InferenceEvaluationInfo
from purias_utils.error_modelling_torus.non_parametric_attraction_model.parameters_gp_prior import AttractionErrorDistributionParametersPrior, KernelEvaluationInfo
from purias_utils.error_modelling_torus.non_parametric_attraction_model.emissions_distribution import (
    skew_wrapped_stable_distribution_likelihood,
    wrapped_normal_distribution_likelihood,
    von_mises_distribution_likelihood,
    skew_wrapped_stable_generate_samples,
    wrapped_normal_generate_samples,
    von_mises_generate_samples
)

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out




def generate_synthetic_target_responses(num_datapoints, device = 'cuda'):
    "Locations of synthetic inputs (target values)"
    synthetic_target_responses = torch.rand(num_datapoints) * 2 * torch.pi - torch.pi
    synthetic_target_responses = torch.sort(synthetic_target_responses)[0].to(device)
    return synthetic_target_responses


def generate_sample_dataset_from_inference_eval_info(synthetic_eval_info: Dict[str, InferenceEvaluationInfo], sampling_function: Callable, model: WrappedStableAttractionErrorModel, model_idx: int = 0) -> _T:
    # Generate samples from these synthetic functions -> [1, num_function_sampless, num_datapoints]
    all_synthetic_function_samples = {k: v.take_samples(v.num_data) for k, v in synthetic_eval_info.items()}

    # Extract one sample, for ancestor sampling, and transform to correct range
    synthetic_function_ancestor_samples_unlinked = {k: v.prepare_samples_for_ancestor_sampling(all_synthetic_function_samples[k]) for k, v in synthetic_eval_info.items()}
    synthetic_function_ancestor_samples = {k: model.get_link_function(k)(v) for k, v in synthetic_function_ancestor_samples_unlinked.items()}

    # Sample from resulting wrapped stable distribution
    synthetic_errors = sampling_function(
        **{f'{k}_parameter_samples': v for k, v in synthetic_function_ancestor_samples.items()}
    )
    synthetic_errors = synthetic_errors[model_idx]

    return synthetic_errors



class WrappedStableAttractionErrorModel(nn.Module):

    wss_param_names: List[str] = ['alpha', 'beta', 'gamma', 'delta']
    likelihood_function = staticmethod(skew_wrapped_stable_distribution_likelihood)
    sampling_function = staticmethod(skew_wrapped_stable_generate_samples)

    def __init__(
        self,
        alpha_prior: AttractionErrorDistributionParametersPrior,
        beta_prior: AttractionErrorDistributionParametersPrior,
        gamma_prior: AttractionErrorDistributionParametersPrior,
        delta_prior: AttractionErrorDistributionParametersPrior,
        alpha_variational_approx: SVGPApproximation,
        beta_variational_approx: SVGPApproximation,
        gamma_variational_approx: SVGPApproximation,
        delta_variational_approx: SVGPApproximation,
    ) -> None:
        super().__init__()
        
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.gamma_prior = gamma_prior
        self.delta_prior = delta_prior
        self.alpha_variational_approx = alpha_variational_approx
        self.beta_variational_approx = beta_variational_approx
        self.gamma_variational_approx = gamma_variational_approx
        self.delta_variational_approx = delta_variational_approx
        
    @staticmethod
    def alpha_link_function(alpha_parameter_samples_real: _T) -> _T:
        offset = 0.8
        return (2. - offset) * alpha_parameter_samples_real.sigmoid() + offset
    
    @staticmethod
    def inverse_alpha_link_function(alpha_parameter_samples: _T) -> _T:
        offset = 0.8
        return torch.logit((alpha_parameter_samples - 0.8) / (2. - offset))

    @staticmethod
    def beta_link_function(beta_parameter_samples_real: _T) -> _T:
        return beta_parameter_samples_real.sigmoid() * 2.0 - 1.0

    @staticmethod
    def inverse_beta_link_function(beta_parameter_samples: _T) -> _T:
        return torch.logit((beta_parameter_samples + 1.0) / 2.0)

    @staticmethod
    def gamma_link_function(gamma_parameter_samples_real: _T) -> _T:
        offset = 0.1
        return torch.nn.functional.softplus(gamma_parameter_samples_real) + offset

    @staticmethod
    def inverse_gamma_link_function(gamma_parameter_samples: _T) -> _T:
        offset = 0.1
        return inv_softplus(gamma_parameter_samples - offset)

    @staticmethod
    def delta_link_function(delta_parameter_samples_real: _T) -> _T:
        return rectify_angles(delta_parameter_samples_real)
    
    def inverse_delta_link_function(self, delta_parameter_samples: _T) -> _T:
        assert self.verify_angle(delta_parameter_samples)
        return delta_parameter_samples

    def get_link_function(self, parameter_name) -> _T:
        return getattr(self, f'{parameter_name}_link_function')

    def get_inverse_link_function(self, parameter_name) -> _T:
        return getattr(self, f'inverse_{parameter_name}_link_function')

    @staticmethod
    def verify_angle(tensor: _T) -> _T:
        return (tensor.max() <= torch.pi) and (tensor.min() >= -torch.pi)

    def get_variational_posteriors(
        self,
        target_angles: _T,
        max_variational_batch_size: int = 0
    ) -> Tuple[Dict[str, List[KernelEvaluationInfo]], Dict[str, List[InferenceEvaluationInfo]], List[slice]]:
        """
        target_angles of shape [B] - just batch size - input
        estimation_error of shape [B] - just batch size - output
        max_variational_batch_size: int - to avoid input collapse we break into minibatches
        """

        if max_variational_batch_size < 1:
            minibatched_target_angles = [target_angles]
            minibatch_slices = slice(None, None)
        else:
            batch_size, = target_angles.shape
            num_minibatches = (batch_size // max_variational_batch_size) + (1 if batch_size % max_variational_batch_size else 0)
            minibatch_slices = [slice(j*max_variational_batch_size, (j+1)*max_variational_batch_size) for j in range(num_minibatches)]
            minibatched_target_angles = [target_angles[minibatch_slices[j]] for j in range(num_minibatches)]

        all_priors: Dict[str, AttractionErrorDistributionParametersPrior] = {
            wss_param: getattr(self, f'{wss_param}_prior') for wss_param in self.wss_param_names
        }
        all_svgp_approx: Dict[str, SVGPApproximation] = {
            wss_param: getattr(self, f'{wss_param}_variational_approx') for wss_param in self.wss_param_names
        }

        all_prior_kernel_infos = {
            f"{wss_param}_prior_kernel_infos": all_priors[wss_param].generate_kernel_info(
                minibatched_target_angles, all_svgp_approx[wss_param].Z
            )
            for wss_param in self.wss_param_names
        }

        all_variational_inference_infos = {
            f'{wss_param}_variational_inference_infos': all_svgp_approx[wss_param].variational_gp_inference_minibatched(
                all_prior_kernel_infos[f"{wss_param}_prior_kernel_infos"]
            )
            for wss_param in self.wss_param_names
        }

        return (
            all_prior_kernel_infos,
            all_variational_inference_infos,
            minibatch_slices
        )

    def inference_on_grid(self, num_grid_points: int = 100, max_variational_batch_size: int = 0, device: str = 'cuda'):
        grid_data = torch.linspace(-torch.pi, +torch.pi, num_grid_points + 1)[:-1].to(device)
        return grid_data, *self.get_variational_posteriors(grid_data, max_variational_batch_size)

    def plot_inducing_points_to_axes(
        param_name: str,
        axes: Axes,
        link_function: bool
    ):
        axes.scatter(
            self.
        )

    def emissions_likelihood(
        self, estimation_error: _T, **parameter_samples_real: _T
    ):
        """
        estimation_error of shape [B] - just batch size
        ._parameter_samples_real of shape [Q, I] - Q models trained, I samples from each
        
        output of shape [Q, I, B] - one likelihood for each item-model pair
        """
        likelihood_kwargs = {
            f'{wss_param}_parameter_samples': self.get_link_function(wss_param)(parameter_samples_real[f'{wss_param}_parameter_samples_real'])
            for wss_param in self.wss_param_names
        }

        return self.likelihood_function(estimation_error = estimation_error, **likelihood_kwargs)

    def get_function_marginalised_loglikelihood(
        self,
        target_angles: _T,
        estimation_error: _T,
        num_prior_samples: int = 64,
        max_variational_batch_size: int = 0,
        evaluate_kl: bool = False,
        *_,
        override_variational_posterior_info: Optional[Dict[str, List[InferenceEvaluationInfo]]] = None
    ):
        """
        target_angles of shape [B] - just batch size - input
        estimation_error of shape [B] - just batch size - output
        max_variational_batch_size: int - to avoid input collapse we break into minibatches

        1. Generate variational posteriors from functions
        2. Take samples from variational posteriors
        3. Evaluate log(!!)likelihood for each of these samples
        4. Integrate <log p(y | f, x)>_{f ~ q(f)}
        """

        B, = estimation_error.shape
        assert tuple(target_angles.shape) == (B,)

        if override_variational_posterior_info is None:
            # Gaussian moments over functions
            prior_kernel_infos, variational_posterior_info, minibatch_slices = self.get_variational_posteriors(
                target_angles, max_variational_batch_size
            )
        else:
            assert all(len(v) == 1 for v in override_variational_posterior_info.values())
            assert all(v[0].num_data == B for v in override_variational_posterior_info.values())
            variational_posterior_info = override_variational_posterior_info
            prior_kernel_infos = None
            minibatch_slices = [slice(None, None)]
            assert max_variational_batch_size < 1
            assert not evaluate_kl

        # Samples - join up minibatches!
        all_likelihoods = []
        for nm, mbs in enumerate(minibatch_slices):
            samples_from_variational_posteriors = {
                f'{wss_param}_parameter_samples_real': variational_posterior_info[f'{wss_param}_variational_inference_infos'][nm].take_samples(num_prior_samples) 
                for wss_param in self.wss_param_names
            }
            new_likelihoods = self.emissions_likelihood(estimation_error[mbs], **samples_from_variational_posteriors)   # [Q, I, B]
            all_likelihoods.append(new_likelihoods)

        likelihoods = torch.concat(all_likelihoods, -1)
        marginalised_loglikelihood = (likelihoods).log().mean(1)
        if marginalised_loglikelihood.isnan().any() or marginalised_loglikelihood.isinf().any():
            raise ValueError('marginalised_loglikelihood has inf or NaN')

        ret = {
            "prior_kernel_infos": prior_kernel_infos,
            "variational_posterior_info": variational_posterior_info,
            "samples_from_variational_posteriors": samples_from_variational_posteriors,
            "marginalised_loglikelihood": marginalised_loglikelihood,
        }

        if evaluate_kl:

            kl_terms = {}

            warnings.warn('total_elbo calculation in AttractionErrorModel.get_function_marginalised_loglikelihood assumes all data is passed in already')
            total_elbo = marginalised_loglikelihood.sum(-1)

            for wss_param in self.wss_param_names:
                relevant_kernel_info = prior_kernel_infos[f"{wss_param}_prior_kernel_infos"][0]
                relevant_svgp_object: SVGPApproximation = getattr(self, f"{wss_param}_variational_approx")
                
                kl_terms[wss_param] = relevant_svgp_object.kl_loss(K_uu = relevant_kernel_info.K_uu, K_uu_inv = relevant_kernel_info.K_uu_inv)

                total_elbo -= kl_terms[wss_param]   # No beta needed, if all data is passed here!!
            
            ret['kl'] = kl_terms
            ret['total_elbo'] = total_elbo

        return ret

    @torch.no_grad
    def generate_sample_data(self, target_angles: _T, max_variational_batch_size: int = 0, model_idx: int = 0):
        """
        Sample y ~ p(y | x, D) = \int p(y | f, x) p(f | D)
        Approximate this by y ~ \int p(y | f, x) q(f)
        Achieve this by doing ancestor sampling of joint q(f) * p(y | f, x)

        target_angles of shape [B] - just batch size - input

        output of shape [Q, I, B]
            Q = num_models, B = batch size as above

        1. Generate variational posterior for target angles
        2. Take (ancestor) samples from this
        """
        prior_kernel_infos, variational_posterior_info, minibatch_slices =\
            self.get_variational_posteriors(target_angles = target_angles, max_variational_batch_size = max_variational_batch_size)

        all_synth_data = []
        for nm in range(len(minibatch_slices)):
            new_synth_data = generate_sample_dataset_from_inference_eval_info(
                synthetic_eval_info={
                    wss_param: variational_posterior_info[f'{wss_param}_variational_inference_infos'][nm]
                    for wss_param in self.wss_param_names
                },
                sampling_function = self.sampling_function,
                model = self,
            )
            all_synth_data.append(new_synth_data)

        synth_data = torch.concat(all_synth_data, model_idx)

        return synth_data



class WrappedNormalAttractionErrorModel(WrappedStableAttractionErrorModel):

    wss_param_names: List[str] = ['mu', 'sigma']
    likelihood_function = staticmethod(wrapped_normal_distribution_likelihood)
    sampling_function = staticmethod(wrapped_normal_generate_samples)

    def __init__(
        self,
        mu_prior: AttractionErrorDistributionParametersPrior,
        sigma_prior: AttractionErrorDistributionParametersPrior,
        mu_variational_approx: SVGPApproximation,
        sigma_variational_approx: SVGPApproximation,
    ) -> None:
        super(WrappedStableAttractionErrorModel, self).__init__()
        
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.mu_variational_approx = mu_variational_approx
        self.sigma_variational_approx = sigma_variational_approx

    @staticmethod
    def mu_link_function(mu_parameter_samples_real: _T) -> _T:
        return rectify_angles(mu_parameter_samples_real)
    
    def inverse_mu_link_function(self, mu_parameter_samples: _T) -> _T:
        assert self.verify_angle(mu_parameter_samples)
        return mu_parameter_samples

    @staticmethod
    def sigma_link_function(sigma_parameter_samples_real: _T) -> _T:
        return torch.nn.functional.softplus(sigma_parameter_samples_real)

    @staticmethod
    def inverse_sigma_link_function(sigma_parameter_samples: _T) -> _T:
        return inv_softplus(sigma_parameter_samples)



class VonMisesAttractionErrorModel(WrappedStableAttractionErrorModel):

    wss_param_names: List[str] = ['mu', 'kappa']
    likelihood_function = staticmethod(von_mises_distribution_likelihood)
    sampling_function = staticmethod(von_mises_generate_samples)

    def __init__(
        self,
        mu_prior: AttractionErrorDistributionParametersPrior,
        kappa_prior: AttractionErrorDistributionParametersPrior,
        mu_variational_approx: SVGPApproximation,
        kappa_variational_approx: SVGPApproximation,
    ) -> None:
        super(WrappedStableAttractionErrorModel, self).__init__()
        
        self.mu_prior = mu_prior
        self.kappa_prior = kappa_prior
        self.mu_variational_approx = mu_variational_approx
        self.kappa_variational_approx = kappa_variational_approx

    @staticmethod
    def mu_link_function(mu_parameter_samples_real: _T) -> _T:
        return rectify_angles(mu_parameter_samples_real)
    
    def inverse_mu_link_function(self, mu_parameter_samples: _T) -> _T:
        assert self.verify_angle(mu_parameter_samples)
        return mu_parameter_samples

    @staticmethod
    def kappa_link_function(kappa_parameter_samples_real: _T) -> _T:
        offset = 10.0
        scale = 6.0
        return torch.nn.functional.softplus(kappa_parameter_samples_real * scale + offset)

    @staticmethod
    def inverse_kappa_link_function(kappa_parameter_samples: _T) -> _T:
        offset = 10.0
        scale = 6.0
        return (inv_softplus(kappa_parameter_samples) - offset) / scale
