import torch
from torch import Tensor as _T

import warnings
from tqdm import tqdm

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.util.error_modelling import zero_mean_von_mises_log_prob_with_kappa


DEBUGGING = False


def skew_wrapped_stable_distribution_likelihood(
    estimation_error: _T,
    alpha_parameter_samples: _T,
    beta_parameter_samples: _T,
    gamma_parameter_samples: _T,
    delta_parameter_samples: _T,
    p_cut_off: int = 1000,
    throw_error: bool = True,
    negative_threshold = 0.005,
    boundary_clip = 1e-9
):
    """
    estimation_error of shape [B] - just batch size
    ._parameter_samples of shape [Q, I, B] - Q models trained, I samples from function, B points at which it is evaluated
        i.e. function_evals[q,i,:] gives smooth function, 
                function_evals[q,:,b] gives you all the function evaluations sampled at one point

    Arthur Pewsey 2008
    
    output of shape [Q, I, B] - one likelihood for each item-model pair
    """
    Q, I, B = alpha_parameter_samples.shape
    assert (B,) == tuple(estimation_error.shape)

    estimation_error = estimation_error[None,None].repeat(Q, I, 1)   # [Q, I, B]
    result = torch.ones(Q, I, B).to(estimation_error.device)                       # [Q, I, B]
    
    alpha_is_one_mask = (alpha_parameter_samples == 1.0)

    tan_alpha_times_half_pi = (torch.pi * alpha_parameter_samples[~alpha_is_one_mask] / 2).tan()

    for p in tqdm(range(1, p_cut_off + 1), disable= not DEBUGGING):
        
        pgamma = gamma_parameter_samples * p
        pgamma_to_the_alpha = torch.pow(pgamma, alpha_parameter_samples)

        mu_p = delta_parameter_samples * p
        
        mu_p[alpha_is_one_mask] = mu_p[alpha_is_one_mask] - rectify_angles(
            beta_parameter_samples[alpha_is_one_mask] * 2 * pgamma[alpha_is_one_mask] * pgamma[alpha_is_one_mask].log() / torch.pi
        )
        mu_p[~alpha_is_one_mask] = mu_p[~alpha_is_one_mask] + (
            tan_alpha_times_half_pi *
            beta_parameter_samples[~alpha_is_one_mask] * 
            rectify_angles(
                pgamma_to_the_alpha[~alpha_is_one_mask] - pgamma[~alpha_is_one_mask]
            )
        )

        rho_p = torch.exp(- pgamma_to_the_alpha)

        pth_term = 2 * rho_p * (p * estimation_error - mu_p).cos()   # [Q, I, B]
        result = result + pth_term

    # import pdb; pdb.set_trace()
    # from matplotlib import pyplot as plt 
    # fig, axes = plt.subplots(2)
    # axes[0].scatter(estimation_error.cpu().numpy(), result[0,0].detach().cpu().numpy())
    # axes[1].plot(gamma_parameter_samples[0,:,0].detach().cpu())
    # fig.savefig('asdf')

    # [result == result.min()]
    result = result / (2 * torch.pi)

    if result.min() < 0.0:
        alpha = alpha_parameter_samples[result == result.min()].item()
        beta = beta_parameter_samples[result == result.min()].item()
        gamma = gamma_parameter_samples[result == result.min()].item()
        delta = delta_parameter_samples[result == result.min()].item()
        message = (
            f'''
            skew_wrapped_stable_distribution_likelihood found a sub-zero likelihood: {result.min().item()}
            
            Offending parameters were:
            α = {alpha}
            β = {beta}
            γ = {gamma}
            δ = {delta}
            '''
        )

        if result.min() < - negative_threshold:
            if throw_error:
                raise ValueError(message)
            else:
                warnings.warn(message)
        
    result = result.clip(min=boundary_clip)
    
    return result


def skew_wrapped_stable_generate_samples(
    alpha_parameter_samples: _T,
    beta_parameter_samples: _T,
    gamma_parameter_samples: _T,
    delta_parameter_samples: _T,
):
    """
    Method taken from Wikipedia!

    ._parameter_samples of shape [Q, I, B] - Q models trained, I samples from each prior, B input points at which the functions are evaluated
        see InferenceEvaluationInfo.take_samples
        OR of shape [Q, B] when doing ancestor sampling
    
    output of shape [Q, I, B] - one sample of output data from each (function sample)-(input location) combination
        OR of shape [Q, K], K == B - one sample for each ancestor sample from the functions
    """
    assert all(
        param.shape == alpha_parameter_samples.shape for 
        param in [beta_parameter_samples, gamma_parameter_samples, delta_parameter_samples]
    )

    half_pi = 0.5 * torch.pi

    zeta = - beta_parameter_samples * (half_pi * alpha_parameter_samples).tan()
    xi: _T = torch.ones_like(beta_parameter_samples) * half_pi
    alpha_is_one_mask = (alpha_parameter_samples == 1.0)
    xi[~alpha_is_one_mask] = (-zeta[~alpha_is_one_mask]).arctan() / alpha_parameter_samples[~alpha_is_one_mask]

    u_samples = (torch.rand_like(alpha_parameter_samples) * torch.pi) - (torch.pi / 2)
    w_samples = torch.ones_like(alpha_parameter_samples)
    w_samples.exponential_(lambd=1)

    alpha_one_tan_term = (
        (half_pi) + 
        (beta_parameter_samples[alpha_is_one_mask] * u_samples[alpha_is_one_mask])
    ) * u_samples[alpha_is_one_mask].tan()

    alpha_one_log_term = beta_parameter_samples[alpha_is_one_mask] * (
        (half_pi * w_samples[alpha_is_one_mask] * u_samples[alpha_is_one_mask].cos()) / 
        (half_pi + u_samples[alpha_is_one_mask] * beta_parameter_samples[alpha_is_one_mask])
    ).log()

    alpha_u_plus_xi_alpha_not_one: _T = alpha_parameter_samples[~alpha_is_one_mask] * (u_samples[~alpha_is_one_mask] + xi[~alpha_is_one_mask])
    
    alpha_not_one_sin_term: _T = (
        alpha_u_plus_xi_alpha_not_one.sin()
    ) / torch.pow(
        u_samples[~alpha_is_one_mask].cos(),
        1.0 / alpha_parameter_samples[~alpha_is_one_mask]
    )

    alpha_not_one_cos_term: _T = torch.pow(
        (u_samples[~alpha_is_one_mask] - alpha_u_plus_xi_alpha_not_one).cos() / w_samples[~alpha_is_one_mask],
        (1.0 - alpha_parameter_samples[~alpha_is_one_mask]) / (alpha_parameter_samples[~alpha_is_one_mask])
    )

    alpha_not_one_zi_term = torch.pow(
        1 + torch.pow(zeta[~alpha_is_one_mask], 2.0),
        0.5 / alpha_parameter_samples[~alpha_is_one_mask]
    )

    x_samples = torch.zeros_like(alpha_parameter_samples)
    x_samples[alpha_is_one_mask] = (alpha_one_tan_term - alpha_one_log_term) / xi[alpha_is_one_mask]
    x_samples[~alpha_is_one_mask] = alpha_not_one_sin_term * alpha_not_one_cos_term * alpha_not_one_zi_term

    x_samples = x_samples * gamma_parameter_samples + delta_parameter_samples
    x_samples[~alpha_is_one_mask] = x_samples[~alpha_is_one_mask] + zeta[~alpha_is_one_mask] * gamma_parameter_samples[~alpha_is_one_mask]

    return rectify_angles(x_samples)


def wrapped_normal_distribution_likelihood(
    estimation_error: _T,
    mu_parameter_samples: _T,
    sigma_parameter_samples: _T,
):
    Q, I, B = mu_parameter_samples.shape
    assert (B,) == tuple(estimation_error.shape)

    estimation_error = estimation_error[None,None].repeat(Q, I, 1)   # [Q, I, B]
    raise NotImplementedError('Need to evaluate wrapped_normal_distribution_likelihood')


def wrapped_normal_generate_samples(
    mu_parameter_samples: _T,
    sigma_parameter_samples: _T,
):
    raise NotImplementedError


def von_mises_distribution_likelihood(
    estimation_error: _T,
    mu_parameter_samples: _T,
    kappa_parameter_samples: _T,
):
    """
    estimation_error of shape [B] - just batch size
    ._parameter_samples of shape [Q, I, B] - Q models trained, I samples from function, B points at which it is evaluated
        i.e. function_evals[q,i,:] gives smooth function, 
                function_evals[q,:,b] gives you all the function evaluations sampled at one point

    output of shape [Q, I, B] - one likelihood for each item-model pair
    """
    Q, I, B = mu_parameter_samples.shape
    assert (B,) == tuple(estimation_error.shape)

    estimation_error = estimation_error[None,None].repeat(Q, I, 1)   # [Q, I, B]

    vm = torch.distributions.VonMises(loc = mu_parameter_samples, concentration = kappa_parameter_samples)
    log_prob = vm.log_prob(estimation_error)

    return log_prob.exp()


def von_mises_generate_samples(
    mu_parameter_samples: _T,
    kappa_parameter_samples: _T,
):
    """
    ._parameter_samples of shape [Q, I, B] - Q models trained, I samples from each prior, B input points at which the functions are evaluated
        see InferenceEvaluationInfo.take_samples
        OR of shape [Q, B] when doing ancestor sampling
    
    output of shape [Q, I, B] - one sample of output data from each (function sample)-(input location) combination
        OR of shape [Q, K], K == B - one sample for each ancestor sample from the functions
    """
    assert all(
        param.shape == kappa_parameter_samples.shape for 
        param in [mu_parameter_samples, kappa_parameter_samples]
    )

    vm = torch.distributions.VonMises(loc = mu_parameter_samples, concentration = kappa_parameter_samples)
    samples = vm.sample(())

    return samples

