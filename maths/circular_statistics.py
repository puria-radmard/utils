import numpy as np
import torch
from torch import Tensor as _T

from tqdm import tqdm
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


def mean_resultant_length_from_angles(angles: _T, weights: _T):
    assert torch.isclose(weights.mean(), torch.tensor(1.0).to(weights.dtype))
    angles, weights = angles.cpu().numpy(), weights.cpu().numpy()

    complex_vectors_1 = np.exp(1j * angles) * weights
    average_vector_1 = complex_vectors_1.mean()
    R_bar_1 = np.abs(average_vector_1)
    return R_bar_1


def kurtosis_from_angles(angles: _T, weights: _T):
    assert torch.isclose(weights.mean(), torch.tensor(1.0).to(weights.dtype))
    angles, weights = angles.cpu().numpy(), weights.cpu().numpy()

    complex_vectors_1 = np.exp(1j * angles) * weights
    complex_vectors_2 = np.exp(2j * angles) * weights
    average_vector_1 = complex_vectors_1.mean()
    average_vector_2 = complex_vectors_2.mean()
    
    R_bar_1 = np.abs(average_vector_1)
    R_bar_2 = np.abs(average_vector_2)
    theta_bar_1 = np.angle(average_vector_1)
    theta_bar_2 = np.angle(average_vector_2)

    k_numerator = R_bar_2 * np.cos(theta_bar_2 - 2 * theta_bar_1) - np.power(R_bar_1, 4.0)
    k_denominator = np.power(1 - R_bar_1, 2.0)

    return k_numerator / k_denominator



def wrapped_stable_kurtosis(alpha, gamma):
    gamma_to_the_alpha = np.power(gamma, alpha)
    numerator = np.exp(-gamma_to_the_alpha * np.power(2.0, alpha)) - np.exp(-4 * gamma_to_the_alpha)
    denominator = np.power(1 - np.exp(-gamma_to_the_alpha), 2.0)
    return numerator / denominator


def wrapped_stable_mean_resultant_length(alpha, gamma):
    gamma_to_the_alpha = np.power(gamma, alpha)
    return np.exp(-gamma_to_the_alpha)


def symmetric_zero_mean_wrapped_stable(theta_axis, alpha, gamma, p_cut_off = 100):

    result = torch.ones_like(theta_axis) / (2 * torch.pi)
    
    gamma_to_the_alpha = torch.pow(gamma, alpha)
    rho_p = torch.ones_like(gamma_to_the_alpha) # rho_0

    for p in range(1, p_cut_off + 1):

        p_minus_1_tensor = torch.tensor(p - 1.0)
        log_r_p_minus_1 = gamma_to_the_alpha * (torch.pow(p_minus_1_tensor, alpha) - torch.pow(p_minus_1_tensor + 1.0, alpha))
        rho_p = rho_p * log_r_p_minus_1.exp()

        pth_term = (rho_p * (p * theta_axis).cos()) / torch.pi

        result = result + pth_term

    return result


def sample_from_wrapped_stable(alpha, gamma, sample_shape):
    u_samples = (torch.rand(sample_shape) * torch.pi) - (torch.pi / 2)
    w_samples = torch.ones(sample_shape)
    w_samples.exponential_(lambd=1)
    sin_term = (alpha * u_samples).sin() / torch.pow(u_samples.cos(), 1. / alpha)
    exp_term = torch.pow((u_samples * (1. - alpha)).cos() / w_samples, ((1. - alpha) / alpha))
    x_samples = sin_term * exp_term
    return rectify_angles(gamma * x_samples)


def fit_symmetric_zero_mean_wrapped_stable_to_samples(alpha_0, gamma_0, samples, weights = 1.0, num_iter = 300, lr = 0.1):

    all_losses = []
    all_alphas = []
    all_gammas = []

    params = torch.tensor([torch.atanh(alpha_0 - 1.0), torch.log(gamma_0)])
    params = torch.nn.parameter.Parameter(params, requires_grad=True)
    opt = torch.optim.Adam([params], lr = lr)

    for t in tqdm(range(num_iter)):
        
        opt.zero_grad()
        alpha = params[0].tanh() + 1.0
        gamma = params[1].exp()
        nllh = - (symmetric_zero_mean_wrapped_stable(samples, alpha, gamma).log() * weights).sum()
        nllh.backward()
        opt.step()
        
        all_losses.append(nllh.item())
        all_alphas.append(alpha.item())
        all_gammas.append(gamma.item())
    
    return all_losses, all_alphas, all_gammas, params.detach()


def find_wrapped_stable_parameters_contour_direction(gradient_direction, increasing_alpha):
    "gradient_direction of shape [batch, 2]"
    canvas = np.zeros_like(gradient_direction)
    canvas[:, 0] = 1.0 if increasing_alpha else -1.0
    canvas[:, 1] = - canvas[:, 0] * gradient_direction[:, 0] / gradient_direction[:, 1]
    return canvas / np.square(canvas).sum(-1, keepdims=True)


def wrapped_stable_mean_resultant_length_grad(alpha, gamma):
    "assert alpha.shape == gamma.shape and len(alpha.shape) == 1 done upstream!"
    f = -np.power(gamma, alpha)
    wrt_alpha = f * np.log(gamma) * np.exp(f)
    wrt_gamma = alpha * np.exp(f) * f / gamma
    grads = np.stack([wrt_alpha, wrt_gamma], axis = -1) # 2d now
    norm_grad = grads / np.square(grads).sum(-1, keepdims=True)
    return norm_grad

def wrapped_stable_kurtosis_grad(alpha, gamma):
    "assert alpha.shape == gamma.shape and len(alpha.shape) == 1 done upstream!"
    f = -np.power(gamma, alpha)
    exp_f = np.exp(f)
    f_reflx = 1. - exp_f
    shared_denominator = np.power(f_reflx, 3.)
    two_to_the_alpha = np.power(2., alpha)
    exp_g = np.exp(two_to_the_alpha * f)
    exp_4f = np.exp(4. * f)

    wrt_alpha_numerator_first_term = f_reflx * f * ((two_to_the_alpha * np.log(2 * gamma) * exp_g) - (4 * np.log(gamma) * exp_4f))
    wrt_alpha_numerator_second_term = 2. * exp_f * f * np.log(gamma) * (exp_g - exp_4f)
    
    wrt_gamma_numerator_first_term = f_reflx * f * ((two_to_the_alpha * exp_g) - (4 * exp_4f)) * alpha / gamma
    wrt_gamma_numerator_second_term = 2 * exp_f * f * alpha * (exp_g - exp_4f) / gamma

    wrt_alpha = (wrt_alpha_numerator_first_term + wrt_alpha_numerator_second_term) / shared_denominator
    wrt_gamma = (wrt_gamma_numerator_first_term + wrt_gamma_numerator_second_term) / shared_denominator
    grads = np.stack([wrt_alpha, wrt_gamma], axis = -1) # 2d now
    norm_grad = grads / np.square(grads).sum(-1, keepdims=True)
    return norm_grad

def step_along_mean_resultant_length_contour(alpha, gamma, step_size = 0.0001, increasing_alpha=True):
    assert alpha.shape == gamma.shape and len(alpha.shape) == 1
    grad = wrapped_stable_mean_resultant_length_grad(alpha, gamma)
    contour_direction = find_wrapped_stable_parameters_contour_direction(grad, increasing_alpha)
    new_alpha = alpha + step_size * contour_direction[:,0]
    new_gamma = gamma + step_size * contour_direction[:,1]
    return new_alpha, new_gamma


def step_along_kurtosis_contour(alpha, gamma, step_size = 0.0001, increasing_alpha=True):
    assert alpha.shape == gamma.shape and len(alpha.shape) == 1
    grad = wrapped_stable_kurtosis_grad(alpha, gamma)
    contour_direction = find_wrapped_stable_parameters_contour_direction(grad, increasing_alpha)
    new_alpha = alpha + step_size * contour_direction[:,0]
    new_gamma = gamma + step_size * contour_direction[:,1]
    return new_alpha, new_gamma

