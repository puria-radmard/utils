import numpy as np
import torch
from tqdm import tqdm
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

def kurtosis_from_angles(angles, weights):

    if isinstance(weights, torch.Tensor):
        assert torch.isclose(weights.mean(), torch.tensor(1.0).to(weights.dtype))

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
