import torch
import torch.nn.functional as F
from math import log
from torch import nn
from torch import Tensor as _T
from torch.nn.parameter import Parameter
from maths.gsm import gabor


SCALE_LIMITS = (0.2, 1.0)
SCALE_TO_SIGMA = 5
SCALE_TO_K = 0.5
GAMMA = 1

def gsm_forward_pass(filter_set: _T, latent_values: _T, noise_term: _T, contrast) -> _T:
    return contrast * (filter_set @ latent_values) + noise_term


def gabor(theta: _T, scale: _T, _x: _T, _y: _T, square_size: int, device='cuda') -> _T:
    """
    "Forward-pass" of a set of filter parameters (theta_A) to the actual filter set A
    """

    theta = torch.clip(theta, -torch.pi, torch.pi)
    scale = torch.clip(scale, *SCALE_LIMITS)
    _x = torch.clip(_x, -square_size / 2, square_size / 2)
    _y = torch.clip(_y, -square_size / 2, square_size / 2)

    num_filters = _y.shape[0]
    sigma = SCALE_TO_SIGMA * scale
    k = SCALE_TO_K / scale
    gamma = GAMMA

    # Both are [square_size, square_size]
    canvas_x, canvas_y = torch.meshgrid(
        torch.arange(-square_size / 2, square_size / 2).to(device),
        torch.arange(-square_size / 2, square_size / 2).to(device),
    )

    # Both are [square_size, square_size, num_filters]
    canvas_x_offset = canvas_x.unsqueeze(-1).repeat(1, 1, num_filters) - _x
    canvas_y_offset = canvas_y.unsqueeze(-1).repeat(1, 1, num_filters) - _y

    # Both are [square_size, square_size, num_filters]
    axis_1 = canvas_x_offset * torch.cos(theta) + canvas_y_offset * torch.sin(theta)
    axis_2 = canvas_y_offset * torch.cos(theta) - canvas_x_offset * torch.sin(theta)

    gauss = torch.exp(
        -((gamma ** 2) * (axis_1 ** 2) + (axis_2 ** 2)) / (2 * (sigma ** 2))
    )
    sinusoid = torch.cos(k * axis_1)
    fields = gauss * sinusoid

    return fields, gauss, sinusoid


def ols_fit(filter_set: _T, image_set: _T) -> _T:
    # filter set: [full image size, num_filters]
    # image_set: [num_images, full image size]
    # output: [num_images, num_filters]
    left_inv = torch.linalg.inv( filter_set.T @ filter_set ) @ filter_set.T
    return (left_inv @ image_set.unsqueeze(-1)).squeeze()


def ols_projection(filter_set: _T, image_set: _T) -> _T:
    "Appendix A, equation 2"
    # filter set: [full image size, num_filters]
    # image_set: [num_images, full image size]
    # output: [num_images, full image size]
    latents = ols_fit(filter_set, image_set)
    return torch.stack([(lt * filter_set).sum(1) for lt in latents])


def unexplained_variance_loss(x: _T, x_ols: _T, mean = True) -> _T:

    "Appendix A, equation 3. Assume all of size [n_filters, n_pixels]"
    error_squared = torch.square(x - x_ols).sum(-1)
    image_power = torch.square(x).sum(-1)
    fvu = (error_squared / image_power)
    return fvu.mean() if mean else fvu


def _p_x_given_z_gauss(A: _T, C: _T, z: _T, sigma_x: _T) -> _T:
    # Avoid repeated calcs that don't depend on X

    Nz = z.shape[0]
    filter_size = A.shape[0]

    # [total filter size, total filter size, Nz]
    ACAT = (A.to(torch.float64) @ C.to(torch.float64) @ A.to(torch.float64).T).unsqueeze(0).repeat(Nz, 1, 1)

    eye = torch.eye(filter_size).to(A.device)
    noise_term = (torch.square(sigma_x) * eye).unsqueeze(0).repeat(Nz, 1, 1)
    x_given_z_covar = ((z.reshape(-1, 1, 1) ** 2) * (ACAT)) + noise_term

    # [Nz, total filter size]
    zero = torch.zeros(x_given_z_covar.shape[:-1]).to(A.device)

    # try:
    #     x_given_z_covar_chol = torch.linalg.cholesky(x_given_z_covar)
    # except:
    #     import pdb; pdb.set_trace()

    gauss = torch.distributions.MultivariateNormal(zero, covariance_matrix = x_given_z_covar) # scale_tril=x_given_z_covar_chol)

    return gauss


def log_p_x_given_z(xs: _T, A: _T = None, C: _T = None, z: _T = None, sigma_x: _T = None, gauss = None) -> _T:

    if gauss is None:
        gauss = _p_x_given_z_gauss(A=A, C=C, z=z, sigma_x=sigma_x)

    batch_xs = xs.unsqueeze(1)
    
    # [num images, Nz]
    gaussian_loglikelihood = gauss.log_prob(batch_xs)

    return gaussian_loglikelihood


def log_p_z(z: _T, alpha: _T, beta: _T) -> _T:
    gamma_likelihood = torch.distributions.gamma.Gamma(
        concentration=alpha, rate=beta
    ).log_prob(z)
    return gamma_likelihood


def p_y_given_x_z(z: _T, sigma_x: _T, A: _T, C: _T, x: _T):
    Nz = z.shape[0]

    # [num_filters, num_filters, Nz]
    Cinv = torch.linalg.inv(C).unsqueeze(-1).repeat(1, 1, Nz)

    # [num_filters, num_filters, Nz]
    AtA = (A.T @ A).unsqueeze(-1).repeat(1, 1, Nz)

    # [Nz]
    coeff = (z / sigma_x) ** 2  

    # [num_filters, num_filters, Nz] -> changed
    S_z = torch.linalg.inv((Cinv + (coeff * AtA)).permute(-1, 0, 1))  

    # [num_images, num_filters, Nz]
    mu_z = (coeff / z) * ((S_z @ A.T) @ x.T).permute(2, 1, 0)  
    
    return mu_z, S_z.permute(2, 1, 0)
