import torch
from torch import pi
from torch import Tensor as _T
from torch.distributions.von_mises import _log_modified_bessel_fn

import math

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


def g_distr(theta: _T):
    return (pi - rectify_angles(theta).abs()) / (pi**2)

def g_hat_distr(theta: _T):
    return (rectify_angles(theta).abs()) / (pi**2)

def zero_mean_von_mises_log_prob(von_mises_sigma2: _T, theta_axis: _T) -> _T:
    concentration = 1. / von_mises_sigma2
    log_prob = concentration * torch.cos(theta_axis)     # Always zero mean
    log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn(concentration, order=0)
    return log_prob

def zero_mean_von_mises_log_prob_with_kappa(von_mises_concentration: _T, theta_axis: _T) -> _T:
    log_prob = von_mises_concentration * torch.cos(theta_axis)     # Always zero mean
    log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn(von_mises_concentration, order=0)
    return log_prob


def normalised_log_normal_statistics(mean: _T, std: _T, pi_u_tilde: _T, scale_ln_std = 1.0):
    "For u ~ N(mean, std^2), generates v = e^{u - pi_u_tilde} and returns mean and std"
    # Normal shifted before exponential
    mu = mean - pi_u_tilde
    var = std.square()

    # Log-Normal stats
    ln_mean = (mu + (0.5 * var)).exp()
    ln_esv = ((2.0 * mu) + (2.0 * var)).exp()
    ln_var = ln_esv - ln_mean.square()
    ln_std = ln_var.sqrt()

    return ln_mean, ln_std * scale_ln_std


def kl(p1, p2):
    log_grid = (p1 * (p1 / p2).log())
    log_grid[p1 == 0.] = 0.0
    return log_grid.sum(-1).mean(0)


def anneal_categorical(pis: _T, tau: float):
    "Expecting categorical size as last axis"
    scaled_log_pis = pis.log() / tau
    return scaled_log_pis.softmax(-1)


def annealed_kl(p1, p2, temperature):
    "Anneal both pmfs before calculating kl divergence"
    ap1 = anneal_categorical(p1, temperature)
    ap2 = anneal_categorical(p2, temperature)
    return kl(ap1, ap2)


def cross_entropy(p1, p2):
    log_grid = - p1 * p2.log()
    return log_grid.sum(-1).mean(0)


def div_norm(x: _T):
    norm = x.square().sum(-1, keepdim = True)
    return x / norm

def I0(conc: _T):
    return _log_modified_bessel_fn(conc, order=0).exp()

def convert_plot_to_circular(rad_x: _T, rad_y: _T):
    "rad_x determines angle (done weirdly due to arctan!), rad_y + 1 determines radius"
    radius = (1.0 + rad_y)
    cart_x = radius * rad_x.sin()
    cart_y = radius * rad_x.cos()
    return cart_x, cart_y

