from torch import Tensor as T
from torch.nn import functional as F

def make_rectified_power_law_activation_function(k, gamma):
    def rectified_power_law_activation_function(u: T):
        return k * (F.relu(u) ** gamma)
    rectified_power_law_activation_function.k = k
    rectified_power_law_activation_function.gamma = gamma
    return rectified_power_law_activation_function


def make_elementwise_thresholded_rectified_power_law_activation_function(k_vector, theta_vector, gamma_vector):
    def elementwise_thresholded_rectified_power_law_activation_function(u: T):
        trials_theta_vector = theta_vector.unsqueeze(-1).unsqueeze(0).repeat(u.shape[0], 1, u.shape[-1])
        trials_gamma_vector = gamma_vector.unsqueeze(-1).unsqueeze(0).repeat(u.shape[0], 1, u.shape[-1])
        trials_k_vector = k_vector.unsqueeze(-1).unsqueeze(0).repeat(u.shape[0], 1, u.shape[-1])
        rectified = F.relu(u - trials_theta_vector)
        return trials_k_vector * (rectified ** trials_gamma_vector)
    elementwise_thresholded_rectified_power_law_activation_function.k_vector = k_vector
    elementwise_thresholded_rectified_power_law_activation_function.theta_vector = theta_vector
    elementwise_thresholded_rectified_power_law_activation_function.gamma_vector = gamma_vector
    return elementwise_thresholded_rectified_power_law_activation_function


def identity(x):
    return x
