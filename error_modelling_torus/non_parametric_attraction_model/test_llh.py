import torch
import matplotlib.pyplot as plt

from purias_utils.error_modelling_torus.non_parametric_attraction_model.emissions_distribution import skew_wrapped_stable_distribution_likelihood, skew_wrapped_stable_generate_samples


num_points = 1000

estimation_error = torch.linspace(-torch.pi, +torch.pi, num_points)

# These are problematic parameters
# alpha_parameter_samples = torch.ones(1, 1, num_points) *  0.02220804677216481
# beta_parameter_samples = torch.ones(1, 1, num_points) *              0.7969702955699913
# gamma_parameter_samples = torch.ones(1, 1, num_points) *              4.067670441198219
# delta_parameter_samples = torch.ones(1, 1, num_points) *             -0.8263661176863124

alpha_parameter_samples = torch.ones(1, 1, num_points) * 1.764465461315126
beta_parameter_samples = torch.ones(1, 1, num_points) * 0.9936384494358959
gamma_parameter_samples = torch.ones(1, 1, num_points) * 0.11552389050755572
delta_parameter_samples = torch.ones(1, 1, num_points) * 0.5002900436026771


lh = skew_wrapped_stable_distribution_likelihood(
    estimation_error,
    alpha_parameter_samples,
    beta_parameter_samples,
    gamma_parameter_samples,
    delta_parameter_samples,
    p_cut_off = 1000,
    throw_error=False,
    negative_threshold=0.0,
    boundary_clip=-float('inf')
)


samples = skew_wrapped_stable_generate_samples(
    alpha_parameter_samples,
    beta_parameter_samples,
    gamma_parameter_samples,
    delta_parameter_samples,
)


plt.plot(estimation_error.numpy(), lh[0,0].numpy())
plt.hist(samples[0,0].numpy(), 40, density = True)

print((lh.sum(-1) * (estimation_error[-1] - estimation_error[-2])).item())

plt.savefig('asdf')
