import torch

from purias_utils.util.api import return_as_obj

from numpy import ndarray as _A
from typing import Dict

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel


def get_elbo_terms(variational_model: NonParametricSwapErrorsVariationalModel, generative_model: NonParametricSwapErrorsGenerativeModel, deltas, data, I, training_method, max_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}):
    
    R = variational_model.R
    Q, M, N = data.shape
    assert Q == variational_model.num_models
    
    all_deduplicated_deltas, M_minis = variational_model.deduplicate_deltas(deltas, max_batch_size)  # "~M/batch_size length list of entries of shape [Q, ~batch*N, D]"

    # Use kernel all here:
    K_dds = [generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]                         # each [Q, ~batch*N, ~batch*N]
    K_uu = generative_model.swap_function.evaluate_kernel(N, variational_model.Z)                                                                               # [Q, R, R]
    k_uds = [generative_model.swap_function.evaluate_kernel(N, variational_model.Z, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]    # each [Q, R, ~batch*N]

    # Inverse isn't always symmetric!!
    K_uu_inv = torch.linalg.inv(K_uu)
    assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
    if not torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all():
        K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
        K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))

    # Get the KL term of the loss
    if return_kl:
        kl_term = variational_model.kl_loss(K_uu = K_uu, K_uu_inv=K_uu_inv)     # [Q]
    else:
        kl_term = torch.ones(Q) * torch.nan # Won't plot!

    # Make variational inferences for q(f)
    mus, sigma_chols = [], []
    for k_ud, K_dd in zip(k_uds, K_dds):
        _mu, _sigma, _sigma_chol = variational_model.variational_gp_inference(k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv)  # [Q, ~batch*N], [Q, ~batch*N, ~batch*N], [Q, ~batch*N, ~batch*N]
        mus.append(_mu), sigma_chols.append(_sigma_chol)

    # Get the samples of f evaluated at the data
    all_f_samples = [
        variational_model.reparameterised_sample(num_samples = I, mu = mu, sigma_chol = sigma_chol, M = M, N = N)
        for mu, sigma_chol, M in zip(mus, sigma_chols, M_minis)
    ]   # Each of shape [Q, I, ~batchsize, N]

    # Shouldn't be any numerical problems after this
    f_samples = torch.concat(all_f_samples, 2)  # [Q, I, M, N]

    prior_info = generative_model.swap_function.generate_pi_vectors(
        set_size = N, model_evaulations = f_samples
    )
    priors = prior_info['pis']

    # Get the ELBO first term, depending on training mode (data is usually errors)
    if training_method == 'error':
        total_log_likelihood, likelihood_per_datapoint, posterior_vectors = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = data, pi_vectors = priors,
            kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
        )
    elif training_method == 'beta':
        raise Exception
        llh_term = generative_model.get_component_log_likelihood(
            selected_components = data, pi_vectors = pi_vectors
        )
        posterior, unaggregated_lh = None, None

    #     import matplotlib.pyplot as plt
    #     plt.clf()
    #     plt.figure()
    #     for f_sample in f_samples[:,:,1:].detach().cpu().numpy(): plt.scatter(deltas[:,1:,0].flatten().cpu().numpy(), f_sample.flatten())
    #     plt.savefig('asdf_me.png')
    #     plt.close('all')

    distance_loss = K_uu.tril(-1).max() / K_uu.max()

    return {
        'total_log_likelihood': total_log_likelihood, 
        'likelihood_per_datapoint': likelihood_per_datapoint, 
        'posterior': posterior_vectors, 
        'priors': priors, 
        'f_samples': f_samples,             # [Q, I, M, N]
        'kl_term': kl_term, 
        'distance_loss': distance_loss
    }



def get_elbo_terms_spike_and_slab(
    generative_model: NonParametricSwapErrorsGenerativeModel, errors, M, N, training_method, kwargs_for_individual_component_likelihoods = {}
):
    prior_info = generative_model.swap_function.generate_pi_vectors(set_size = N, batch_size = M)
    priors = prior_info['pis']

    if training_method == 'error':
        total_log_likelihood, likelihood_per_datapoint, posterior_vectors = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = errors, pi_vectors = priors, kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
        )
    elif training_method == 'beta':
        raise Exception
        llh_term = generative_model.get_component_log_likelihood(
            selected_components = errors, pi_vectors = pi_vectors
        )
        posterior, unaggregated_lh = None, None

    distance_loss = torch.zeros(generative_model.num_models, device = priors.device, dtype = priors.dtype)

    return {
        'total_log_likelihood': total_log_likelihood,       # [Q]
        'likelihood_per_datapoint': likelihood_per_datapoint, # [Q, M]
        'posterior': posterior_vectors,     # [Q, M, N+1]
        'priors': priors,                   # [Q, M, N+1]
        'f_samples': None,
        'kl_term': distance_loss,           # [Q]
        'distance_loss': distance_loss      # [Q]
    }




def inference_inner(set_size, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, flattened_deltas):
    # Use kernel all here, but without depduplication

    with torch.no_grad():

        R = variational_model.R
        # TODO: add this noise injection to report!
        K_dd = generative_model.swap_function.evaluate_kernel(set_size, flattened_deltas)
        K_uu = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z)
        k_ud = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z, flattened_deltas)
        K_uu_inv = torch.linalg.inv(K_uu)
        # K_dd_inv = torch.linalg.inv(K_dd)

        K_uu_inv = torch.linalg.inv(K_uu)
        assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
        K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
        K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
        assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

        # Make variational inferences for q(f)
        mu, sigma, sigma_chol = variational_model.variational_gp_inference(
            k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
        )

    return mu, sigma, sigma_chol


@return_as_obj
def inference(generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, deltas):
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas, 0)[0][0]
    Q, M, set_size, D = deltas.shape
    mu, sigma, sigma_chol = inference_inner(set_size, generative_model, variational_model, deduplicated_deltas)
    return {
        'mu': mu, 
        'sigma': sigma, 
        'sigma_chol': sigma_chol, 
        'deduplicated_deltas': deduplicated_deltas
    }


def inference_mean_only_inner(set_size: int, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model, flattened_deltas):
    # Use kernel all here, but without depduplication
    raise Exception ('Please stop using inference_mean_only')

    with torch.no_grad():

        R = variational_model.R
        K_uu = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z)
        k_ud = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z, flattened_deltas)
        K_uu_inv = torch.linalg.inv(K_uu)

        K_uu_inv = torch.linalg.inv(K_uu)
        assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
        K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
        K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
        assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

        # Make variational inferences for q(f)
        mu = variational_model.variational_gp_inference_mean_only(k_ud = k_ud, K_uu_inv = K_uu_inv)

    return mu


def inference_mean_only(generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, deltas):
    raise Exception ('Please stop using inference_mean_only')
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas, 0)[0][0]
    Q, M, set_size, D = deltas.shape
    mu = inference_mean_only_inner(set_size, generative_model, variational_model, deduplicated_deltas)
    return mu


def inference_on_grid(swap_type: str, set_size: int, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, grid_count: int, device: str = 'cuda') -> Dict[str, _A]:
    """
    Set up a suitably sized grid and perform inference over it

    TODO: shapes
    """
    # full_grid = torch.linspace(-torch.pi, +torch.pi, grid_count + 1)[:-1]
    full_grid = torch.linspace(-torch.pi, +torch.pi, grid_count)
    grid_locs = torch.tensor([0.0]) if swap_type == 'est_dim_only' else full_grid.clone()
    grid_cols = torch.tensor([0.0]) if swap_type == 'cue_dim_only' else full_grid.clone()
    grid_x, grid_y = torch.meshgrid(grid_locs, grid_cols, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y], -1).reshape(len(grid_cols) * len(grid_locs), 2).to(device)
    if swap_type == 'cue_dim_only':
        grid_points = grid_points[...,[0]]    # Only locations
    elif swap_type == 'est_dim_only':
        grid_points = grid_points[...,[1]]    # Only colours
    
    grid_points = grid_points.unsqueeze(0).repeat(generative_model.num_models, 1, 1)
    flat_mu_est, flat_sigma_est, sigma_chol = inference_inner(set_size, generative_model, variational_model, grid_points)

    std_est = torch.stack([fse.diag() for fse in flat_sigma_est], 0).sqrt() # [Q, 100]

    eps = torch.randn(generative_model.num_models, 3, flat_mu_est.shape[1], dtype = flat_mu_est.dtype, device = flat_mu_est.device) # [Q, 3, MN (dedup size)]
    grid_f_samples = flat_mu_est.unsqueeze(1) + torch.bmm(eps, sigma_chol.transpose(-1, -2))   # [Q, 3, MN]

    return {
        'one_dimensional_grid': full_grid.cpu().numpy(),            # [Q, 100]
        'all_grid_points': grid_points.cpu().numpy(),               # [Q, 100, 1]
        'mean_surface': flat_mu_est.cpu().numpy(),                  # [Q, 100]
        'std_surface': std_est.cpu().numpy(),                       # [Q, 100]
        'function_samples_on_grid': grid_f_samples.cpu().numpy()    # [Q, 3, 100]
    }
