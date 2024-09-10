import torch

from purias_utils.util.api import return_as_obj

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel


def get_elbo_terms(variational_model: NonParametricSwapErrorsVariationalModel, generative_model: NonParametricSwapErrorsGenerativeModel, deltas, data, I, training_method, return_kl = True):
    
    R = variational_model.R
    batch_size, set_size = deltas.shape[:2]
    
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas)

    # Use kernel all here:
    K_dd = generative_model.swap_function.evaluate_kernel(set_size, deduplicated_deltas)
    K_uu = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z)
    k_ud = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z, deduplicated_deltas)

    K_uu_inv = torch.linalg.inv(K_uu)
    # K_dd_inv = torch.linalg.inv(K_dd)

    assert ((K_uu_inv @ K_uu).round().abs().detach().cpu() == torch.eye(R)).all()

    # Get the KL term of the loss
    if return_kl:
        kl_term = variational_model.kl_loss(K_uu = K_uu, K_uu_inv=K_uu_inv)
    else:
        kl_term = torch.tensor(torch.nan) # Won't plot!

    # Make variational inferences for q(f)
    mu, sigma, sigma_chol = variational_model.variational_gp_inference(
        k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
    )

    # Get the samples of f evaluated at the data
    f_samples = variational_model.reparameterised_sample(
        num_samples = I, mu = mu, sigma_chol = sigma_chol, 
        M = batch_size, N = set_size
    )

    pi_vectors, exp_f_evals = generative_model.swap_function.generate_pi_vectors(
        set_size, model_evaulations = f_samples, return_exp_grid = True
    )

    # Get the ELBO first term, depending on training mode (data is usually errors)
    if training_method == 'error':
        llh_term, posterior, unaggregated_lh = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = data, pi_vectors = pi_vectors
        )
    elif training_method == 'beta':
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

    return llh_term, kl_term, unaggregated_lh, posterior, pi_vectors, exp_f_evals


def training_step(
    generative_model, variational_model, deltas, errors, 
    I, D, training_method,
    threshold_parameter = 0.95, reg_weighting = 10.0,
):

    llh_term, kl_term, unaggregated_lh, posterior, pi_vectors, exp_f_evals = get_elbo_terms(variational_model, generative_model, deltas, errors, I, training_method, True)

    pdists = (variational_model.Z.unsqueeze(1) - variational_model.Z).abs()   # [R, R, D]
    tril_idx = torch.tril_indices(variational_model.R, variational_model.R, -1)
    pdist_uts = torch.stack([pdi[tril_idx[0], tril_idx[1]] for pdi in pdists.permute(2, 0, 1)], -1)
    cos_p2_dists = ((1 + pdist_uts.cos())**2).sum(-1)                           # [0.5 * R * (R-1)]
    thres = (D * 4) * threshold_parameter
    cos_p2_dists_thres = cos_p2_dists * (cos_p2_dists > thres)
    distance_loss = cos_p2_dists_thres.sum() * reg_weighting

    return llh_term, unaggregated_lh, kl_term, distance_loss, posterior, pi_vectors, exp_f_evals


def get_elbo_terms_spike_and_slab(
    generative_model: NonParametricSwapErrorsGenerativeModel, errors, M, N, training_method
):
    pi_vectors, exp_f_evals = generative_model.swap_function.generate_pi_vectors(
        set_size = N, batch_size = M, return_exp_grid = True
    )

    if training_method == 'error':
        llh_term, posterior, unaggregated_lh = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = errors, pi_vectors = pi_vectors
        )
    elif training_method == 'beta':
        llh_term = generative_model.get_component_log_likelihood(
            selected_components = errors, pi_vectors = pi_vectors
        )
        posterior, unaggregated_lh = None, None

    return llh_term, unaggregated_lh, posterior, pi_vectors, exp_f_evals




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

        assert ((K_uu_inv @ K_uu).round().abs().detach().cpu() == torch.eye(R)).all()

        # Get the KL term of the loss
        kl_term = variational_model.kl_loss(K_uu = K_uu, K_uu_inv=K_uu_inv)

        # Make variational inferences for q(f)
        mu, sigma, sigma_chol = variational_model.variational_gp_inference(
            k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
        )

    return kl_term, mu, sigma, sigma_chol


@return_as_obj
def inference(generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, deltas):
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas)
    set_size = deltas.shape[1]
    kl_term, mu, sigma, sigma_chol = inference_inner(set_size, generative_model, variational_model, deduplicated_deltas)
    return {
        'kl_term': kl_term, 
        'mu': mu, 
        'sigma': sigma, 
        'sigma_chol': sigma_chol, 
        'deduplicated_deltas': deduplicated_deltas
    }


def inference_mean_only_inner(set_size: int, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model, flattened_deltas):
    # Use kernel all here, but without depduplication

    with torch.no_grad():

        R = variational_model.R
        K_uu = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z)
        k_ud = generative_model.swap_function.evaluate_kernel(set_size, variational_model.Z, flattened_deltas)
        K_uu_inv = torch.linalg.inv(K_uu)

        assert ((K_uu_inv @ K_uu).round().abs().detach().cpu() == torch.eye(R)).all()

        # Make variational inferences for q(f)
        mu = variational_model.variational_gp_inference_mean_only(k_ud = k_ud, K_uu_inv = K_uu_inv)

    return mu


def inference_mean_only(generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, deltas):
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas)
    set_size = deltas.shape[1]
    mu = inference_mean_only_inner(set_size, generative_model, variational_model, deduplicated_deltas)
    return mu

