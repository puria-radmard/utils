import torch

from purias_utils.util.api import return_as_obj

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel

def likelihood_inner(data, generative_model: NonParametricSwapErrorsGenerativeModel, training_method, pi_vectors, exp_f_evals, return_posterior: bool):
    """
    data expected in shape:
        if error:   [M, N]      - all errors needed because this sample of pi_vectors may choosen any beta - in fact we marginalise out beta...
        if beta:    [M]         - integers between 0 and N inclusive
        if pi:      [M, N+1]    - valid distributions along axis 1
    """
    posterior_vectors, unaggregated_lh = None, None
    if training_method == 'error':
        llh_term, posterior_vectors, unaggregated_lh = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = data, pi_vectors = pi_vectors, return_posterior = return_posterior
        )
    elif training_method == 'beta':
        llh_term = generative_model.get_component_likelihood(
            selected_components = data, pi_vectors = pi_vectors
        )
    elif training_method == 'pi':
        llh_term = generative_model.get_categorical_likelihood(
            real_pi_vectors = data, exp_f_evals = exp_f_evals
        )
    return llh_term, posterior_vectors, unaggregated_lh


def get_elbo_terms(variational_model: NonParametricSwapErrorsVariationalModel, generative_model: NonParametricSwapErrorsGenerativeModel, deltas, data, M, N, I, training_method, return_kl = True, return_posterior = False):
    
    R = variational_model.R
    
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas)
    set_size = deltas.shape[1]

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
        k_ud=k_ud, K_dd=K_dd, K_uu=K_uu, K_uu_inv=K_uu_inv
    )

    # Get the samples of f evaluated at the data
    f_samples = variational_model.reparameterised_sample(
        num_samples = I, mu = mu, sigma_chol = sigma_chol, 
        M = M, N = N
    )

    pi_vectors, exp_f_evals = generative_model.swap_function.generate_pi_vectors(
        N, model_evaulations = f_samples, return_exp_grid = True
    )

    # Get the ELBO first term, depending on training mode (data is usually errors)
    llh_term, posterior, unaggregated_lh = likelihood_inner(data, generative_model, training_method, pi_vectors, exp_f_evals, return_posterior)

    # if quick_scatter: 
    #     import matplotlib.pyplot as plt
    #     plt.clf()
    #     plt.figure()
    #     for f_sample in f_samples[:,:,1:].detach().cpu().numpy(): plt.scatter(deltas[:,1:,0].flatten().cpu().numpy(), f_sample.flatten())
    #     plt.savefig('asdf_me.png')
    #     plt.close('all')

    return llh_term, kl_term, unaggregated_lh, posterior, pi_vectors, exp_f_evals


def training_step(
    generative_model, variational_model, deltas, errors, 
    M, N, I, D, training_method,
    threshold_parameter = 0.95, reg_weighting = 10.0, return_posterior = False,
):

    llh_term, kl_term, unaggregated_lh, posterior, pi_vectors, exp_f_evals = get_elbo_terms(variational_model, generative_model, deltas, errors, M, N, I, training_method, True, return_posterior)

    pdists = (variational_model.Z.unsqueeze(1) - variational_model.Z).abs()   # [R, R, D]
    tril_idx = torch.tril_indices(variational_model.R, variational_model.R, -1)
    pdist_uts = torch.stack([pdi[tril_idx[0], tril_idx[1]] for pdi in pdists.permute(2, 0, 1)], -1)
    cos_p2_dists = ((1 + pdist_uts.cos())**2).sum(-1)                           # [0.5 * R * (R-1)]
    thres = (D * 4) * threshold_parameter
    cos_p2_dists_thres = cos_p2_dists * (cos_p2_dists > thres)
    distance_loss = cos_p2_dists_thres.sum() * reg_weighting

    return llh_term, unaggregated_lh, kl_term, distance_loss, posterior, pi_vectors, exp_f_evals


def get_elbo_terms_spike_and_slab(
    generative_model: NonParametricSwapErrorsGenerativeModel, errors, M, N, training_method, return_posterior = True
):
    pi_vectors, exp_f_evals = generative_model.swap_function.generate_pi_vectors(
        set_size = N, batch_size = M, return_exp_grid = True
    )

    llh_term, posterior = likelihood_inner(errors, generative_model, training_method, pi_vectors, exp_f_evals, return_posterior)

    return llh_term, posterior, pi_vectors, exp_f_evals




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
            k_ud=k_ud, K_dd=K_dd, K_uu=K_uu, K_uu_inv=K_uu_inv
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

