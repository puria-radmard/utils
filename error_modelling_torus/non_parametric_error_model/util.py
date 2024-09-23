import torch

from purias_utils.util.api import return_as_obj

from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel


def get_elbo_terms(variational_model: NonParametricSwapErrorsVariationalModel, generative_model: NonParametricSwapErrorsGenerativeModel, deltas, data, I, training_method, max_batch_size = 0, return_kl = True, kwargs_for_individual_component_likelihoods = {}):
    
    R = variational_model.R
    Q, M, N = deltas.shape[:2]
    assert Q == variational_model.num_models
    
    all_deduplicated_deltas, M_minis = variational_model.deduplicate_deltas(deltas, max_batch_size)  # "~M/batch_size length list of entries of shape [Q, ~batch*N, D]"

    # Use kernel all here:
    K_dds = [generative_model.swap_function.evaluate_kernel(N, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]                         # each [Q, ~batch*N, ~batch*N]
    K_uu = generative_model.swap_function.evaluate_kernel(N, variational_model.Z)                                                                               # [Q, R, R]
    k_uds = [generative_model.swap_function.evaluate_kernel(N, variational_model.Z, deduplicated_deltas) for deduplicated_deltas in all_deduplicated_deltas]    # each [Q, R, ~batch*N]

    # Inverse isn't always symmetric!!
    K_uu_inv = torch.linalg.inv(K_uu)
    assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
    K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
    K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
    assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

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

    pi_vectors, exp_f_evals = generative_model.swap_function.generate_pi_vectors(
        set_size = N, model_evaulations = f_samples, return_exp_grid = True
    )

    # Get the ELBO first term, depending on training mode (data is usually errors)
    if training_method == 'error':
        llh_term, posterior, unaggregated_lh = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = data, pi_vectors = pi_vectors,
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

    return {
        'llh_term': llh_term, 
        'kl_term': kl_term, 
        'unaggregated_lh': unaggregated_lh, 
        'posterior': posterior, 
        'pi_vectors': pi_vectors, 
        'f_samples': f_samples,
        'K_uu': K_uu
    }



def get_elbo_terms_spike_and_slab(
    generative_model: NonParametricSwapErrorsGenerativeModel, errors, M, N, training_method, kwargs_for_individual_component_likelihoods = {}
):
    pi_vectors, exp_f_evals = generative_model.swap_function.generate_pi_vectors(
        set_size = N, batch_size = M, return_exp_grid = True
    )

    if training_method == 'error':
        llh_term, posterior, unaggregated_lh = generative_model.get_marginalised_log_likelihood(
            estimation_deviations = errors, pi_vectors = pi_vectors, kwargs_for_individual_component_likelihoods = kwargs_for_individual_component_likelihoods
        )
    elif training_method == 'beta':
        raise Exception
        llh_term = generative_model.get_component_log_likelihood(
            selected_components = errors, pi_vectors = pi_vectors
        )
        posterior, unaggregated_lh = None, None

    return {
        'llh_term': llh_term,
        'unaggregated_lh': unaggregated_lh,
        'posterior': posterior,
        'pi_vectors': pi_vectors,
        'f_samples': None,
        'kl_term': torch.tensor(0.0),
        'distance_loss': torch.tensor(0.0),
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

        # Get the KL term of the loss
        kl_term = variational_model.kl_loss(K_uu = K_uu, K_uu_inv=K_uu_inv)

        # Make variational inferences for q(f)
        mu, sigma, sigma_chol = variational_model.variational_gp_inference(
            k_ud=k_ud, K_dd=K_dd, K_uu_inv=K_uu_inv
        )

    return kl_term, mu, sigma, sigma_chol


@return_as_obj
def inference(generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, deltas):
    deduplicated_deltas, Ms = variational_model.deduplicate_deltas(deltas, 0)[0][0]
    set_size = deltas.shape[-1]
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

        K_uu_inv = torch.linalg.inv(K_uu)
        assert torch.isclose(torch.bmm(K_uu_inv, K_uu), torch.eye(R, dtype = K_uu.dtype, device = K_uu.device)).all()
        K_uu_inv_chol = torch.linalg.cholesky(K_uu_inv)
        K_uu_inv = torch.bmm(K_uu_inv_chol, K_uu_inv_chol.transpose(1, 2))
        assert torch.isclose(K_uu_inv, K_uu_inv.transpose(1, 2)).all()

        # Make variational inferences for q(f)
        mu = variational_model.variational_gp_inference_mean_only(k_ud = k_ud, K_uu_inv = K_uu_inv)

    return mu


def inference_mean_only(generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, deltas):
    deduplicated_deltas = variational_model.deduplicate_deltas(deltas, 0)[0][0]
    set_size = deltas.shape[-1]
    mu = inference_mean_only_inner(set_size, generative_model, variational_model, deduplicated_deltas)
    return mu

