import torch
from torch import Tensor as _T
from typing import List

from purias_utils.multiitem_working_memory.util.circle_utils_parallel import generate_circular_feature_list


def gibbs_sample_new_circular_value(current_circular_value: _T, index: int, gap: float, inplace = False):
    """
    Given current zeta_a (location/colour) values of shape [num batches, num samples, num stimuli, 1],
    This will take the index'th stimulus of each sample (current_zeta_a[:,:,index]) and resample it according to Gibbs
        i.e. all items in batch will have same index stimulus value replaced
    
    Maintains a :gap radians on both sides of every stimulus - i.e. dependent sample structure
    """

    batch_size, num_chains, n_stim, f_dim = current_circular_value.shape
    total_batch_size = batch_size * num_chains
    assert f_dim == 1

    if gap == 0.0 or n_stim == 1.0:
        # Save some time in the independent case
        new_zeta_index = torch.rand(num_chains) * 2 * torch.pi
    
    else:
        non_cued_indices = [i for i in range(n_stim) if i != index]
        new_zeta_index = torch.zeros(batch_size, num_chains)

        existing_values = current_circular_value[...,non_cued_indices,0].reshape(total_batch_size, -1)
        asdf = generate_circular_feature_list(
            total_batch_size, n_stim, gap, existing_features = existing_values.numpy()
        )
        new_zeta_index = torch.tensor(asdf.reshape(batch_size, num_chains, n_stim, 1)[:,:,-1,0])

    new_circular_value = current_circular_value if inplace else current_circular_value.clone()
    new_circular_value[:,:,index,0] = new_zeta_index

    return new_circular_value



def old_full_gibbs_iteration(feature_set: List[_T], expected_D, feature_margins):
    """
    Changed in 19.4.24 update - new shape expectations
    See above for shape expections. Save memory by making individual changes inplace.
    """

    _, _, n_stim, _ = feature_set[0].shape
    new_feature_set = [fs.clone() for fs in feature_set]
    assert len(new_feature_set) == expected_D

    for i in range(n_stim):
        for d in range(expected_D):
            new_feature_set[d] = gibbs_sample_new_circular_value(
                new_feature_set[d], i, feature_margins[d], 
                inplace = True
            )
    
    return new_feature_set



def old_bin_colour_samples_samples(all_target_features, num_bins, cued_index):
    """
    all_target_features of shape [batch, iterations, chains, stim, 1],
        as output by ToroidalFeaturesIdealObserverBase.iterate_stimulus_gibbs_sampling

    Return samples in shape [batch_size, num_iterations, num_chains, num_stimuli, feature_dim (1)]

    Remember, we are taking expectation with respect to all zeta_zs and zeta_cs EXCEPT one of the zeta_cs
        This is the indexth sample!
    Therefore, we need to extract all samples from the MCMC chains binned by that remaining zeta_cs

    This will return a tensor of num_bins bin boundaries (circular), and a tensor of num_bins binary masks
        Each mask will be of size [num_iterations, num_chains], with 1s where a sample is in that bin!
        The masks[j] will relate to the bin with lower bound bin_boundaries[j]
    """
    bin_boundaries = torch.linspace(0, 2*torch.pi, num_bins + 1)    # Includes 2pi at the end also!
    relevant_features = all_target_features[...,cued_index,0]
    masks = [
        torch.logical_and(bin_lb <= relevant_features, relevant_features < bin_ub)
        for bin_lb, bin_ub in zip(bin_boundaries[:-1], bin_boundaries[1:])
    ]
    return bin_boundaries, torch.stack(masks, dim = 0).float()


