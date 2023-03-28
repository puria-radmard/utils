import torch
from torch import linalg
from torch import Tensor as T
from torch.nn import functional as F
from tqdm import tqdm
from maths.graph import binarise_weight_matrix
from itertools import combinations


def pairwise_average(W: T):
    """
    Average E-E connectivity
    This follows same convention as tuning_curves_pdist
    """
    num_neurons = W.shape[0]
    assert W.shape == (num_neurons, num_neurons), "pairwise_average made for square matrix"
    averaged_weights = 0.5 * (W + W.T)
    indices = torch.triu_indices(num_neurons, num_neurons, 1).T
    return torch.tensor([averaged_weights[idx[0], idx[1]] for idx in indices])



def input_weights_dot_product(W: T, norm_curves = False):
    """
    Similarity in tuning curve
    This follows same convention as pairwise_average
    """
    num_neurons, input_size = W.shape   # i.e. rows are tuning curves
    tuning = (W / linalg.vector_norm(W, dim=1).unsqueeze(-1)) if norm_curves else W
    outer_product = torch.einsum('id,jd->ij', tuning, tuning)
    indices = torch.triu_indices(num_neurons, num_neurons, 1).T
    return torch.tensor([outer_product[idx[0], idx[1]] for idx in indices])


def get_all_triplet_indices(num_cells: int):
    symbols = list(range(num_cells))
    for combo in combinations(symbols, 3):
        yield combo

def get_all_pair_indices(num_cells: int):
    symbols = list(range(num_cells))
    for combo in combinations(symbols, 2):
        yield combo

def get_subweights_from_triplet(W: T, triplet_indices):
    ti = list(triplet_indices)
    return W[ti][:,ti]




def get_feature_type_from_tiplet_subweight(subweights):
    """
    Return number based on Song 05 figure 4B
    NB for subweights - outwards from neuron i is the ith row
    
    We discount autosynapsing!!
    """
    assert subweights.shape == (3, 3)
    for i in range(3): subweights[i,i] = 0

    total_sum = subweights.sum().item()
    incoming = list(subweights.sum(0))
    outgoing = list(subweights.sum(1))

    full_range = {0, 1, 2}
    
    # Simple check based on salient patterns:
    basic_checks = {0: 1, 1: 2, 5: 15, 6: 16}
    if total_sum in basic_checks.keys():
        return basic_checks[total_sum]
    
    # Patters with two connections only:
    if total_sum == 2:
        
        if 2 in outgoing:
            return 4
        
        elif 2 in incoming:
            return 5
        
        elif torch.all(subweights == subweights.T):
            return 3
        
        else:
            return 6
        
    # Patterns with three connections
    if total_sum == 3:
        
        if set(incoming) == set(outgoing) == {1}:
            return 11
        
        elif (set(outgoing) == {1}) and (set(incoming) == full_range):
            return 7
        
        elif set(incoming) == set(outgoing) == full_range:
            return 10
        
        else:
            return 8
        
        
    # Patterns with four connections
    if total_sum == 4:
        
        if set(incoming) == set(outgoing) == {1, 2}:
            return 13
        
        elif set(outgoing) == full_range:
            return 12
        
        elif torch.all(subweights == subweights.T):
            return 9
        
        else:
            return 14
        
    raise ValueError
        


def count_features_in_binary(W: T, tqdm_it = False):
    """
    Feature type ordering is based on Song 05 figure 4B
    """
    num_cells = len(W)
    assert W.shape[1] == num_cells
    
    all_triplet_types = []
    
    for triplet_idxs in tqdm(list(get_all_triplet_indices(num_cells = num_cells)), disable = not tqdm_it):
        sw = get_subweights_from_triplet(W, triplet_idxs)
        new_feature = get_feature_type_from_tiplet_subweight(sw)
        all_triplet_types.append(new_feature)
        
    return all_triplet_types


def average_over_dicts(*dicts):
    all_keys = []
    for d in dicts:
        all_keys.extend(list(d.keys()))
    out_dict = {
        ak: sum([d.get(ak, 0) for d in dicts]) / len(dicts)
        for ak in all_keys
    }
    return out_dict


def chance_connection_types(binarised_weights):
    # First, get the probabilities in the 'cortex'
    non_conn, uni_conn, bi_conn = 0, 0, 0
    for i, j in get_all_pair_indices(binarised_weights.shape[0]):
        if i == j:
            continue    # Ignore autosynapsing, shouldn't even appear here
        if (binarised_weights[i,j] == 1.0) and (binarised_weights[j,i] == 1.0):
            bi_conn += 1
        elif (binarised_weights[i,j] == 1.0) or (binarised_weights[j,i] == 1.0):
            uni_conn += 1
        else:
            non_conn += 1

    total = sum([non_conn, uni_conn, bi_conn])
    non_prob, uni_prob, bi_prob = non_conn/total, uni_conn/total, bi_conn/total
    return non_prob, uni_prob, bi_prob



def null_hypothesis_over_features(non_prob, uni_prob, bi_prob):
    """
    Feature type ordering is based on Song 05 figure 4B
    """
    null_hyp_probs = {
        1: non_prob * non_prob * non_prob,
        2: non_prob * non_prob * uni_prob,
        3: non_prob * non_prob * bi_prob,
        4: non_prob * uni_prob * uni_prob,
        5: non_prob * uni_prob * uni_prob,
        6: non_prob * uni_prob * uni_prob,
        7: non_prob * uni_prob * bi_prob,
        8: non_prob * uni_prob * bi_prob,
        9: non_prob * bi_prob * bi_prob,
        10: uni_prob * uni_prob * uni_prob,
        11: uni_prob * uni_prob * uni_prob,
        12: uni_prob * uni_prob * bi_prob,
        13: uni_prob * uni_prob * bi_prob,
        14: uni_prob * uni_prob * bi_prob,
        15: bi_prob * bi_prob * uni_prob,
        16: bi_prob * bi_prob * bi_prob,
    }
    return null_hyp_probs





if __name__ == '__main__':

    ## TESTING CONVENTION ABOVE
    n_input = 8
    n_neurons = 5
    W_input = torch.zeros([n_neurons, n_input])
    W_hidden = torch.zeros([n_neurons, n_neurons])
    W_input[1, 4] = 1
    W_input[2, 3] = 1
    W_hidden[1, 2] = 5
    W_hidden[2, 1] = 6

    print(W_input)
    print(W_hidden)

    print(input_weights_dot_product(W_input))
    print(pairwise_average(W_hidden))

    ## ==> same position for biggest difference
