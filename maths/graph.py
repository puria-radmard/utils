"""
seRNN analysis measures and regs implemented here
"""
import torch, bct
from typing import Union
import numpy as np

import torch
from torch import Tensor as T
from torch.nn import functional as F

def outbound_degree(W: T):
    return W.abs().sum(1)


def communicability_reg(W: T):
    degree = outbound_degree(W=W)
    sqrtdegree =  torch.diag(1 / torch.sqrt(degree))
    exponent = sqrtdegree @ W.abs() @ sqrtdegree
    return torch.linalg.matrix_exp(exponent)


def modularity(param, gamma=1, for_json=False):
    if isinstance(param, T):
        abs_param = param.abs().numpy()
    else:
        abs_param = abs(param)
    c, q = bct.modularity_und(abs_param, gamma)
    return {
        "c": c.tolist() if for_json else c,
        "q": float(q) if for_json else q,
    }


def binarise_weight_matrix(mat: Union[np.ndarray, T], thres_prop):
    """
    Turn weight matrix into binary adjacency matrix
    """
    try:
        abs_mat = mat.abs()
    except AttributeError:
        abs_mat = abs(mat)
    numel = np.prod(abs_mat.shape)
    thres_idx = numel * thres_prop
    thres_weight = np.sort(abs_mat.flatten())[-int(thres_idx)]
    return abs_mat >= thres_weight


def random_matched_graph(mat: Union[np.ndarray, T]):
    """
    Same number of nodes and edges
    """
    # Checks and counts
    num_nodes = mat.shape[0]
    assert num_nodes == mat.shape[1]  # ensure mat is square
    assert set(np.unique(mat)) in [{1, 0}, {True, False}]  # ensure mat is binary

    num_edges = mat.sum().item()

    # Choose active elements
    active_nodes = np.random.choice(
        (num_nodes * num_nodes), size=num_edges, replace=False
    )
    x = active_nodes // num_nodes
    y = active_nodes % num_nodes

    # Generate random graph
    canvas = np.zeros_like(mat)
    for x_i, y_i in zip(x, y):
        canvas[x_i, y_i] = 1
    if isinstance(mat, T):
        canvas = torch.tensor(canvas)
    return canvas


def small_worldness(param, thres_prop, num_random_graphs):

    if isinstance(param, T):
        param = param.numpy()

    adj_mat_full = binarise_weight_matrix(param, thres_prop).astype(int)
    indexer = (adj_mat_full.sum(0) > 1) & (adj_mat_full.sum(1) > 1)
    adj_mat = adj_mat_full[indexer][:, indexer]
    # adj_mat = binarise_weight_matrix(param, thres_prop).astype(int)

    random_graph_supply = [
        random_matched_graph(adj_mat) for _ in range(num_random_graphs)
    ]

    # fig, axes = plt.subplots(3, 2)
    # axes[0, 0].hist(abs(param).flatten(), 50)
    # axes[0, 1].imshow(abs(param));              axes[0, 0].set_title('abs param value')
    # axes[1, 0].imshow(adj_mat_full);            axes[1, 0].set_title(f'{thres_prop} cutoff (adj_map_full)')
    # axes[2, 0].imshow(adj_mat);                 axes[2, 0].set_title('reduced adj_map')
    # axes[2, 1].imshow(random_graph_supply[0]);  axes[2, 1].set_title('random matched graph')
    # fig.savefig('param.png')

    C = bct.clustering_coef_bd(adj_mat.astype(float)).mean()
    Cr = 0
    for random_graph in random_graph_supply:
        Cr += (
            bct.clustering_coef_bd(random_graph.astype(float)).mean()
            / num_random_graphs
        )

    dist_mat = bct.distance_bin(
        adj_mat
    )  # TODO - is the characteristic distance also on the binarised graph??
    L = bct.charpath(dist_mat, include_diagonal=False)[0]  # TODO - include diagonal?

    all_Lrs = []
    for random_graph in random_graph_supply:
        dist_mat_r = bct.distance_bin(random_graph)
        Lr_contrib = bct.charpath(dist_mat_r, include_diagonal=False)[0]
        if not np.isinf(Lr_contrib):
            all_Lrs.append(Lr_contrib)
    Lr = np.mean(all_Lrs)

    small_worldness_value = (C / Cr) / (L / Lr)

    return {
        "C": C,
        "L": L,
        "Cr": Cr,
        "Lr": Lr,
        "small_worldness": small_worldness_value,
        "eff_dim": adj_mat.shape[0],
    }


def weights_fundamentals(
    param: T, distances: T,
):
    weights = param.abs().flatten().numpy()
    diag_weights = param.diag().abs().flatten().numpy()
    distances = distances.abs().flatten().numpy()
    return {
        "average_abs_weight": float(weights.mean()),
        "average_abs_diag_weight": float(diag_weights.mean()),
        "weights_distances_correlation": float(np.corrcoef(distances, weights)[0, 1]),
    }
