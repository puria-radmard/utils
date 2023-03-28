"""
Utils for DNG
"""

import torch

def recurrent_mitosis(W, i):
    new_column = (W[:,i] * 0.5).unsqueeze(-1)
    W = torch.concat([W[:,:i], new_column, new_column, W[:,i+1:]], axis=1)
    new_row = W[i].unsqueeze(0)
    W = torch.concat([W[:i], new_row, new_row, W[i+1:]], axis=0)
    return W


def feedforward_mitosis(W, i, neural_dimension = 0):
    num_neurons = W.shape[neural_dimension]
    new_slice = torch.narrow(W, neural_dimension, i, 1)
    W = torch.concat(
        [
            torch.narrow(W, neural_dimension, 0, i),
            new_slice,
            new_slice,
            torch.narrow(W, neural_dimension, i+1, num_neurons - i - 1),
        ],
        neural_dimension
    )
    return W


