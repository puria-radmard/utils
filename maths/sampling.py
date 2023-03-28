import torch
from torch import Tensor as T
from torch.nn import functional as F


def time_meaned_trial_covariance(trial_samples: T):

    assert len(trial_samples.shape) == 4, "need timeseries_covariance input to be of shape [batch, time, neurons, trials]"

    B, T, N, trials = trial_samples.shape

    total_cov = torch.zeros([B, N, N]).to(trial_samples.device)

    for time_item in trial_samples.permute(1, 0, 3, 2):  # [batchsize, trials, N]

        trial_mean = time_item.mean(1, keepdim=True)
        spread = time_item - trial_mean
        flattened_spread = spread.reshape(B*trials, N)
        flattened_spread_squared = torch.bmm(flattened_spread.unsqueeze(2), flattened_spread.unsqueeze(1))
        cov = flattened_spread_squared.reshape(B, trials, N, N).sum(dim=1) / (trials - 1)
        cov = cov.reshape(B, N, N)
        total_cov += cov / float(T)

    return total_cov
