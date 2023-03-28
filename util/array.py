import itertools
import torch, math
from torch import Tensor as T

from typing import List

def gaussian_on_a_circle(num_bins: int, mean_bins: T, std_angles: T, scales: T):

    assert mean_bins.shape == std_angles.shape == scales.shape
    assert len(mean_bins.shape) == 1

    num_bumps = mean_bins.shape[0]
    theta = torch.linspace(-math.pi, math.pi, num_bins).unsqueeze(0).repeat(num_bumps, 1)  # [num_bumps, num_bins]
    var = (std_angles ** 2).unsqueeze(-1)
    log_std = std_angles.log().unsqueeze(-1)
    gaussians = -(theta ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
    gaussians = gaussians.exp() # central at first

    for i in range(gaussians.shape[0]):
        rolled = torch.roll(gaussians[i], mean_bins[i].item())
        gaussians[i] = scales[i] * rolled

    return gaussians


def grid_points(counts: List[int], lowers: List[int], uppers: List[int]) -> T:

    linspaces = [
        torch.linspace(lower, upper, count).tolist() for lower, upper, count in
        zip(lowers, uppers, counts)
    ]

    return torch.tensor(list(itertools.product(*linspaces)))

