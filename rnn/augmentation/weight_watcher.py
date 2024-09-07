from __future__ import annotations

from abc import ABC, abstractmethod
import random

import torch
from torch import Tensor as _T

types_dict = {
    "last": 'LastSplitWW',
    "rnd": 'LargestRandomSplitWW',
    "big_final": 'LargestFinalMagnitudeWW',
    "big_var": 'LargestHistorysVarianceWW',
    "big_traj": 'LargestTrajectoryWW',
    "big_del": 'LargestOverallMagChangeWW',
    "small_final": 'SmallestFinalMagnitudeWW',
    "small_var": 'SmallestHistorysVarianceWW',
    "small_traj": 'SmallestTrajectoryWW',
    "small_del": 'SmallestOverallMagChangeWW',
}


class WeightWatcher(ABC):
    """
    Everytime a weight is changed, register it here.
    At the end of the training round, i.e. when DNG is about to occur, provide a list of
        viable indices and an axis to check along, and this will return the index of the weight
        which should be split.

    NB: at the moment these are only equipped for 2 axis weights
    """
    def __init__(self, initial_weight: _T, stats_dimension: int) -> None:
        self.weight_shape = torch.tensor(initial_weight.shape)
        assert 0 <= stats_dimension < len(self.weight_shape)
        self.stats_dimension = stats_dimension
    
    def register_weight(self, weight_matrix: _T) -> None:
        assert all(torch.tensor(weight_matrix.shape) == self.weight_shape)
        self.__register_weight__(weight_matrix)

    def reccomend_index(self, viable_indices: list) -> None:
        num_neurons = self.weight_shape[self.stats_dimension]
        assert all([0 <= vi < num_neurons for vi in viable_indices])
        self.weight_shape[self.stats_dimension] += 1   # in case we are reusing this instance
        return self.__reccomend_index__(viable_indices)


    @abstractmethod
    def __register_weight__(self, weight_matrix: _T) -> None:
        ...

    @abstractmethod
    def __reccomend_index__(self, viable_indices: list) -> None:
        ...


class LastSplitWW(WeightWatcher):
    """Just select the last viable cell to split, no memory needed"""
    def __register_weight__(self, weight_matrix: _T) -> None:
        pass

    def __reccomend_index__(self, viable_indices: list) -> None:
        return max(viable_indices)

    
class LargestRandomSplitWW(WeightWatcher):
    """Just select a random viable cell to split, no memory needed"""
    def __register_weight__(self, weight_matrix: _T) -> None:
        pass
    
    def __reccomend_index__(self, viable_indices: list) -> None:
        return random.choice(viable_indices)


class LargestFinalMagnitudeWW(WeightWatcher):
    """Just select a random viable cell to split, no memory needed"""
    def __register_weight__(self, weight_matrix: _T) -> None:
        self.latest_weight = weight_matrix.detach().cpu().data

    def selector(self, stats: _T) -> _T:
        return stats.argmax().item()
    
    def __reccomend_index__(self, viable_indices: list) -> None:
        abs_weights = self.latest_weight.abs()
        weights_checked = {i: self.selector(torch.narrow(
            abs_weights, self.stats_dimension, i, 1
        ).sum(0)) for i in viable_indices}
        return max(weights_checked.keys(), key=weights_checked.get)


class LargestHistorysVarianceWW(WeightWatcher):
    """Select the weight with the largest variance over the course of the training"""

    weight_history = []

    def __register_weight__(self, weight_matrix: _T) -> None:
        self.weight_history.append(weight_matrix.detach().cpu().data)

    def selector(self, stats: _T) -> _T:
        return stats.argmax().item()
    
    def __reccomend_index__(self, viable_indices: list) -> None:
        weight_vars = torch.stack(self.weight_history, 0).var(0)
        vars_checked = {i: self.selector(torch.narrow(
            weight_vars, self.stats_dimension, i, 1
        ).sum(0)) for i in viable_indices}
        return max(vars_checked.keys(), key=vars_checked.get)

    
class LargestTrajectoryWW(WeightWatcher):
    """Select the weight which had the largest summed magnitude deltas over course of training"""

    def __init__(self, initial_weight: _T, stats_dimension: int) -> None:
        super().__init__(initial_weight, stats_dimension)
        self.previous_weight = initial_weight.detach().cpu().data
        self.delta_history = []

    def __register_weight__(self, weight_matrix: _T) -> None:
        new_weight = weight_matrix.detach().cpu().data
        self.delta_history.append((new_weight - self.previous_weight).abs())
        self.previous_weight = new_weight

    def selector(self, stats: _T) -> _T:
        return stats.argmax().item()
    
    def __reccomend_index__(self, viable_indices: list) -> None:
        unravelled_trajectory = torch.stack(self.delta_history, 0).sum(0)
        trajectories_checked = {i: self.selector(torch.narrow(
            unravelled_trajectory, self.stats_dimension, i, 1
        ).sum(0)) for i in viable_indices}
        return max(trajectories_checked.keys(), key=trajectories_checked.get)

        
    
class LargestOverallMagChangeWW(LargestTrajectoryWW):
    """Select the weight which had the largest overall magnitude change over course of training"""

    def __init__(self, initial_weight: _T, stats_dimension: int) -> None:
        super().__init__(initial_weight, stats_dimension)
        self.start_weight = initial_weight.detach().cpu().data
        self.latest_weight: _T = None

    def __register_weight__(self, weight_matrix: _T) -> None:
        self.latest_weight = weight_matrix.detach().cpu().data

    def selector(self, stats: _T) -> _T:
        return stats.argmax().item()
    
    def __reccomend_index__(self, viable_indices: list) -> None:
        overall_delta = (self.latest_weight - self.latest_weight).abs()
        deltas_checked = {i: self.selector(torch.narrow(
            overall_delta, self.stats_dimension, i, 1
        ).sum(0)) for i in viable_indices}
        return max(deltas_checked.keys(), key=deltas_checked.get)

class SmallestFinalMagnitudeWW(LargestFinalMagnitudeWW):
    def selector(self, stats: _T) -> _T:
        return stats.argmin().item()

class SmallestHistorysVarianceWW(LargestHistorysVarianceWW):
    def selector(self, stats: _T) -> _T:
        return stats.argmin().item()

class SmallestTrajectoryWW(LargestTrajectoryWW):
    def selector(self, stats: _T) -> _T:
        return stats.argmin().item()

class SmallestOverallMagChangeWW(LargestOverallMagChangeWW):
    def selector(self, stats: _T) -> _T:
        return stats.argmin().item()



