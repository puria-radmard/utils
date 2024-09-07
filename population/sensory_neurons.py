"""
Sensory neurons have tuning curves in each feature dimension, and each ones response
gets multiplied.
"""

from torch import nn
from torch import Tensor as _T
from purias_utils.multiitem_working_memory.stimulus_design.stimuli_features import *
from purias_utils.multiitem_working_memory.stimulus_design.stimulus_board import *


class SensoryNeuronPopulationBase(nn.Module):
    population_size: int
    feature_name: str
    def __init__(self) -> None: ...
    def generate_features(self, boards: List[StimulusBoardBase]) -> _T: ...
    def forward(self) -> None: ...
    def forward_from_features(self) -> None: ...
    def generate_grid(self, *counts) -> _T:
        """
        Generate full grid of values, with `counts` gridlines along each relevant dimension. 
        Pass through self.forward to get equivalent tuning curves
        """
    def check_set_size(self, boards):#, max_stim):
        set_sizes = [board.set_size for board in boards]
        assert all([set_sizes[0] == ss for ss in set_sizes])
        return set_sizes[0]
        max_set_size = max([board.set_size for board in boards])
        if max_stim is not None:
            assert max_stim >= max_set_size
            max_set_size = max_stim
        return max_set_size


class BoardLocationNeuronPopulation(SensoryNeuronPopulationBase):
    """
    Regularily spaced, equal width Gaussian tuning curves along x and y coords of a board.
    'Circular' width, i.e. equal in both dimensions
    """
    def __init__(self, x_side_n: int, y_side_n: float, board_x_size: float, board_y_size: float, width: float) -> None:
        super(SensoryNeuronPopulationBase, self).__init__()

        self.x_side_n = x_side_n
        self.y_side_n = y_side_n
        self.population_size = self.x_side_n * self.y_side_n
        self.width = width

        self.board_x_size = board_x_size
        self.board_y_size = board_y_size

        self.x_locs = self.generate_center_locations(board_x_size, x_side_n)
        self.y_locs = self.generate_center_locations(board_y_size, y_side_n)

        self.tuning_curve = torch.distributions.normal.Normal(loc = 0, scale = self.width)

    def generate_grid(self, x_count, y_count) -> _T:
        x_locs = self.generate_center_locations(self.board_x_size, x_count)
        y_locs = self.generate_center_locations(self.board_y_size, y_count)
        grid_locs = torch.cat(torch.meshgrid(x_locs, y_locs), 1)
        return grid_locs

    @staticmethod
    def generate_center_locations(board_dim, num_bumps):
        if num_bumps > 1:
            distance = board_dim / (num_bumps - 1)  # Two bumps centered on boundaries
            return torch.tensor([i * distance for i in range(num_bumps)])
        else:
            return torch.tensor([board_dim / 2])  # central bump

    def generate_features(self, boards: List[StimulusBoardBase]) -> _T:
        set_size = self.check_set_size(boards)
        locations = torch.zeros(len(boards), set_size, 2).float()
        for j, board in enumerate(boards):
            locations[j, :board.set_size] = board.feature_batch('location')
        return torch.stack(locations, 0)

    def forward_from_features(self, locations):
        "Locations come in shape [..., stim count, 2 (x, y)]"
        spare_dims = len(locations.shape) - 2
        _shape = [-1] + ([1] * (spare_dims + 1))
        x_diffs = locations[..., 0] - self.x_locs.reshape(*_shape)    # [bumps, ..., stim]
        y_diffs = locations[..., 1] - self.y_locs.reshape(*_shape)    # [bumps, ..., stim]

        x_acts = torch.moveaxis(self.tuning_curve.log_prob(x_diffs).exp(), 0, -2) # [..., bumps, stim]
        y_acts = torch.moveaxis(self.tuning_curve.log_prob(y_diffs).exp(), 0, -2)

        all_acts = (x_acts.unsqueeze(-2) * y_acts.unsqueeze(-3)) # [..., bumps_x, bumps_y, stim]
        all_acts = torch.moveaxis(all_acts, -1, spare_dims)
        all_acts = all_acts.reshape(*all_acts.shape[:-2], -1)  # [..., stim, bumps * bumps]

        return all_acts

    def forward(self, boards: List[StimulusBoardBase]):
        # TODO: GENERALISE THIS FUNCTIONALITY
        locations = self.generate_features(boards)
        return self.forward_from_features(locations)



class CircularFeatureNeuronPopulation(SensoryNeuronPopulationBase):
    """
    Regularily spaced, equal width exp^(width * cos(theta)) tuning curves around a circle
    Width is actually 1/concentration of the vonMises
    """
    def __init__(self, n: int, feature_name: str, width: float) -> None:
        super(SensoryNeuronPopulationBase, self).__init__()

        distance = 2 * pi / n # No repeat on theta = 0
        self.locs = torch.tensor([i * distance for i in range(n)])
        self.population_size = n
        self.feature_name = feature_name
        self.width = width

        if self.width == 'original':
            raise Exception('Deprecated')
            #self.tuning_curve = None
            #nan_val = torch.exp(torch.tensor(2.0)) / 4
        else:
            self.tuning_curve = torch.distributions.VonMises(loc = 0, concentration = 1/self.width)
            log_nan_val = self.tuning_curve.log_prob(torch.tensor(0))    # The stand-in value for cued representation coding, i.e. when sensitivity is removed

        self.register_parameter('log_nan_val', nn.Parameter(torch.tensor(log_nan_val), requires_grad=True))
    
    def generate_grid(self, circle_count) -> _T:
        distance = 2 * pi / circle_count # No repeat on theta = 0
        return torch.tensor([i * distance for i in range(circle_count)])

    def generate_features(self, boards: List[StimulusBoardBase]):
        set_size = self.check_set_size(boards)
        feature_batches = torch.zeros(len(boards), set_size, 1).float()
        for j, board in enumerate(boards):
            feature_batches[j, :board.set_size] = board.feature_batch(self.feature_name)
        return feature_batches

    def forward_from_features(self, oris):
        "Orientations come in shape [..., stim count, 1 (angle)]"

        spare_dims = len(oris.shape) - 2
        _shape = [-1] + ([1] * (spare_dims + 1))

        if self.width == 'original':
            raise Exception
            cosines = torch.cos(self.locs - oris)  # [batch, stim, bumps]
            all_acts = torch.exp(cosines + 1) / 4
        else:
            ori_diffs = oris[..., 0] - self.locs.reshape(*_shape)    # [bumps, ..., stim]
            modded_ori_diffs = ori_diffs.nan_to_num(0.0)             # make computable but remember where they were
            all_acts = self.tuning_curve.log_prob(modded_ori_diffs).exp()
            nan_idx = torch.isnan(oris[..., 0].unsqueeze(0)).repeat_interleave(self.population_size, 0)
            all_acts[nan_idx] = self.log_nan_val.exp().to(all_acts.dtype)

        # This should work with the nan_to_num above, but just to be doubly sure...
        all_acts = torch.moveaxis(all_acts, 0, -2) # [..., bumps, stim]
        all_acts = torch.moveaxis(all_acts, -1, spare_dims)

        return all_acts

    def forward(self, boards: List[StimulusBoardBase]):
        # TODO: GENERALISE THIS FUNCTIONALITY
        feature_set = self.generate_features(boards)
        return self.forward_from_features(feature_set)

    def to(self, *args, **kwargs):
        self.locs = self.locs.to(*args, **kwargs)
        return super().to(*args, **kwargs)



class MultiFeatureNeuronPopulationSet(nn.Module):
    """
    Conjunctive tuning over multiple dimensions
    Holds neuron populations of sizes n_1, n_2, ..., n_D
    Total resulting population will be of size n_1*n_2*...*n_D
    """
    populations: List[SensoryNeuronPopulationBase]

    def __init__(self, *populations: SensoryNeuronPopulationBase) -> None:
        super(MultiFeatureNeuronPopulationSet, self).__init__()

        self.populations = nn.ModuleList(populations)
        self.population_size = 1
        for population in self.populations:
            self.population_size *= population.population_size
        self.num_features = len(self.populations)

    @classmethod
    def toroidal_conjunctive_set(cls, num_tuning_curves, tuning_curve_widths):
        num_pops = len(num_tuning_curves)
        assert len(tuning_curve_widths) == num_pops
        populations = []
        for i in range(num_pops):
            populations.append(CircularFeatureNeuronPopulation(num_tuning_curves[i], i, tuning_curve_widths[i]))
        return cls(*populations)

    @staticmethod
    def combine_activations(all_acts):
        combined_activation = all_acts[0]
        for pop_act in all_acts[1:]:
            inter_combined_activation = combined_activation.unsqueeze(-1) @ pop_act.unsqueeze(-2) # [..., stim, existing_prod_n, new_n]
            combined_activation = inter_combined_activation.reshape(*inter_combined_activation.shape[:-2], -1)
        return combined_activation  

    def generate_features(self, boards: List[StimulusBoardBase]) -> List[_T]:
        return torch.stack([pop.generate_features(boards=boards) for pop in self.populations], 0)

    def forward_from_features(self, feature_set: List[_T]):
        assert len(feature_set) == self.num_features
        all_activations = [pop.forward_from_features(fs) for fs, pop in zip(feature_set, self.populations)]  # shapes [stim, n_i]
        return self.combine_activations(all_activations)

    def forward(self, boards: List[StimulusBoardBase]) -> _T:
        all_activations = [pop(boards) for pop in self.populations] # [batch, stim, n] for each dimension
        return self.combine_activations(all_activations)    # XXX CHECK SHAPE

    def to(self, *args, **kwargs):
        for population in self.populations:
            population.to(*args, **kwargs)
        return super().to(*args, **kwargs)




class MixedMultiFeatureNeuronPopulationSet(MultiFeatureNeuronPopulationSet):
    """
    First saw this in Matthey et al., 2015
    Mixture of conjunctive tuning and individual feature coding, concatenated together
    """

    

