"""
Sensory neurons have tuning curves in each feature dimension, and each ones response
gets multiplied.
"""

from torch import nn, Tensor as T
from purias_utils.multiitem_working_memory.stimuli import *
from purias_utils.multiitem_working_memory.stimulus_board import *


class SensoryNeuronPopulationBase(nn.Module):
    population_size: int
    feature_name: str
    def __init__(self) -> None: ...
    def format_features(self, board: StimulusBoardBase) -> None: ...
    def forward(self) -> None: ...
    def generate_grid(self, *counts) -> T:
        """
        Generate full grid of values, with `counts` gridlines along each relevant dimension. 
        Pass through self.forward to get equivalent tuning curves
        """

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

        self.tuning_curve = torch.distributions.normal.Normal(loc = 0, scale = width)

    def generate_grid(self, x_count, y_count) -> T:
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
        
    def format_features(self, board: StimulusBoardBase) -> None:
        return torch.tensor(board.feature_batch('location'))

    def forward_from_features(self, locations, mask = 1.0):
        x_diffs = locations[:,:,0] - self.x_locs.reshape(-1, 1, 1)    # [bumps, batch, stim]
        y_diffs = locations[:,:,1] - self.y_locs.reshape(-1, 1, 1)

        x_acts = self.tuning_curve.log_prob(x_diffs).exp().permute(1, 2, 0) # [batch, bumps, stim]
        y_acts = self.tuning_curve.log_prob(y_diffs).exp().permute(1, 2, 0)

        all_acts = (x_acts.unsqueeze(-1) * y_acts.unsqueeze(2)) # [batch, stim, bumps, bumps]
        all_acts = all_acts.reshape(*all_acts.shape[:2], -1)  # [batch, stim, bumps * bumps]

        return all_acts * mask

    def forward(self, boards: List[StimulusBoardBase]):
        "Excess stimuli in each batch are returned as 0, which should be dealt with in downstream sensory masking"
        # TODO: GENERALISE THIS FUNCTIONALITY
        max_set_size = max([board.set_size for board in boards])
        mask = torch.zeros(len(boards), max_set_size, 1).float()
        locations = torch.zeros(len(boards), max_set_size, 2).float()
        for j, board in enumerate(boards):
            mask[j, :board.set_size] = 1.
            locations[j, :board.set_size] = board.feature_batch('location')
        return self.forward_from_features(locations, mask)



class CircularFeatureNeuronPopulation(SensoryNeuronPopulationBase):
    """
    Regularily spaced, equal width exp^cos tuning curves around a circle
    """
    def __init__(self, n: int, feature_name: str) -> None:
        super(SensoryNeuronPopulationBase, self).__init__()

        distance = 2 * pi / n # No repeat on theta = 0
        self.locs = torch.tensor([i * distance for i in range(n)])
        self.population_size = n
        self.feature_name = feature_name
    
    def generate_grid(self, circle_count) -> T:
        distance = 2 * pi / circle_count # No repeat on theta = 0
        return torch.tensor([i * distance for i in range(circle_count)])

    def forward_from_features(self, oris, mask = 1.0):
        cosines = torch.cos(self.locs - oris)  # [batch, stim, bumps]
        # tuning = torch.exp(5 * cosines - 2.5) # before dealing with nans
        tuning = torch.exp(cosines + 1) / 4 # before dealing with nans
        if torch.isnan(tuning).any():
            tuning = tuning.nan_to_num(np.exp(2) / 4)
        return tuning * mask

    def forward(self, boards: List[StimulusBoardBase]):
        # TODO: GENERALISE THIS FUNCTIONALITY
        max_set_size = max([board.set_size for board in boards])
        mask = torch.zeros(len(boards), max_set_size, 1).float()
        feature_batches = torch.zeros(len(boards), max_set_size, 1).float()
        for j, board in enumerate(boards):
            mask[j, :board.set_size] = 1.
            feature_batches[j, :board.set_size] = board.feature_batch(self.feature_name)
        return self.forward_from_features(feature_batches, mask)



class MultiFeatureNeuronPopulationSet(nn.Module):
    """
    Holds neuron populations of sizes n_1, n_2, ..., n_M
    Total resulting population will be of size n_1*n_2*...*n_M
    """
    def __init__(self, *populations: SensoryNeuronPopulationBase) -> None:
        super(MultiFeatureNeuronPopulationSet, self).__init__()

        self.populations = nn.ModuleList(populations)
        self.population_size = 1
        for population in self.populations:
            self.population_size *= population.population_size

    @staticmethod
    def combine_activations(all_acts):
        combined_activation = all_acts[0]
        for pop_act in all_acts[1:]:
            inter_combined_activation = combined_activation.unsqueeze(3) @ pop_act.unsqueeze(2) # [batch, stim, existing_prod_n, new_n]
            combined_activation = inter_combined_activation.reshape(*inter_combined_activation.shape[:2], -1)
        return combined_activation  

    def forward_from_features(self, feature_set: List[T], mask = 1.0):
        all_activations = [pop.forward_from_features(features) for pop, features in zip(self.populations, feature_set)]
        return self.combine_activations(all_activations)

    def forward(self, board: StimulusBoardBase):
        all_activations = [pop(board) for pop in self.populations]  # shapes [stim, n_i]
        return self.combine_activations(all_activations)

