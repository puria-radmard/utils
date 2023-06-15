"""
Sensory neurons have tuning curves in each feature dimension, and each ones response
gets multiplied.
"""

from torch import nn
from purias_utils.multiitem_working_memory.stimuli import *
from purias_utils.multiitem_working_memory.stimulus_board import *


class SensoryNeuronPopulationBase(nn.Module):
    population_size: int
    feature_name: str
    def __init__(self) -> None: ...
    def format_features(self, board: StimulusBoardBase) -> None: ...
    def forward(self) -> None: ...


class BoardLocationDescriptor(SensoryNeuronPopulationBase):
    """
    Just normalise to -1,1 and give it as is.
    TODO: learned augmentations of this space perhaps?
    """
    def __init__(self, board_size: float) -> None:
        super(SensoryNeuronPopulationBase, self).__init__()

        self.population_size = 2
        self.width = board_size

    def forward(self, boards: List[StimulusBoardBase]):
        locations = torch.stack([board.feature_batch('location') for board in boards], 0)        


class BoardLocationNeuronPopulation(SensoryNeuronPopulationBase):
    """
    Regularily spaced, equal width Gaussian tuning curves along x and y coords of a board.
    'Circular' width, i.e. equal in both dimensions
    """
    def __init__(self, side_n: int, board_size: float, width: float) -> None:
        super(SensoryNeuronPopulationBase, self).__init__()

        self.side_n = side_n
        self.population_size = side_n ** 2
        self.width = width

        distance = board_size / (side_n - 1)  # Two bumps centered on boundaries
        self.x_locs = torch.tensor([i * distance for i in range(side_n)])
        self.y_locs = torch.tensor([i * distance for i in range(side_n)])

        self.tuning_curve = torch.distributions.normal.Normal(loc = 0, scale = width)
        
    def format_features(self, board: StimulusBoardBase) -> None:
        return torch.tensor(board.feature_batch('location'))

    def forward(self, boards: List[StimulusBoardBase]):
        "Excess stimuli in each batch are returned as 0, which should be dealt with in downstream sensory masking"

        # TODO: GENERALISE THIS FUNCTIONALITY
        max_set_size = max([board.set_size for board in boards])
        mask = torch.zeros(len(boards), max_set_size, 1).long()
        locations = torch.zeros(len(boards), max_set_size, 2).long()
        for j, board in enumerate(boards):
            mask[j, :board.set_size] = 1.
            locations[j, :board.set_size] = board.feature_batch('location')
        
        x_diffs = locations[:,:,0] - self.x_locs.reshape(-1, 1, 1)    # [bumps, batch, stim]
        y_diffs = locations[:,:,1] - self.y_locs.reshape(-1, 1, 1)

        x_acts = self.tuning_curve.log_prob(x_diffs).exp().permute(1, 2, 0) # [batch, bumps, stim]
        y_acts = self.tuning_curve.log_prob(y_diffs).exp().permute(1, 2, 0)

        all_acts = (x_acts.unsqueeze(-1) * y_acts.unsqueeze(2)) # [batch, stim, bumps, bumps]
        all_acts = all_acts.reshape(*all_acts.shape[:2], -1)  # [batch, stim, bumps * bumps]

        return all_acts * mask


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

    def forward(self, boards: List[StimulusBoardBase]):
        # TODO: GENERALISE THIS FUNCTIONALITY
        max_set_size = max([board.set_size for board in boards])
        mask = torch.zeros(len(boards), max_set_size, 1).long()
        feature_batches = torch.zeros(len(boards), max_set_size, 1).long()
        for j, board in enumerate(boards):
            mask[j, :board.set_size] = 1.
            feature_batches[j, :board.set_size] = board.feature_batch(self.feature_name)
        cosines = torch.cos(self.locs - feature_batches)  # [batch, stim, bumps]
        tuning = torch.exp(5 * cosines - 2.5) # before dealing with nans
        if torch.isnan(tuning).any():
            tuning = tuning.nan_to_num(np.exp(2.5))
        return tuning


class MultiFeatureNeuronPopulationSet(nn.Module):
    """
    Holds neuron populations of sizes n_1, n_2, ..., n_M
    Total resulting population will be of size n_1*n_2*...*n_M
    """
    def __init__(self, *populations: SensoryNeuronPopulationBase) -> None:
        super(MultiFeatureNeuronPopulationSet, self).__init__()

        self.populations = populations
        self.population_size = 1
        for population in self.populations:
            self.population_size *= population.population_size

    def forward(self, board: StimulusBoardBase):
        all_activations = [pop(board) for pop in self.populations]  # shapes [stim, n_i]
        combined_activation = all_activations[0]
        for pop_act in all_activations[1:]:
            inter_combined_activation = combined_activation.unsqueeze(3) @ pop_act.unsqueeze(2) # [batch, stim, existing_prod_n, new_n]
            combined_activation = inter_combined_activation.reshape(*inter_combined_activation.shape[:2], -1)
        return combined_activation

