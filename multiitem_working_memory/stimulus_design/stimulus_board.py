"""
Combining many stimuli (of multiple features) into a stimulus board that can 
"""

from __future__ import annotations

import random, torch
from typing import Tuple, List
from copy import deepcopy

from purias_utils.multiitem_working_memory.util.circle_utils import generate_circular_feature_list, norm_points
from purias_utils.multiitem_working_memory.stimulus_design.stimuli_features import *


class Stimulus:

    "TODO: RENAME THIS TO 2D STIMULUS - SEE THE generate_slice METHOD and total_size ARGUMENT FOR __init__"

    set_size: int = 1
    num_feature_dims: int
    total_size: float
    feature_order = ['colour', 'orientation']
    
    def __init__(self, *args: FeatureBase, total_size = 3):  # Total size is like diameter
        self.num_feature_dims = len(args)
        self.features = {f.name: f for f in args}
        self.total_size = total_size
    
    @property
    def stimuli(self):
        return [self]
    
    def generate_image(self, ):
        canvas = torch.zeros(self.total_size, self.total_size, 3)
        for feature_name in self.feature_order:
            if feature_name in self.features:
                canvas = self.features[feature_name].alter_image(canvas)
        return canvas

    def generate_slice(self):
        x, y = self.features['location'].value
        x_slice = slice(int(x - self.total_size/2), int(x + self.total_size/2))
        y_slice = slice(int(y - self.total_size/2), int(y + self.total_size/2))
        return x_slice, y_slice

    def change(self, feature_name, *args, **kwargs) -> Stimulus:
        """I think 'changed' instance will share most of its features with this instance"""
        new_stim = deepcopy(self)
        new_stim.features[feature_name] = new_stim.features[feature_name].change(*args, **kwargs)
        return new_stim
    
    def set(self, feature_name, *args, **kwargs) -> Stimulus:
        new_stim = deepcopy(self)
        new_stim.features[feature_name] = new_stim.features[feature_name].set(*args, **kwargs)
        return new_stim

    def generate_image_board(self):
        raise Exception('generate_image_board is deprecated. Use classes in purias_utils.multiitem_working_memory.generate_displays instead.')

    def feature_batch(self, feature_name):
        torch.tensor(self.features[feature_name].value)
        feature_batch = torch.tensor(
            [stim.features[feature_name].value for stim in self.stimuli]
        ).unsqueeze(-1)
        return feature_batch



class MultiOrientationStimulus(Stimulus):

    set_size: int = 1
    num_feature_dims: int
    total_size: float = 0.0
    
    def __init__(self, *args: CircularFeatureBase):
        self.num_feature_dims = len(args)
        self.features = [Orientation(norm_points(feature_value)) for feature_value in args]

    @property
    def stimuli(self):
        return [self]

    def generate_image(self): raise NotImplementedError

    def generate_slice(self): raise NotImplementedError

    def change(self, feature_idx, *args, **kwargs) -> Stimulus:
        new_stim = deepcopy(self)
        new_stim.features[feature_idx] = new_stim.features[feature_idx].change(*args, **kwargs)
        return new_stim

    def set(self, feature_idx, *args, **kwargs) -> Stimulus:
        new_stim = deepcopy(self)
        new_stim.features[feature_idx] = new_stim.features[feature_idx].set(*args, **kwargs)
        return new_stim

    def generate_image_board(self):
        raise Exception('generate_image_board is deprecated. Use classes in purias_utils.multiitem_working_memory.generate_displays instead.')

    def feature_batch(self, feature_idx):
        torch.tensor(self.features[feature_idx].value)
        feature_batch = torch.tensor(
            [stim.features[feature_idx].value for stim in self.stimuli]
        ).unsqueeze(-1)
        return feature_batch



class StimulusBoardBase:

    set_size: int
    num_feature_dims: int
    stimuli: List[Stimulus]
    board_size: float
    background: Tuple[float] = (0.5, 0.5, 0.5)

    def generate_image_board(self):
        raise Exception('generate_image_board is deprecated. Use classes in purias_utils.multiitem_working_memory.generate_displays instead.')
        canvas = torch.zeros(*self.board_size, 3)
        for i in range(3):
            canvas[:,:,i] = self.background[i]
        for stim in self.stimuli:
            stim_image = stim.generate_image()
            stim_slices = stim.generate_slice()
            canvas[stim_slices[0], stim_slices[1]] = stim_image
        return canvas
    
    def feature_batch(self, feature_name):
        feature_batch = torch.tensor([stim.features[feature_name].value for stim in self.stimuli])
        if len(feature_batch.shape) == 1:
            feature_batch = feature_batch.unsqueeze(-1)
        return feature_batch

    def copy(self):
        # TODO: check this! i.e. stimuli now independent?
        return deepcopy(self)

    def make_cue_board(self, idx):
        raise NotImplementedError
    
    def make_all_cue_boards(self):
        return [self.make_cue_board(i) for i in range(self.set_size)]



class ColouredSquaresBoard(StimulusBoardBase):
    "2D coloured stimulus display, randomly generated"
    def __init__(self, set_size, board_x_size = 32, board_y_size = 32, stim_size = 3, loc_border = 2.5, col_border = 0.0, background = (0.5, 0.5, 0.5)) -> None:
        
        self.loc_border = loc_border
        self.stim_size = stim_size
        self.board_size = (board_x_size, board_y_size)
        self.num_feature_dims = 2

        self.stimuli = []
        self.background = background

        stim_radius = stim_size / 2
        if col_border == 0.0:
            colours = [random.random() * 2 * pi for _ in range(set_size)]
        else:
            colours = generate_circular_feature_list(num_stim = set_size, feature_border = col_border)
        

        # for _ in range(set_size):
        while len(self.stimuli) < set_size:
            ver = False
            breaker = False
            counter = 0
            while not ver:
                # new_location = Location(random_location(stim_radius, self.board_size - stim_radius))
                new_location = Location(random_location(stim_radius, self.board_size[0] - stim_radius, stim_radius, self.board_size[1] - stim_radius))
                ver = self.verify_location(new_location)
                counter += 1
                if counter > 1000:  # This should definitely get fixed....
                    breaker = True
                    break
            if breaker:
                # break
                self.stimuli = []
            new_colour = Colour(colours[len(self.stimuli) - 1])
            new_stim = Stimulus(new_location, new_colour, total_size=stim_size)
            self.stimuli.append(new_stim)

        self.set_size = len(self.stimuli)

    @classmethod
    def init_from_features(
        cls,
        colours: list,
        x_locs: list,
        y_locs: list,
        board_x_size = 32, 
        board_y_size = 32, 
        stim_size = 3, 
        loc_border = 2.5, 
        background = (0.5, 0.5, 0.5)
    ) -> None:

        board = cls(
            set_size = 0,
            board_x_size = board_x_size,
            board_y_size = board_y_size,
            stim_size = stim_size,
            loc_border = loc_border,
            background = background
        )

        assert board.stimuli == []

        for c, x, y in zip(colours, x_locs, y_locs):
            new_loc = Location((x, y))
            new_col = Colour(c)
            assert board.verify_location(new_loc=new_loc)
            board.stimuli.append(Stimulus(new_loc, new_col, total_size=stim_size))

        board.set_size = len(board.stimuli)
        
        return board

    # This should definitely get fixed....
    def verify_location(self, new_loc: Location):
        for stim in self.stimuli:
            distances = new_loc.center_distances(stim.features['location'])
            euc_distance = (distances[0]**2 +  distances[1]**2) **0.5
            if euc_distance < self.loc_border + self.stim_size:
                return False
        return True

    def make_cue_board(self, idx):
        copied_board = self.copy()
        for i in range(len(copied_board.stimuli)):
            # ALL of these should be changed, no clues given
            changed_stim = copied_board.stimuli[i].change('colour', float('nan'))
            changed_stim = changed_stim.set('location', *copied_board.stimuli[idx].features['location'].value)
            copied_board.stimuli[i] = changed_stim
        return copied_board



class MultiOrientationStimulusBoard(StimulusBoardBase):
    """
        A board where all features of all objects are orientations (even loc),
            and any one of them can be cued
        TODO: implement stim border for orientations
    """

    def __init__(self, set_size: int, num_features: int, feature_borders: list, **kwargs):

        self.feature_borders = feature_borders
        self.num_feature_dims = num_features
        self.set_size = set_size
        
        all_feature_lists = [
            generate_circular_feature_list(num_stim = self.set_size, feature_border = feature_borders[i])
            for i in range(num_features)
        ] # Makes sure feature_border is not violated

        self.stimuli = [
            MultiOrientationStimulus(*[features[i] for features in all_feature_lists])
            for i in range(self.set_size)
        ]

    @classmethod
    def init_from_features(cls, list_of_features):
        num_features = len(list_of_features)
        num_stimuli = len(list_of_features[0])
        assert all([len(lof) == num_stimuli for lof in list_of_features])
        instance = cls(0, num_features, [0.0 for _ in range(num_features)])
        instance.set_size = num_stimuli
        instance.stimuli = [
            MultiOrientationStimulus(*[features[i] for features in list_of_features])
            for i in range(num_stimuli)
        ]
        return instance

    def generate_image_board(self):
        raise Exception('generate_image_board is deprecated. Use classes in purias_utils.multiitem_working_memory.generate_displays instead.')

    def make_cue_board(self, idx: int, target_feature_idx: int):
        """
            Make a board where the only stimulus has:
                target feature set to NaN (to be dealt with by sensory population)
                all other features set to the idx^th stimulus' value
        """
        features = [[f.value] for f in self.stimuli[idx].features]
        features[target_feature_idx] = [float('nan')]
        return MultiOrientationStimulusBoard.init_from_features(features)
        # copied_board = self.copy()
        # cued_feature_values = {
        #     i: copied_board.stimuli[idx].features[i].value 
        #     for i in range(self.num_feature_dims) if i != target_feature_idx
        # }
        # for i in range(len(copied_board.stimuli)):
        #     changed_stim = copied_board.stimuli[i]
        #     for j in range(changed_stim.num_feature_dims):
        #         if j == target_feature_idx:
        #             changed_stim = changed_stim.set(j, float('nan'))
        #         else:
        #             changed_stim = changed_stim.set(j, cued_feature_values[j])
        #     copied_board.stimuli[i] = changed_stim
        # return copied_board
    
    def make_all_cue_boards(self, target_feature_idx: int):
        return [self.make_cue_board(i, target_feature_idx) for i in range(self.set_size)]

    def full_feature_batch(self):
        "Of shape [N, D]"
        dimensional_feature_batches = torch.concat([self.feature_batch(i) for i in range(self.num_feature_dims)], -1)
        return dimensional_feature_batches

