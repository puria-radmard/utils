"""
Combining many stimuli (of multiple features) into a stimulus board that can 
"""

from __future__ import annotations

import random, torch
from typing import Tuple, List
from copy import deepcopy

from purias_utils.multiitem_working_memory.stimuli import *


class Stimulus:

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
        raise NotImplementedError

    def feature_batch(self, feature_name):
        torch.tensor(self.features[feature_name].value)
        feature_batch = torch.tensor(
            [stim.features[feature_name].value for stim in self.stimuli]
        ).unsqueeze(-1)
        return feature_batch


class StimulusBoardBase:

    set_size: int
    num_feature_dims: int
    stimuli: List[Stimulus]
    board_size: float
    background: Tuple[float] = (0.5, 0.5, 0.5)

    def generate_image_board(self):
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
    "Simplest stimulus board, randomly generated"
    def __init__(self, set_lower, set_upper, board_x_size = 32, board_y_size = 32, stim_size = 3, loc_border = 2.5, background = (0.5, 0.5, 0.5)) -> None:
        
        self.loc_border = loc_border
        self.stim_size = stim_size
        self.board_size = (board_x_size, board_y_size)
        self.num_feature_dims = 2
        target_set_size = random.randint(set_lower, set_upper)

        self.stimuli = []
        self.background = background

        stim_radius = stim_size / 2

        # for _ in range(target_set_size):
        while len(self.stimuli) < target_set_size:
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
            new_colour = Colour(random_orientation())
            new_stim = Stimulus(new_location, new_colour, total_size=stim_size)
            self.stimuli.append(new_stim)

        self.set_size = len(self.stimuli)

    # This should definitely get fixed....
    def verify_location(self, new_loc: Location):
        for stim in self.stimuli:
            distances = new_loc.center_distances(stim.features['location'])
            if distances[0] < self.loc_border + self.stim_size:
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

