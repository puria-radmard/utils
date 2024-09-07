from typing import Union
import torch
from torch import Tensor as _T
import numpy
from math import pi

import matplotlib.pyplot as plt

from purias_utils.multiitem_working_memory.stimulus_design.stimulus_board import MultiOrientationStimulusBoard
from purias_utils.multiitem_working_memory.util.circle_utils import rot_to_rgb_torch


class MultiOrientationStimulusBoardVisualiser:

    background = torch.tensor([0.0, 0.0, 0.0])
    cue_stim = torch.tensor([1.0, 1.0, 1.0])

    # Standard dimension_location: [batch, (stim if needed), Y, X, rgb(3)]

    def __init__(self, num_features: int = None, image_size: int = 32, stimulus_radius: int = 2, location_radius: int = 12) -> None:

        self.image_size = image_size
        self.stimulus_radius = stimulus_radius
        self.location_radius = location_radius
        
        assert (self.location_radius + self.stimulus_radius) < (self.image_size / 2)

        self.rot_to_rgb = lambda x: rot_to_rgb_torch(x, nan_replacement = self.cue_stim)

        assert (self.background == 0.0).all(), "Require black background for speeding up..."
        
        if num_features != 2:
            raise NotImplementedError('Can only visualise circular location and colour for now')
        self.num_features = num_features

        self.stimulus_size = 2 * self.stimulus_radius

    def generate_canvas(self, num_batches: int, num_stim: int):
        return torch.ones([num_batches, num_stim, self.image_size, self.image_size, 3]) * self.background.reshape(1, 3)

    def normalised_feature_extraction(self, boards: list[MultiOrientationStimulusBoard], feature_idx: int, ):
        max_set_size = max([board.set_size for board in boards])
        output = torch.zeros([len(boards), max_set_size])
        for b, board in enumerate(boards):
            for s, stimulus in enumerate(board.stimuli):
                output[b, s] = stimulus.features[feature_idx].value
        return output

    def generate_stimulus_patches_from_colours(self, stimulus_colours: _T, stimulus_mask: _T):
        B, S = stimulus_colours.shape
        
        stimulus_patch_shape = torch.ones([B, S, self.stimulus_size, self.stimulus_size, 3])
        background_patches = stimulus_patch_shape * self.background.reshape(1, 3)

        stimuli_rgbs = self.rot_to_rgb(stimulus_colours).reshape(B, S, 1, 1, 3)
        scaled_patches = stimuli_rgbs * stimulus_patch_shape
        
        smask = stimulus_mask #.reshape(B, S, 1, 1, 1)
        masked_patches = (smask * scaled_patches) + ((1. - smask) * background_patches)

        return masked_patches

    def generate_stimulus_patches(self, boards: list[MultiOrientationStimulusBoard], stimulus_mask: _T):
        stimulus_colours = self.normalised_feature_extraction(boards, 1)
        return self.generate_stimulus_patches_from_colours(stimulus_colours, stimulus_mask)

    def generate_xy_locations_from_stimulus_locations(self, stimulus_locations: _T, stimulus_mask: _T):
        x_locs: _T = (self.location_radius * stimulus_locations.cos())
        y_locs: _T = (self.location_radius * stimulus_locations.sin())

        # bring to center if masked... hacky...
        x_locs = self.image_size / 2 + (x_locs * stimulus_mask)
        y_locs = self.image_size / 2 - (y_locs * stimulus_mask) # images are top down!

        return x_locs.int(), y_locs.int()

    def generate_xy_locations(self, boards: list[MultiOrientationStimulusBoard]):
        stimulus_locations = self.normalised_feature_extraction(boards, 0)
        return self.generate_xy_locations_from_stimulus_locations(stimulus_locations)

    def generate_board_images_from_locations_and_patches(self, x_locs, y_locs, stim_patches):
        B, S, *_ = stim_patches.shape
        canvas = self.generate_canvas(B, 1).squeeze(1)

        x_lowers = x_locs-self.stimulus_radius
        x_uppers = x_locs+self.stimulus_radius
        y_lowers = y_locs+self.stimulus_radius  # ys backwards!
        y_uppers = y_locs-self.stimulus_radius

        for b in range(B):

            for s in range(S - 1, -1, -1): # work backwards because of masking (trust me)

                x_lower = x_lowers[b,s]
                x_upper = x_uppers[b,s]
                y_lower = y_lowers[b,s]
                y_upper = y_uppers[b,s]

                # ys backwards!
                canvas[b, y_upper:y_lower, x_lower:x_upper] = stim_patches[b, s]

        return canvas

    def generate_board_images(self, boards: list[MultiOrientationStimulusBoard], stimulus_mask: Union[_T, float]):
        x_locs, y_locs = self.generate_xy_locations(boards, stimulus_mask)
        stim_patches = self.generate_stimulus_patches(boards, stimulus_mask)
        return self.generate_board_images_from_locations_and_patches(self, x_locs, y_locs, stim_patches)




if __name__ == '__main__':

    from purias_utils.multiitem_working_memory.wm_tasks.wm_tasks import MultipleOrientationDelayedSingleEstimationTask

    task = MultipleOrientationDelayedSingleEstimationTask(
        board_kwargs={
            "set_lower": 2,
            "set_upper": 7,
            "num_features": 2,
            "feature_borders": [0.95 * pi / 7, 0.95 * pi / 7] 
        },
        epoch_kwargs={},
        batch_size=64,
        cue_during_response=False
    )

    vis = MultiOrientationStimulusBoardVisualiser(2, 224, 10, 80)

    def main():

        num_examples = 5
        fig, axes = plt.subplots(num_examples, 5, figsize = (25, num_examples*5))

        epoch_num = 0
        for epoch_info in task.generate_epoch(target_feature_idx=1):

            aug_boards = epoch_info['aug_boards']
            smask = epoch_info['sensory_masks']
            tmask = epoch_info['target_masks']
            targets = epoch_info['targets']

            images = vis.generate_board_images(aug_boards, smask)
            normed_images = (images - 1.0) * 2.0

            for ei in range(num_examples):
                axes[ei, epoch_num].imshow(images[ei].numpy())
            epoch_num += 1

        fig.savefig('example_tasks.png')

    import timeit
    print(timeit.repeat(main, number=1, repeat=1))
