import torch
import random
from typing import List

from purias_utils.multiitem_working_memory.wm_tasks.base import SimpleMultiItemWMDelayedEstimationTask
from purias_utils.multiitem_working_memory.stimulus_design.generate_displays import MultiOrientationStimulusBoardVisualiser
from purias_utils.multiitem_working_memory.stimulus_design.stimulus_board import StimulusBoardBase, MultiOrientationStimulusBoard

from purias_utils.util.api import yield_as_obj


class MultipleOrientationDelayedSingleEstimationTask(SimpleMultiItemWMDelayedEstimationTask):
    """
    Very simple, only uses the ColouredSquaresBoard object for board
    Required epoch_kwargs:
        :dt (float)             = discretisation timestep
        :prestim_dur_lower (float, +)   = seconds before initial stimuli onset
        :prestim_dur_upper (float, +)
        :stim_dur_lower (float, +)      = seconds during first stimuli presentation
        :stim_dur_upper (float, +)
        :wm_dur_lower (float, +)        = seconds spent in delay before cue
        :wm_dur_upper (float, +)
        :cue_dur_lower (float, +)     = seconds spent given cue
        :cue_dur_upper (float, +)
        :resp_dur_lower (float, +)      = seconds spent awaiting a continuous response, with cue still shown
        :resp_dur_upper (float, +)

    Always cues the first item during response period!
    """

    # Defaults from Bays 2009
    board_kwargs = dict(
        set_sizes=[1, 2, 4, 6],
        num_features=2,
        feature_borders=[torch.pi/8, torch.pi/8]
    )

    epoch_kwargs = dict(
        cue_during_response = True,
        dt = 0.01,
        prestim_dur_lower = 0.1,
        prestim_dur_upper = 0.1,
        stim_dur_lower = 0.4,
        stim_dur_upper = 0.4,
        wm_dur_lower = 0.8,
        wm_dur_upper = 1.1,
        cue_dur_lower = 0.2,
        cue_dur_upper = 0.2,
        resp_dur_lower = 0.4,
        resp_dur_upper = 0.4,
    )

    epoch_names = ['prestim', 'stim', 'wm', 'cue', 'resp']

    def __init__(
        self, 
        epoch_kwargs: dict, 
        board_kwargs: dict, 
        batch_size: int, 
        visualiser: MultiOrientationStimulusBoardVisualiser = None
    ) -> None:
        self.epoch_kwargs.update(epoch_kwargs)
        self.board_kwargs.update(board_kwargs)
        self.batch_size = batch_size
        self._changed_idx = []
        self.visualiser = visualiser

    def generate_boards(self, *args, **kwargs) -> List[MultiOrientationStimulusBoard]: 
        # set_size = random.randint(self.board_kwargs['set_lower'], self.board_kwargs['set_upper'])
        set_size = random.choice(self.board_kwargs['set_sizes'])
        return [MultiOrientationStimulusBoard(set_size = set_size, **self.board_kwargs) for _ in range(self.batch_size)], set_size

    def augment_boards(self, boards: List[MultiOrientationStimulusBoard], epoch_name: str, target_feature_idx: int): 
        if epoch_name in ['prestim', 'stim', 'wm']:
            return boards   # no stimulus epochs shown will be done with the mask instead
        elif epoch_name in ['cue', 'resp']:  # deal with resp period choice in with generate_sensory_mask instead!
            altered_boards = []
            for board in boards:
                cue_board = board.make_cue_board(0, target_feature_idx=target_feature_idx)   # Always first item cued
                altered_boards.append(cue_board)
            return altered_boards
        else:
            raise ValueError(epoch_name)

    def generate_sensory_mask(self, epoch_name):
        "because of varying set size, also need to cover everything else"
        if self.epoch_kwargs['cue_during_response']:
            return 1.0 if epoch_name in ['stim', 'cue', 'resp'] else 0.0
        else:
            return 1.0 if epoch_name in ['stim', 'cue'] else 0.0

    def generate_fixation(self, epoch_name):
        return 0. if epoch_name == 'resp' else 1.

    def generate_targets(self, boards: List[StimulusBoardBase], epoch_name: str, target_feature_idx: int):
        "orientation and xy location on unit circle. before response epoch, only fixation matters, so just place 0s"
        if epoch_name == 'resp':
            angles = torch.tensor([board.stimuli[0].features[target_feature_idx].value for board in boards])   # Always first item cued
            distractor_angles = [
                torch.tensor([board.stimuli[rr].features[target_feature_idx].value for board in boards])
                for rr in range(1, boards[0].set_size)
            ]   # All distractors in a list!
            targets = torch.stack([angles.sin(), angles.cos()], -1)
        else:
            angles =  torch.tensor([torch.nan for _ in boards])
            distractor_angles = None
            targets =  torch.tensor([[0., 0.] for _ in boards])
        return {'angles': angles, 'saccade_targets': targets, 'distractor_angles': distractor_angles}

    @yield_as_obj
    def generate_epoch_from_boards(self, target_feature_idx, boards, set_size, *args, **kwargs):
        
        for epoch_name in self.epoch_names:
            ret = dict(
                duration = self.generate_epoch_duration(epoch_name),                                        # Single number, shared amongst boards
                fixation = self.generate_fixation(epoch_name),                                              # Fixation shared amonst all boards, should depend only on which epoch we're in
                aug_boards = self.augment_boards(boards, epoch_name, target_feature_idx),                   # Board, augmented for this epoch
                sensory_masks = self.generate_sensory_mask(epoch_name),                                     # Mask, specified for each stimulus in each board
                targets = self.generate_targets(boards, epoch_name, target_feature_idx),                    # Supervision targets (always first item)
                original_boards = boards,
                epoch_name = epoch_name,
                set_size = set_size
            )

            if self.visualiser is not None:
                ret['images'] = self.visualiser.generate_board_images(
                    ret['aug_boards'], ret['sensory_masks']
                )
                
            yield ret

    
    def generate_epoch(self, target_feature_idx, *args, **kwargs): 
        """
        Bounding method that outputs the model inputs and the supervision targets.
        
        """
        boards, set_size = self.generate_boards()

        for epoch_info in self.generate_epoch_from_boards(target_feature_idx, boards, set_size, *args, **kwargs):
            yield epoch_info


