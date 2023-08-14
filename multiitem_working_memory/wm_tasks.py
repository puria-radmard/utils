"""
Some simple WM tasks that produce stimulus_board.py objects in batches too
"""

from math import pi
import random, torch
from typing import List, Dict
from purias_utils.multiitem_working_memory.stimulus_board import Stimulus, StimulusBoardBase, ColouredSquaresBoard
from purias_utils.multiitem_working_memory.stimuli import *

class MultiItemWMTaskBase:

    epoch_names: List[str]

    def generate_boards(self, *args, **kwargs) -> List[StimulusBoardBase]:
        """
        Generate the boards as they would appear in the 'stimulus' 
        (or equivalent) phase of any task.
        Depending on the task, any number of these boards would have
        to be created per task
        """

    def generate_epoch_duration(self, epoch_name, *args, **kwargs) -> Dict[str, int]:
        """
        Generate the time durations (in timesteps!) of each phase of the trial.
        These should be shared amongst the batch items, just for simplicity!
        """
        return random.uniform(
            self.epoch_kwargs[f'{epoch_name}_dur_lower'],
            self.epoch_kwargs[f'{epoch_name}_dur_upper']
        ) / self.epoch_kwargs['dt']

    def augment_boards(self, boards: List[StimulusBoardBase], epoch_name: str):
        """
        Change the primary board layout (see generate_boards) to how they would 
        appear in this named epoch. May require some saving of state, depending on task
        """

    def generate_sensory_mask(self, boards: List[StimulusBoardBase], epoch_name: str):
        """
        Mask should be of size num_sim
        """

    def generate_fixation(self, boards: List[StimulusBoardBase], epoch_name: str):
        ...

    def generate_targets(self, boards: List[StimulusBoardBase], epoch_name: str):
        """
        Will use saved states (see augment_boards) to generate supervision targets
        for each board
        """

    def generate_target_masks(self, boards: List[StimulusBoardBase], epoch_name: str):
        """
        Will use saved states (see augment_boards) to generate supervision target weightings
        for each board
        """

    def generate_epoch(self, *args, **kwargs):
        """
        Bounding method that outputs the model inputs and the supervision targets
        """
        boards = self.generate_boards()
        for epoch_name in self.epoch_names:
            yield dict(
                duration = self.generate_epoch_duration(epoch_name),                 # Single number, shared amongst boards
                fixations = self.generate_fixation(boards, epoch_name),  # Fixation shared amonst all boards, shoudl depend only on which epoch we're in
                aug_boards = self.augment_boards(boards, epoch_name),                # Board, augmented for this epoch
                sensory_masks = self.generate_sensory_mask(boards, epoch_name),      # Mask, specified for each stimulus in each board
                targets = self.generate_targets(boards, epoch_name),                 # Supervision targets
                target_masks = self.generate_target_masks(boards, epoch_name),                 # Supervision targets
                original_boards = boards,
                epoch_name = epoch_name
            )





class SimpleMultiItemWMChangeDetectionTask(MultiItemWMTaskBase):
    """
    Very simple, only uses the ColouredSquaresBoard object for board
    Required epoch_kwargs:
        :dt (float)             = discretisation timestep
        :prestim_dur_lower (float, +)   = seconds before initial stimuli onset
        :prestim_dur_upper (float, +)
        :stim_dur_lower (float, +)      = seconds during first stimuli presentation
        :stim_dur_upper (float, +)
        :wm1_dur_lower (float, +)       = seconds spent in delay between boards
        :wm1_dur_upper (float, +)
        :stim2_dur_lower (float, +)     = seconds during second stimuli presentation
        :stim2_dur_upper (float, +)
        :wm2_dur_lower (float, +)       = seconds spent in delay after second boards
        :wm2_dur_upper (float, +)
        :resp_dur_lower (float, +)      = seconds spent awaiting a (binary) response
        :resp_dur_upper (float, +)
        :p_change (float, [0,1])        = probability of an actual stimulus change
    """

    board_kwargs = dict(
        set_lower=3,
        set_upper=7,
        board_size = 32, 
        stim_size = 3, 
        loc_border = 2.5,
    )

    epoch_kwargs = dict(
        dt = 0.01,
        prestim_dur_lower = 0.2,
        prestim_dur_upper = 0.2,
        stim_dur_lower = 0.5,
        stim_dur_upper = 0.5,
        wm1_dur_lower = 0.5,
        wm1_dur_upper = 0.5,
        stim2_dur_lower = 0.5,
        stim2_dur_upper = 0.5,
        wm2_dur_lower = 0.5,
        wm2_dur_upper = 0.5,
        resp_dur_lower = 0.5,
        resp_dur_upper = 0.5,
        p_change = 0.5,
    )

    epoch_names = ['prestim', 'stim', 'wm1', 'stim2', 'wm2', 'resp']

    def __init__(self, epoch_kwargs: dict, board_kwargs: dict, batch_size: int) -> None:
        self.epoch_kwargs.update(epoch_kwargs)
        self.board_kwargs.update(board_kwargs)
        self.batch_size = batch_size
        self.p_change = self.epoch_kwargs['p_change']
        self._was_changed = None
        
    def generate_boards(self, *args, **kwargs):
        return [ColouredSquaresBoard(**self.board_kwargs) for _ in range(self.batch_size)]

    def augment_boards(self, boards, epoch_name):
        if epoch_name in ['prestim', 'stim', 'wm1', 'wm2', 'resp']:
            return boards   # no stimulus epochs shown will be done with the mask instead
        elif epoch_name == 'stim2':
            altered_boards = []
            self._was_changed = []
            for board in boards:
                copied_board = board.copy()
                if random.random() < self.p_change: # change one with prob p_change, else no change
                    change_amount = random.uniform(0, 2 * pi)   # Uniform change!
                    selected_stim_idx = random.randrange(0, copied_board.set_size)
                    changed_stim = copied_board.stimuli[selected_stim_idx].change('colour', change_amount)
                    copied_board.stimuli[selected_stim_idx] = changed_stim
                    self._was_changed.append(True)
                else:
                    self._was_changed.append(False)
                altered_boards.append(copied_board)
            return altered_boards
        else:
            raise ValueError(epoch_name)

    def generate_sensory_mask(self, boards, epoch_name):
        if epoch_name in ['stim', 'stim2']:
            return torch.tensor([[[1.] for _ in board.stimuli] for board in boards])
        else:
            return torch.tensor([[[0.] for _ in board.stimuli] for board in boards])

    def generate_fixation(self, boards, epoch_name):
        return 0. if epoch_name == 'resp' else 1.
    
    def generate_targets(self, boards, epoch_name):
        "three categories"
        if epoch_name == 'resp':
            targets = []
            for changed in self._was_changed:
                targets.append(1. if changed else 2.)
            self._was_changed = None
        else:
            targets = [0. for _ in boards]
        return targets

    def generate_target_masks(self, boards, epoch_name):
        "maximal weighting to the response epoch"
        # TODO: MAKE PARAMETERISABLE
        if epoch_name == 'resp':
            targets = [10. for _ in boards]
        else:
            targets = [1. for _ in boards]
        return targets



class SimpleMultiItemWMDelayedEstimationTask(MultiItemWMTaskBase):
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
    """

    board_kwargs = dict(
        set_lower=3,
        set_upper=7,
        board_x_size = 32, 
        board_y_size = 32, 
        stim_size = 3, 
        loc_border = 2.5,
    )

    epoch_kwargs = dict(
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
        p_change = 0.5,
    )

    epoch_names = ['prestim', 'stim', 'wm', 'cue', 'resp']

    def __init__(self, epoch_kwargs: dict, board_kwargs: dict, batch_size: int, cue_during_response: bool = True) -> None:
        self.epoch_kwargs.update(epoch_kwargs)
        self.board_kwargs.update(board_kwargs)
        self.batch_size = batch_size
        self._changed_idx = []
        self.cue_during_response = cue_during_response
        
    def generate_boards(self, *args, **kwargs):
        return [ColouredSquaresBoard(**self.board_kwargs) for _ in range(self.batch_size)]

    def augment_boards(self, boards, epoch_name):
        if epoch_name in ['prestim', 'stim', 'wm']:
            return boards   # no stimulus epochs shown will be done with the mask instead
        elif epoch_name in ['cue', 'resp']:
            altered_boards = []
            self._selected_idx = []
            for board in boards:
                selected_stim_idx = random.randrange(0, board.set_size)
                cue_board = board.make_cue_board(selected_stim_idx)
                altered_boards.append(cue_board)
                self._selected_idx.append(selected_stim_idx)
            return altered_boards
        else:
            raise ValueError(epoch_name)

    def generate_sensory_mask(self, boards, epoch_name):
        "because of varying set size, also need to cover everything else"
        max_set_size = max([board.set_size for board in boards])
        if (epoch_name in ['stim', 'cue']) or (epoch_name == 'resp' and self.cue_during_response):
            return torch.tensor([[[1.] for _ in board.stimuli] + [[0.] for _ in range(max_set_size - board.set_size)] for board in boards])
        else:
            return torch.tensor([[[0.] for _ in range(max_set_size)] for board in boards])

    def generate_fixation(self, boards, epoch_name):
        return 0. if epoch_name == 'resp' else 1.
    
    def generate_targets(self, boards, epoch_name):
        "fixation, x coord, y coord. before response epoch, only fixation matters, so just place 0s"
        if epoch_name == 'resp':
            targets = []
            for j, board in enumerate(boards):
                new_target = [0., board.stimuli[self._selected_idx[j]].features['colour'].value]
                targets.append(new_target)
        else:
            targets = [[1., 0.] for _ in boards]
        return targets

    def generate_target_masks(self, boards, epoch_name):
        "before response epoch, just weight the fixation. during response epoch"
        if epoch_name == 'resp':
            target_masks = [[1. / self.epoch_kwargs['fixation_mag_div'], 1.] for _ in boards]
        elif epoch_name == 'prestim':
            target_masks = [[0., 0.] for _ in boards]
        else:
            target_masks = [[1. / self.epoch_kwargs['fixation_mag_div'], 0.] for _ in boards]
        return target_masks


class SimpleSingleItemWMDelayedLocationRecall(MultiItemWMTaskBase):
    """
    Even simpler, just presents a colour (alternatively: an orientation), and expect an
    output. Also has a fixation signal, as above. 
    """

    epoch_kwargs = dict(
        dt = 0.01,
        prestim_dur_lower = 0.1,
        prestim_dur_upper = 0.1,
        stim_dur_lower = 0.4,
        stim_dur_upper = 0.4,
        wm_dur_lower = 0.8,
        wm_dur_upper = 1.1,
        resp_dur_lower = 0.4,
        resp_dur_upper = 0.4,
        p_change = 0.5,
    )

    epoch_names = ['prestim', 'stim', 'wm', 'resp']


    def __init__(self, epoch_kwargs: dict, batch_size: int) -> None:
        self.epoch_kwargs.update(epoch_kwargs)
        self.batch_size = batch_size
        
    def generate_boards(self, *args, **kwargs):
        return [Stimulus(Colour(random_orientation())) for _ in range(self.batch_size)]

    def augment_boards(self, boards, epoch_name):
        return boards

    def generate_sensory_mask(self, boards, epoch_name):
        "because of varying set size, also need to cover everything else"
        return 1. if epoch_name == 'stim' else 0.

    def generate_fixation(self, boards, epoch_name):
        return 0. if epoch_name == 'resp' else 1.

    def generate_targets(self, boards, epoch_name):
        "fixation, x coord, y coord. before response epoch, only fixation matters, so just place 0s"
        return [[0., board.features['colour'].value] if epoch_name == 'resp' else [1., 0.] for board in boards]

    def generate_target_masks(self, boards, epoch_name):
        "before response epoch, just weight the fixation. during response epoch"
        if epoch_name == 'resp':
            target_masks = [[1. / self.epoch_kwargs['fixation_mag_div'], 1.] for _ in boards]
        elif epoch_name == 'prestim':
            target_masks = [[0., 0.] for _ in boards]
        else:
            target_masks = [[1. / self.epoch_kwargs['fixation_mag_div'], 0.] for _ in boards]
        return target_masks
