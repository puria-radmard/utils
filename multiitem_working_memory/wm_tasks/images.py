import torch
import random
from torch import Tensor as _T

from purias_utils.util.api import yield_as_obj

from purias_utils.multiitem_working_memory.wm_tasks.base import SimpleMultiItemWMDelayedEstimationTask

class SimpleImageSaccadeClassificationTask(SimpleMultiItemWMDelayedEstimationTask):
    """
    Keeping this super simple:
        prestim:    no display
        stim:       batch of images
        wm:         no display
        resp:       no display, fixation turned off

    Also requires num_classes to set the location around the circle for the targets

    Targets are zeros when fixation is off, then a unit circle point during resp
        --> designed to be used with MSELoss with training throughout
        (for the communication subspace stuff!)
    As in MultipleOrientationDelayedSingleEstimationTask however, targets are provided 
        as dict, this time also with a simple 'class' idx included

    Input:
        images      = [dataset size, ...]
        labels      = [dataset size (0 to C-1)]
        xy_targets  = [C]
    """

    board_kwargs = dict(
        image_during_response = False
    )

    epoch_kwargs = dict(
        dt = 0.01,
        prestim_dur_lower = 0.1,
        prestim_dur_upper = 0.1,
        stim_dur_lower = 0.15,
        stim_dur_upper = 0.15,
        wm_dur_lower = 0.0,
        wm_dur_upper = 0.0,
        resp_dur_lower = 0.15,
        resp_dur_upper = 0.15,
    )

    epoch_names = ['prestim', 'stim', 'wm', 'resp']

    def __init__(
        self, 
        epoch_kwargs: dict, 
        board_kwargs: dict, 
        train_images: _T, 
        train_labels: _T, 
        test_images: _T, 
        test_labels: _T, 
        angle_targets: _T, 
        batch_size: int, 
        num_classes: int,
        test_batch_size: int = None,
        **kwargs
    ):

        self.epoch_kwargs.update(epoch_kwargs)
        self.board_kwargs.update(board_kwargs)

        if self.board_kwargs['image_during_response']:
            assert self.epoch_kwargs['wm_dur_lower'] == self.epoch_kwargs['wm_dur_upper'] == 0.0

        self.num_classes = num_classes
        self.num_train_images, *self.image_size = train_images.shape
        self.num_test_images, *_ = test_images.shape
        self.angle_targets = angle_targets
        self.xy_targets = torch.stack([angle_targets.cos(), angle_targets.sin()], dim = 1)
        assert list(angle_targets.shape) == [self.num_classes]
        assert train_labels.min() == train_labels.min() == 0.0
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        self.batch_size = batch_size
        self.test_batch_size = batch_size * 8 if test_batch_size is None else test_batch_size
        self.all_indices = range(self.num_train_images)
        self.num_test_batches = (test_images.shape[0] // self.test_batch_size) + (test_images.shape[0] % self.test_batch_size != 0)

    def generate_fixation(self, epoch_name):
        return 0. if epoch_name == 'resp' else 1.

    def generate_sensory_mask(self, epoch_name):
        "because of varying set size, also need to cover everything else"
        if self.board_kwargs['image_during_response']:
            return 1. if epoch_name in ['stim', 'resp'] else 0.
        else:
            return 1. if epoch_name == 'stim' else 0.

    def generate_targets(self, batch_labels, epoch_name):
        target_classes = batch_labels.long()
        if epoch_name == 'resp':
            target_orientations = self.angle_targets[target_classes]
            target_locations = self.xy_targets[target_classes]
            target_distributions = torch.nn.functional.one_hot(target_classes, self.num_classes)
        else:
            target_orientations = torch.nan * torch.ones(len(batch_labels), 1)
            target_locations = torch.zeros(len(batch_labels), 2)
            target_distributions = torch.ones(len(batch_labels), self.num_classes) / self.num_classes

        return {
            'angles': target_orientations, 
            'saccade_targets': target_locations, 
            'distribution_targets': target_distributions,
            'class': target_classes,
        }

    @yield_as_obj
    def generate_epoch(self, test_batch_idx: int, *args, **kwargs): 
        """        
        test_batch_idx chooses from ordered test set.
        Set to -1 for a random training batch
            NB: batch is selected truly i.i.d., i.e. no rotation...
        """
        
        if test_batch_idx == -1:
            batch_indices = random.sample(self.all_indices, self.batch_size)
            batch_images = self.train_images[batch_indices]                 # [batch, H, W, 3]
            batch_labels = self.train_labels[batch_indices]
        else:
            assert 0 <= test_batch_idx < self.num_test_batches
            batch_indices = slice(test_batch_idx*self.test_batch_size, (1+test_batch_idx)*self.test_batch_size)
            batch_images = self.test_images[batch_indices]
            batch_labels = self.test_labels[batch_indices]

        for epoch_name in self.epoch_names:

            sensory_mask = self.generate_sensory_mask(epoch_name)

            ret = dict(
                duration = self.generate_epoch_duration(epoch_name),
                fixation = self.generate_fixation(epoch_name),
                targets = self.generate_targets(batch_labels, epoch_name),
                images = batch_images * sensory_mask,
                epoch_name = epoch_name,
            )

            yield ret


