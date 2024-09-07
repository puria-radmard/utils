import torch
import random
from torch import Tensor as _T

from purias_utils.util.api import yield_as_obj
from purias_utils.util.nieder2021 import generate_image_with_numerousity, 

from purias_utils.multiitem_working_memory.wm_tasks.base import SimpleMultiItemWMDelayedEstimationTask


class DMSTaskBase(SimpleMultiItemWMDelayedEstimationTask):
    """
    phases:
        prestim
        Stim 1 is an image
        Delay 1 of variable length
        stim 2 is an image
        (Delay 2 of variable length)
        response: fixation off

    Trained to fixate at center until fixaiton is off. 
    Saccade to correct class if stims match category or remain fixated
    """

    board_kwargs = dict(
        dms_match_probability = 0.5
    )

    epoch_kwargs = dict(
        dt = 0.01,
        prestim_dur_lower = 0.1,
        prestim_dur_upper = 0.1,
        stim1_dur_lower = 0.10,
        stim1_dur_upper = 0.10,
        stim2_dur_lower = 0.10,
        stim2_dur_upper = 0.10,
        wm_dur_lower = 0.0,
        wm_dur_upper = 0.0,
        wm2_dur_lower = 0.0,
        wm2_dur_upper = 0.0,
        resp_dur_lower = 0.15,
        resp_dur_upper = 0.15,
    )

    epoch_names = ['prestim', 'stim1', 'wm', 'stim2', 'wm2', 'resp']

    def __init__(
        self, 
        epoch_kwargs: dict, 
        board_kwargs: dict, 
        batch_size: int, 
        test_batch_size: int = None,
        **kwargs
    ):

        self.epoch_kwargs.update(epoch_kwargs)
        self.board_kwargs.update(board_kwargs)

        self.batch_size = batch_size
        self.test_batch_size = batch_size * 8 if test_batch_size is None else test_batch_size

    def generate_fixation(self, epoch_name):
        "Done purely for the sake of training!"
        return 0. if epoch_name == 'resp' else 1.

    def generate_sensory_mask(self, epoch_name):
        "because of varying set size, also need to cover everything else"
        return 1. if epoch_name in ['stim1', 'stim2'] else 0.

    def generate_targets(self, batch_labels1, batch_labels2, epoch_name):
        ""
        if epoch_name == 'resp':
            target_classes = torch.stack([batch_labels1, batch_labels2], dim = 1)
            match = (batch_labels1 == batch_labels2).float()
        else:
            target_classes = torch.nan * torch.ones(len(batch_labels1), 2)
            match = torch.nan * batch_labels1

        return {
            'classes': target_classes,
            'match': match
        }



class ClassBasedImageDMSTask(DMSTaskBase):
    """
    For training, two batches are chosen i.i.d. (bound by probability of matching classes)
    For testing, one batch is selected systematically cycling through classes, and the other is i.i.d. from the test set (i.e. both batches are unseen)
    """

    def __init__(
        self, 
        epoch_kwargs: dict, 
        board_kwargs: dict, 
        train_images: _T, 
        train_labels: _T, 
        test_images: _T, 
        test_labels: _T, 
        batch_size: int, 
        num_classes: int,
        test_batch_size: int = None,
        **kwargs
    ):

        super().__init__(epoch_kwargs, board_kwargs, batch_size, test_batch_size)

        self.num_classes = num_classes
        self.num_train_images, *_ = train_images.shape
        self.num_test_images, *_ = test_images.shape
        assert train_labels.min() == train_labels.min() == 0.0
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        self.all_indices = list(range(self.num_train_images))
        self.test_indices = list(range(self.num_test_images))
        self.num_test_batches = (test_images.shape[0] // self.test_batch_size) + (test_images.shape[0] % self.test_batch_size != 0)

        all_indices_tensor = torch.tensor(self.all_indices)
        test_indices_tensor = torch.tensor(self.test_indices)
        self.train_label_to_index = {l: all_indices_tensor[self.train_labels == float(l)].tolist() for l in range(num_classes)}
        self.test_label_to_index = {l: test_indices_tensor[self.test_labels == float(l)].tolist() for l in range(num_classes)}

        # Adjust to make exact!
        if self.board_kwargs['dms_match_probability'] != None:
            self.board_kwargs['dms_match_probability'] = (self.num_classes * self.board_kwargs['dms_match_probability'] - 1) / (self.num_classes - 1)
        else:
            self.board_kwargs['dms_match_probability'] = 0.0

    def generate_second_batch_indices(self, batch1_labels: list, full_indices: list, label_to_indices: dict):
        batch2_indices = []
        for b1l in batch1_labels:
            u = random.uniform(0.0, 1.0)
            if u < self.board_kwargs['dms_match_probability']:
                batch2_indices.append(random.sample(label_to_indices[b1l], 1)[0])
            else:
                batch2_indices.append(random.sample(full_indices, 1)[0])
        return batch2_indices

    @yield_as_obj
    def generate_epoch(self, test_batch_idx: int, *args, **kwargs): 
        """        
        test_batch_idx chooses from ordered test set.
        Set to -1 for a random training batch
            NB: batch is selected truly i.i.d., i.e. no rotation...
        """
        
        if test_batch_idx == -1:
            batch_indices1 = random.sample(self.all_indices, self.batch_size)
            batch_images1 = self.train_images[batch_indices1]                 # [batch, H, W, 3]
            batch_labels1 = self.train_labels[batch_indices1]
            batch_indices2 = self.generate_second_batch_indices(batch_labels1.tolist(), self.all_indices, self.train_label_to_index)
            batch_images2 = self.train_images[batch_indices2]
            batch_labels2 = self.train_labels[batch_indices2]
        else:
            assert 0 <= test_batch_idx < self.num_test_batches
            batch_indices1 = slice(test_batch_idx*self.test_batch_size, (1+test_batch_idx)*self.test_batch_size)
            batch_images1 = self.test_images[batch_indices1]
            batch_labels1 = self.test_labels[batch_indices1]
            batch_indices2 = self.generate_second_batch_indices(batch_labels1.tolist(), self.test_indices, self.test_label_to_index)
            batch_images2 = self.test_images[batch_indices2]
            batch_labels2 = self.test_labels[batch_indices2]

        for epoch_name in self.epoch_names:

            sensory_mask = self.generate_sensory_mask(epoch_name)
            relevant_images = (
                batch_images1 if epoch_name == 'stim1' else batch_images2
            )

            ret = dict(
                duration = self.generate_epoch_duration(epoch_name),
                fixation = self.generate_fixation(epoch_name),
                targets = self.generate_targets(batch_labels1, batch_labels2, epoch_name),
                images = relevant_images * sensory_mask,
                epoch_name = epoch_name,
            )

            yield ret


class NiederNumerousityDMSTask(DMSTaskBase):
    """
    In this case, batch labels (called counts) are item counts!
    """
    
    board_kwargs = dict(
        image_size = 224,
        dms_match_probability = 0.5, 
        count_lower = 1,
        count_upper = 7,
        ## XXX: add more control options here!
    )

    def __init__(self, epoch_kwargs: dict, board_kwargs: dict, batch_size: int, test_batch_size: int = None, **kwargs):
        super().__init__(epoch_kwargs, board_kwargs, batch_size, test_batch_size, **kwargs)

        self.batch_size = batch_size
        self.possible_counts = list(range(
            self.board_kwargs['count_lower'], self.board_kwargs['count_upper'] + 1
        ))

    def generate_second_batch_item_counts(self, batch1_counts: list):
        batch2_counts = []
        for b1l in batch1_counts:
            u = random.uniform(0.0, 1.0)
            if u < self.board_kwargs['dms_match_probability']:
                batch2_counts.append(b1l)
            else:
                batch2_counts.append(random.sample(self.possible_counts, 1))
        return torch.tensor(batch2_counts)

    def generate_images_from_counts(self, counts):
        "counts of length batch. output of size [batch, image_size, image_size, 3]"
        
        image_size = self.board_kwargs['image_size']

        all_images = []

        for n in counts:
            new_image = generate_image_with_numerousity(
                n, background_func, image_size, n_channels, max_iter, sample_fn, 
                sample_fn_args, background_args, verify_fn, draw_fn
            )

            all_images.append(new_image)
        
        return torch.stack(all_images)

    @yield_as_obj
    def generate_epoch(self, *args, **kwargs): 
        """        
        "test batch" has no effect here
        """
        batch_counts1 = torch.tensor([random.choice(self.possible_counts) for _ in range(self.batch_size)])
        batch_counts2 = self.generate_second_batch_item_counts(batch_counts1)

        batch_images1 = self.generate_images_from_counts(batch_counts1)
        batch_images2 = self.generate_images_from_counts(batch_counts2)

        for epoch_name in self.epoch_names:

            sensory_mask = self.generate_sensory_mask(epoch_name)
            relevant_images = (
                batch_images1 if epoch_name == 'stim1' else batch_images2
            )

            ret = dict(
                duration = self.generate_epoch_duration(epoch_name),
                fixation = self.generate_fixation(epoch_name),
                targets = self.generate_targets(batch_counts1, batch_counts2, epoch_name),
                images = relevant_images * sensory_mask,
                epoch_name = epoch_name,
            )

            yield ret

