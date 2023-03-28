"""
This is like buffer batch case where training is done on steps only once,
except instead of a fixed training sequence length, the training equipment is returned only after
the episode has terminated.
"""

import torch
from torch import nn
from rl.q_learning.buffer_batch import (
    QLearningBufferBatchDataset,
    QLearningBufferBatchAgent,
)


class QLearningFullEpisodeDataset(QLearningBufferBatchDataset):
    def __init__(self, *args, **kwargs):
        super(QLearningFullEpisodeDataset, self).__init__(training_seq_length=None)

    def select_batch(self):
        return self.data

    def sample_batch(self):
        if all(self.data[-1]["terminated"]):
            batch = super(QLearningBufferBatchDataset, self).sample_batch()
            self.data = []
            return batch


class QLearningFullEpisodeAgent(QLearningBufferBatchAgent):
    """
    No functional difference, just doesn't take a sequence length - based on episodes instead
    """

    dataset_class = QLearningFullEpisodeDataset

    def __init__(self, model: nn.Module, discount_factor=0.9, C=10, *args, **kwargs):
        super(QLearningFullEpisodeAgent, self).__init__(model, None, discount_factor, C)
