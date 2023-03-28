"""
This builds on the base classes for deep Q-learning with a buffer batch training regime
"""

import torch
from torch import nn
from rl.q_learning.experience_replay import (
    QLearningExperienceReplayAgent,
    QLearningExperienceReplayDataset,
)


class QLearningBufferBatchDataset(QLearningExperienceReplayDataset):
    """
    This is made for QLearningBufferBatchAgent

    This algorithm stores the quartet:
        (s_t, a_t, r_t, s_{t+1})
    for the last [training_seq_length] steps.

    After feeding them to the agent, it resets.
    """

    def __init__(self, training_seq_length):
        super(QLearningBufferBatchDataset, self).__init__(training_seq_length)

    def end_episode(self, agent):
        """
        A batch can span episodes, no need to get rid of a part-batch
        """
        pass

    def sample_batch(self, override_ticker=False):
        """
        The whole history saved by this agent is now just the last [training_seq_length] examples

        """
        if override_ticker or len(self.data) >= self.training_seq_length:
            batch = super(QLearningBufferBatchDataset, self).sample_batch()
            self.data = []
            return batch


class QLearningBufferBatchAgent(QLearningExperienceReplayAgent):
    """
    See QLearningExperienceReplayAgent for most functionality

    Batch size also now determines a training step ticker, for the buffer size
    """

    dataset_class = QLearningBufferBatchDataset

    def __init__(
        self, model: nn.Module, training_seq_length=8, discount_factor=0.9, C=10
    ):

        self.training_seq_length = training_seq_length

        super(QLearningBufferBatchAgent, self).__init__(
            model, training_seq_length, discount_factor, C
        )

    def critic_interpretor(self, critic_output):
        """
        Critic is used as the greedy return predictor
        """
        return critic_output.max(-1)

    def get_target_and_pred(self, batch):
        """
        This time, batch might be None! (see QLearningBufferBatchDataset.sample_batch)
        """
        if batch:
            return super(QLearningBufferBatchAgent, self).get_target_and_pred(batch)
        else:
            # Instead of (y_pred, y_true, batch)
            # External script will now have to deal with these, as before
            return None, None, None
