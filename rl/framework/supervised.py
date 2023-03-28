"""
This builds on the base classes for supervised learning of actions

Supervised learning requires only a correct answer from the environment at each timestep, which is optimised by the agent. 
For the correct reward signal, this is just Q-Leanring with a discount factor of 0!
This means we can build up a Q-Learning library, and to save computation, just remove references to a critic
Everything else should be the same, given a reward signal that is suitable
"""

import torch
from torch import nn
from rl.q_learning import *
from rl.q_learning.full_episode_step import QLearningFullEpisodeAgent


class SupervisedLearningMixin:
    def __init__(self, model: nn.Module, training_seq_length=8):
        """
        No need to update step_critic_update if we set C to be infinite
        """
        super(SupervisedLearningMixin, self).__init__(
            model=model,
            training_seq_length=training_seq_length,
            discount_factor=0,
            C=float("inf"),
        )

    def get_value(self, observation, hidden_information, **kwargs):
        """
        There is no longer a 'value' to predict
        This means the Q-Learning objective:
            y_t = r_t + self.discount_factor * non_terminal_mask * greedy_critic_term
        just becomes the current reward, to be maximised

        Q values, given for each action available to the agent, then become logits on
            the ection to take
        Importantly, the reward received from the environment, r_t, becomes a one hot
            enocded vector of the correct action to take

            => get_target_and_pred just returns predicted and true action vectors
        """
        return 0


class SupervisedLearningExperienceReplayAgent(
    SupervisedLearningMixin, QLearningExperienceReplayAgent
):
    """
    Experience replay: after each step train on [training_seq_length] instances drawn from all previous
    experience.
    """


class SupervisedLearningBufferBatchAgent(
    SupervisedLearningMixin, QLearningBufferBatchAgent
):
    """
    Buffer batch: each batch called only once, after a set sized buffer in steps
    """


class SupervisedLearningFullEpisodeAgent(
    SupervisedLearningMixin, QLearningFullEpisodeAgent
):
    """
    Buffer batch: each batch called only once, after a set sized buffer in steps
    """
