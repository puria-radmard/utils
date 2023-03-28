"""
Base classes for the Q-learning/TD family of DRL algorithms
"""

import random
import numpy as np
from torch import nn
import torch
from torch.utils.data import Dataset


class BaseAgent(nn.Module):
    """
    Base class for all (TD) agents, just gives the empty functions which must be populated in all cases
    Tried to keep this as generic as possible.s

    Required args:
        :: actor_network - generates the next action, may use previous action for the sake of state
            representation (see todo above)

        :: actor_network - generates a raw list/tuple of output, extra information
            here, 'output' is always used to generate the next action, and all other values are used
            for future representation (see todo above), so will include things like hidden
            representations for

        :: actor_interperter - interperets the actor_network output for the algorithm,
            e.g. changes a pmf into a single action, add stochasticity etc.
            By default just returns argmax on the -1th axis

        :: critic_network - generates the current value function. By default the same as actor_network

        :: critic_interpretor - same thing for critic_network, to allow actor and critic being the same
            By default just returns max on the -1th axis
            Could also use this as a max operator, while the actor_interperter is the argmax operator
                for the same network

    Basic run through for each iteration:
        1. The agent takes in a new observation
        2. Using the new observation and any catched hidden stuff, a new action is chosen (step_prereward -> choose_actions)
        3. The action is used to generate a reward, and the environment updates, giving a new observation
            NB: This is the same observation as will be again seen in the next step 1!
        4. The reward, old env, new env, etc. are taken in and used for training somehow (step_postreward)
        5. This repeats!

    NB: all networks take in network(observation, hidden_information, **kwargs)
        Models need to be built to deal with this format
    """

    def __init__(
        self, dataset, actor_network, critic_network=None,
    ):
        super(BaseAgent, self).__init__()

        # These will be built up each episode, and dealt with by specific algorithm
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.policy_history = []

        # For recurrent states, etc., which won't exist at first
        self.hidden_state_history = []

        # List of objects of any structure - won't be stacked like a tensor at the end, just returned as a list
        self.extra_history = []

        # See BaseDataset below
        self.dataset = dataset

        self.actor_network = actor_network
        self.critic_network = (
            critic_network if critic_network != None else actor_network
        )

    def actor_interperter(self, actor_output):
        return actor_output.argmax(-1)

    def critic_interpretor(self, critic_output):
        return critic_output.max(-1)

    def receive_reward(self, new_reward):
        """
        Add reward to history, depends on specific algorithm how we deal with this
        """
        if not isinstance(new_reward, torch.Tensor):
            new_reward = torch.tensor(new_reward)
        self.reward_history.append(new_reward)
        return new_reward

    def choose_action(self, new_observation, **kwargs):
        """
        This gets the next action from the current observation, see __init__ docstring
        Also save the action to history
        """
        # Odd case - need to initialise extra history here, to account for batch size
        if len(self.hidden_state_history) == 0:
            initial_hidden_state = self.actor_network.initialise_hidden_state(
                new_observation
            )
            self.hidden_state_history.append(initial_hidden_state)
        previous_hidden_information = self.hidden_state_history[-1]
        policy, hidden_information = self.actor_network(
            new_observation, previous_hidden_information, **kwargs
        )
        action = self.actor_interperter(policy)
        self.policy_history.append(policy)
        self.action_history.append(action)
        self.hidden_state_history.append(hidden_information)
        return action

    def get_value(self, observation, hidden_information, **kwargs):
        """
        Same as above but for the critic network.
        """
        critic_output = self.critic_network(observation, hidden_information, **kwargs)
        pred_value = self.critic_interperter(critic_output)
        return pred_value

    def step_prereward(self, observation: torch.Tensor, **kwargs):
        """
        This is typically just a matter of providing an action to the environment, so by default:

        TODO: observation should just be the previous observation (picked up and saved by previous step_postreward)
        """
        self.observation_history.append(observation)
        self.extra_history.append(kwargs)
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation)
        return self.choose_action(observation, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Gym compatibility, ie. action, _states = model.predict(ob, _states)
        i.e. we don't actually use the _states inputted
        """
        return self.step_prereward(args[0]), self.hidden_state_history

    def step_postreward(self, observation, reward, terminated):
        """
        Very specific to algorithm, so we leave as a complete blank.
        terminated is a boolean, required for certain algorithms' logging purposes
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Clear histories, must be done at the start of each episode.
        Overwrite this for things like training steps done only at the end of the episode (not TD ??).
        Also overwrite for if you want the dataset to store/forget whole episodes

        NB: in some cases, you might want to continue training before ending the episode
            e.g. for buffer batch case, you might want to train on the last batch, or for experience
                replay you might want to continue training on old experience repeatedly
            in this case, you can typically use:

                batch = self.dataset.sample_batch()
                current_policy_q_values, y = self.get_target_and_pred(batch)

            repeatedly, or once in the case of buffer batch
        """

        # Dataset might still need these histories
        self.dataset.end_episode(self)

        # Reset for new episode
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.policy_history = []
        self.hidden_state_history = []
        self.extra_history = []


class BaseDataset(Dataset):
    """
    This is the base class for a dataset that draws from agent experience.
    This is used in the training stage of the algorithm, which can be at agent.step
        and/or agent.end_episode

    It accesses the agent.**_history lists

    In the DRL case, states are replaced by representations, used by the networks to
    generate actions/values
    """

    def __init__(self, training_seq_length):
        super(BaseDataset, self).__init__()

        # This will be populated with, for example, quartets
        self.data = []
        self.training_seq_length = training_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def save_from_agent(self, agent, **kwargs):
        raise NotImplementedError

    def end_episode(self, agent, **kwargs):
        raise NotImplementedError

    def select_batch(self):
        # TODO: make this smarter wrt batch memory
        training_seq_length = max(self.training_seq_length, len(self))
        return random.sample(self.data, training_seq_length)

    def sample_batch(self):
        raise NotImplementedError
