"""
This builds on the base classes for deep Q-learning with experience replay
"""

import copy
import torch
from torch import nn
from rl.base_classes import BaseAgent, BaseDataset


class QLearningExperienceReplayDataset(BaseDataset):
    """
    This is made for QLearningExperienceReplayAgent

    This algorithm allows drawing of any quartet:
        (s_t, a_t, r_t, s_{t+1})
    in the full history of all episodes.
    """

    def save_from_agent(self, agent, terminated):
        """
        In the basic case, we call this at every step to save the most recent quartet
        NB: state/observation histroy starts with a 'head start'

        If there has not been a transition yet, return s_t as zero - model should deal with this

        terminated is required for when this sample is drawn
        """

        pretransition_observation = (
            agent.observation_history[-2]
            if len(agent.observation_history) > 1
            else agent.observation_history[-1] * 0.0
        )

        # Latest values in histories are s_{t+1}, a_t, r_t, h_{t+1}
        self.data.append(
            {
                "s_t": pretransition_observation,
                "h_t": agent.hidden_state_history[-2],
                "s_t_dash": agent.observation_history[-1],
                "h_t_dash": agent.hidden_state_history[-1],
                "a_t": agent.action_history[-1],
                "r_t": agent.reward_history[-1],
                "pi_t": agent.policy_history[-1],
                # TODO: support some tasks in batch ending before others
                "terminated": terminated,
                "extra": agent.extra_history[-1],
            }
        )

    def end_episode(self, agent):
        """
        We want data from all episodes, so don't want to change anything at the end
        of the episode
        """
        pass

    def sample_batch(self):
        """
        This generates the targets for the optimisation step.
        This is just the quartet reward if it is a terminating step, else a discounted
            return using the critic network as a predictor
        Doing this in a vectorised/batched way, to save computation
        """

        # List of dictionaries, see agent.save_from_agent
        batch = self.select_batch()

        # Combine batch size and sequence length dimensions - these are repetitions of single steps
        # from past experience.
        # Basically only preserve Markov step size (typically just 1) for RNN, and observation size

        return {
            # Get the base reward for each batch
            "r_t": torch.hstack(
                [b["r_t"] for b in batch]
            ),  # batch_size * sequence length
            # TODO: support different trial lengths!
            # Which instances in the batch require an additional term from the critic network
            "non_terminal_mask": 1.0
            - torch.hstack(
                [b["terminated"] for b in batch]
            ).float(),  # batch_size * sequence length
            # Which actions were chosen, i.e. which ones do we get the value for using actor network?
            "a_t": torch.hstack(
                [b["a_t"] for b in batch]
            ),  # batch_size * sequence length; indexing action size
            # When using the actor network, i.e. current policy, we use the pre transition state/observation
            "s_t": torch.vstack(
                [b["s_t"] for b in batch]
            ),  # batch_size * sequence length, obs size
            "h_t": torch.vstack(
                [b["h_t"] for b in batch]
            ),  # batch_size * sequence length, hidden size
            # But for the critic, i.e. lagging policy, we use the post transition one
            "s_t_dash": torch.vstack([b["s_t_dash"] for b in batch]),
            "h_t_dash": torch.vstack([b["h_t_dash"] for b in batch]),
            # Policy used at the time, for the sake of supervised learning
            "pi_t": torch.vstack([b["pi_t"] for b in batch]),
            # Finally, unstructured return:
            "extra": [b["extra"] for b in batch],
        }


class QLearningExperienceReplayAgent(BaseAgent):
    """
    Algorithm in figure in: https://jonathan-hui.medium.com/rl-dqn-deep-q-network-e207751f7ae4

    Here, the critc network is Q hat, i.e. is used for the Q-Learning update step.
    It is iniitalised to be the same as the actor network, and every C optimisation steps,
     it is updated to be the same as the actor network again

    Required args:
        :: model - outputs Q value for each action, so output is of shape [batch, num_actions]

        :: C - as explained above
    """

    dataset_class = QLearningExperienceReplayDataset

    def __init__(
        self, model: nn.Module, training_seq_length=8, discount_factor=0.9, C=10,
    ):

        # Clone network
        actor_network = model
        critic_network = copy.deepcopy(model).eval()

        # Initialise

        dataset = self.dataset_class(training_seq_length=training_seq_length)
        super(QLearningExperienceReplayAgent, self).__init__(
            dataset, actor_network, critic_network
        )

        # TODO: freeze frame critic every C steps, and document in docstring above
        # To keep track of the critic tracking the actor:
        self.C = C
        self.ticker = 0

        self.discount_factor = discount_factor

    def critic_interpretor(self, critic_output):
        """
        Critic is used as the greedy return predictor
        """
        return critic_output.max(-1)

    def step_critic_update(self):
        self.ticker += 1
        if self.ticker == self.C:
            self.ticker = 0
            actor_weights = self.actor_network.state_dict().copy()
            self.critic_network.load_state_dict(actor_weights)
            self.critic_network.eval()

    def get_target_and_pred(self, batch):
        """
        See source link for Q-Learning optimisation step
        """

        # Unpack values from dict
        r_t = batch["r_t"]
        non_terminal_mask = batch["non_terminal_mask"]
        a_t = batch["a_t"]
        s_t = batch["s_t"]  # batch, seq len, 1, obs size
        h_t = batch["h_t"]  # 1, seq len, batch, hidden size
        s_t_dash = batch["s_t_dash"]
        h_t_dash = batch["h_t_dash"]

        # Comes out as a seq len sized list of dicts, e.g. [{"task": Tensor[batch size, num tasks]}, {...]
        # Want to keep this general here, so we pass the processing on to the network itself
        extra = self.actor_network.process_extra_information(batch["extra"])

        # a_t is a set of indices, so use it to index the correct values for this step
        all_q_values, hidden_state = self.actor_network(s_t, h_t, **extra)
        current_policy_q_values = all_q_values.gather(1, a_t.unsqueeze(1)).squeeze()

        # Get the extra term from the critic
        greedy_critic_term = self.get_value(s_t_dash, h_t_dash, **extra)

        # Generate scalar values; for non-terminal transitions only, the extra term is needed
        y_t = r_t + self.discount_factor * non_terminal_mask * greedy_critic_term

        # Make all post processing easier
        batch["extra"] = extra

        # Return target and output for downstream optimisation
        return current_policy_q_values, y_t, batch

    def step_postreward(self, reward, terminated):
        """
        As stated in BaseAgent, the only special parts of the iteration step are after the
        reward, so no need to overwrite step_prereward
        """
        # Log the reward
        self.receive_reward(reward)

        # Only save if a transition has occured yet
        # TODO: review and ensure this is right!
        # Save the new quartet to dataset
        self.dataset.save_from_agent(self, terminated)

        # Select a minibatch of past quartets. This is a list of dicts, see BaseDataset
        batch = self.dataset.sample_batch()

        # Output optimisation loss based on these targets
        current_policy_q_values, y, batch = self.get_target_and_pred(batch)

        # After all optimisation outputted, check if we should update critic network
        self.step_critic_update()

        # Return the training material for this step.
        # This can then be used for a loss function externally
        return current_policy_q_values, y, batch
