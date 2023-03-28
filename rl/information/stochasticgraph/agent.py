import torch
from purias_utils.rl.information.stochasticgraph.utils import NormalGamma, ExtendedDirichlet
from purias_utils.rl.information.stochasticgraph.environment import *
from purias_utils.rl.information.stochasticgraph.utils import generate_epsilon_greedy_policy


class StochasticGraphAgentBase:
    """
    Tabular policy for known number of states and actions in each state
    self.Qvals is shaped [S, A], 
        such that self.Q[s] are the Q values for actions from state s
    """

    def __init__(
        self,
        graph: StochasticGraph,
        ) -> None:
        
        # Initialise policy, and some will be forever unused!
        num_states = len(graph.nodes)
        num_actions = max([state.num_a for state in graph.nodes])
        self.Qvals = np.zeros([num_states, num_actions])

        self.graph = graph

        self.previous_state: int = None

    def get_Qs_for_state(self, state):
        num_actions = self.graph.nodes[state].num_a
        return self.Qvals[state, :num_actions]    

    def update_state(self):
        # Also returns Qs
        current_node = self.graph.current_node
        self.previous_state = current_node
        return self.get_Qs_for_state(current_node)

    def generate_policy(self, q_values):
        raise NotImplementedError

    def choose_action(self, policy_pmf):
        # This will become the 'action_taken' parameter in self.receive_new_state_and_reward
        return sample_from_categorical(policy_pmf)

    def receive_new_state_and_reward(self, new_state: int, reward_received: float, action_taken: int, *args):
        raise NotImplementedError




class QLearningStochasticGraphAgent(StochasticGraphAgentBase):

    """
    Epsilon greedy Q-learning
    """

    def __init__(self, graph: StochasticGraph, epsilon: float = 0, gamma: float = 0.9, alpha: float = 0.1) -> None:
        super(QLearningStochasticGraphAgent, self).__init__(graph)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def generate_policy(self, q_values):
        # Just epilson greedy for now!
        return generate_epsilon_greedy_policy(q_values, self.epsilon)

    def receive_new_state_and_reward(self, new_state: int, reward_received: float, action_taken: int):
        # Q-learning!
        old_q = self.Qvals[self.previous_state, action_taken]
        target_q = reward_received + self.gamma * self.get_Qs_for_state(new_state).max()
        error_step = self.alpha * (target_q - old_q)
        self.Qvals[self.previous_state, action_taken] += error_step


class CuriousQLearningStochasticGraphAgent(QLearningStochasticGraphAgent):

    """
    Informational reward mixed into main reward.
    Does Q-learning on new reward.

    self.transition_model holds the S,A,S' dirichlets that are updated with each transition
        Same format as self.Qvals, but rows aren't extended to maximum size,
        i.e. self.transition_model[s] is a self.graph.nodes[s].num_a length list of dirichlets over pmfs of size len(self.graph.nodes[s].neighbours)

    self.reward_model holds the distribution over transition reward distributions
        Similar format to self.Qvals, but there's an extra dimension of neighbour size, and nothing is extended to maximum size,
        i.e. self.reward_model[s][a] is a len(self.graph.nodes[s].neighbours) length list of normal-gammas, for all a in range(self.graph.nodes[s].num_a)
    """

    def __init__(self, graph: StochasticGraph, curiousity_weighting: float, epsilon: float = 0, gamma: float = 0.9, alpha: float = 0.1) -> None:
        super(CuriousQLearningStochasticGraphAgent, self).__init__(graph, epsilon, gamma, alpha)

        self.curiousity_weighting = curiousity_weighting

        self.transition_model = [[ExtendedDirichlet(torch.ones(node.num_neighbours)) for _ in range(node.num_a)] for node in self.graph.nodes]
        self.reward_model = [[[NormalGamma(0.0, 1.0, 1.0, 1.0) for _ in range(node.num_neighbours)] for _ in range(node.num_a)] for node in self.graph.nodes]

    def update_model(self, s, a, r, s_new_neighbour_idx):
        """Update distributions over distributions in model"""
        # TODO: double check these are all the right way around!
        current_normal_loc = self.reward_model[s][a][s_new_neighbour_idx].normal_loc  # mu
        current_normal_precision = self.reward_model[s][a][s_new_neighbour_idx].normal_precision  # beta
        current_gamma_rate = self.reward_model[s][a][s_new_neighbour_idx].gamma_rate  # nu
        
        self.reward_model[s][a][s_new_neighbour_idx].normal_loc = (current_normal_precision * current_normal_loc + r) / (current_normal_precision + 1)
        self.reward_model[s][a][s_new_neighbour_idx].gamma_rate = current_gamma_rate + 0.5 * ((current_normal_precision / (current_normal_precision + 1)) * (current_normal_loc - r)**2)

        self.reward_model[s][a][s_new_neighbour_idx].normal_precision += 1        # alpha
        self.reward_model[s][a][s_new_neighbour_idx].gamma_concentration += 0.5   # k
        
        self.transition_model[s][a].concentration[s_new_neighbour_idx] += 1

    def receive_new_state_and_reward(self, new_state: int, reward_received: float, action_taken: int, new_neighbour_idx: int):
        """
        new_neighbour_idx is just the position that new_state holds in the curren state's neighbour list
        This is needed to index the internal reward model
        """

        predictive_transition_entropy = self.transition_model[self.previous_state][action_taken].predictive_entropy()
        predictive_reward_entropy = self.transition_model[self.previous_state][action_taken].mean_array(
            torch.tensor([h_s_dash.predictive_entropy() for h_s_dash in self.reward_model[self.previous_state][action_taken]])
        )

        # TODO: check order here. shouldn't these be after model update?
        average_transition_entropy = self.transition_model[self.previous_state][action_taken].average_entropy()
        average_reward_entropy = self.transition_model[self.previous_state][action_taken].mean_array(
            torch.tensor([h_s_dash.average_entropy() for h_s_dash in self.reward_model[self.previous_state][action_taken]])
        )

        # TODO: Do catch trials without each component
        transition_curiousity_reward = predictive_transition_entropy - average_transition_entropy
        reward_curiousity_reward = predictive_reward_entropy - average_reward_entropy

        curiousity_reward = transition_curiousity_reward + reward_curiousity_reward

        total_reward = reward_received + (self.curiousity_weighting * curiousity_reward)
        
        super(CuriousQLearningStochasticGraphAgent, self).receive_new_state_and_reward(
            new_state, total_reward, action_taken
        )

        self.update_model(self.previous_state, action_taken, reward_received, new_neighbour_idx)

        return curiousity_reward


