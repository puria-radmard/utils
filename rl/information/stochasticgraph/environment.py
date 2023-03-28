from __future__ import annotations

import numpy as np
from typing import List

from purias_utils.rl.information.stochasticgraph.utils import sample_from_categorical


class StochasticGraph:

    def __init__(self):
        self.nodes: List[StochasticGraphNode] = []
        self.current_node = 0

    def add_node(self, new_node: StochasticGraphNode):
        assert new_node.all_neighbours_assigned
        self.nodes.append(new_node)
        new_node.index_in_graph = len(self.nodes) - 1
        for neighbour_node in new_node.neighbours:
            if not neighbour_node.assigned_to_graph:
                self.add_node(neighbour_node)

    def generate_adjacency_matrix(self):
        n = len(self.nodes)
        a_matrix = np.zeros([n, n])
        for i in range(n):
            for neigbour in self.nodes[i].neighbours:
                assert neigbour.assigned_to_graph
                j = neigbour.index_in_graph
                a_matrix[i,j] = 1
                a_matrix[j,i] = 1
        return a_matrix

    def take_step(self, action: int):
        current_node = self.nodes[self.current_node]
        transition_matrix = current_node.transition_matrix
        assert 0 <= action < current_node.num_a

        # Get next node as an index of current node neighbours
        transition_categorical = transition_matrix[action]
        neighbour_idx = sample_from_categorical(transition_categorical)
        
        # Get next node as 'absolute' index and also reward
        reward = current_node.reward_distributions[neighbour_idx].sample_reward()
        self.current_node = current_node.neighbours[neighbour_idx].index_in_graph
        return ((neighbour_idx, self.current_node), reward)

    def reset(self):
        self.current_node = 0
        for node in self.nodes:
            node.reset()

        

class StochasticGraphNode:
    """
    One node in a StochasticGraph environment.
    The transition_matrix is shaped AxS where:
        A is the number of actions available to the agent from this node
        S is the number of neighbour nodes that this node has
        
        i.e. self.transition_matrix[a] is an S-sized vector that gives the 
            categorical probability that the agent ends up in each neighbour given
            that it takes action a
        
        in the deterministic case, A = S and transition_matrix = I(A)
    """

    def __init__(
        self,
        transition_matrix: np.ndarray
        ):
        self.transition_matrix = transition_matrix
        self.num_a, self.num_neighbours = transition_matrix.shape
        assert transition_matrix.sum() == self.num_a
        self.neighbours: List[StochasticGraphNode] = [None for _ in range(self.num_neighbours)]
        self.reward_distributions: List[StochasticGraphRewardDistribution] = [None for _ in range(self.num_neighbours)]
        self.index_in_graph: int = None

    def assign_neighbour(
        self, 
        neighbour: StochasticGraphNode,
        forward_transition_reward_distribution: StochasticGraphRewardDistribution,
        backward_transition_reward_distribution: StochasticGraphRewardDistribution,
        to_index,
        from_index
        ):

        if neighbour == self:
            assert to_index == from_index

        self.neighbours[to_index] = neighbour
        neighbour.neighbours[from_index] = self

        self.reward_distributions[to_index] = forward_transition_reward_distribution
        neighbour.reward_distributions[from_index] = backward_transition_reward_distribution

    @property
    def all_neighbours_assigned(self):
        return not None in self.neighbours

    @property
    def assigned_to_graph(self):
        return self.index_in_graph is not None

    def __repr__(self) -> str:
        return "node"

    def reset(self):
        for rd in self.reward_distributions:
            rd.reset()



class StochasticGraphRewardDistribution:

    def __init__(self) -> None:
        pass

    def sample_reward(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class ConstantStochasticGraphRewardDistribution:

    def __init__(self, reward_value) -> None:
        self.reward_value = reward_value

    def sample_reward(self):
        return self.reward_value

    def reset(self):
        pass
