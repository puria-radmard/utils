"""
After finding sensory neuron activation (see sensory_neurons.py), which is done per stimulus on the board,
this file offers some ways to project these activations into the network model

This methods vary fairly dramatically...
"""

import torch
from torch import nn
from torch import Tensor as T
from torch.distributions import Poisson
from scipy.stats import ortho_group
import numpy as np

class DDCStyleStimulusCombiner(nn.Module):
    """
    Like a DDC, with p(m) = uniform, as noted in comments
    """
    def __init__(self, sensory_pop_size, output_size, dt = 0.001):
        super(DDCStyleStimulusCombiner, self).__init__()
        self.projection_weights = nn.Linear(sensory_pop_size, output_size, bias = False)
        self.nonlin = nn.Tanh()
        self.dt = dt

        fixation_token = torch.randn(output_size) / output_size

        self.register_parameter(name='fixation_token', param = torch.nn.parameter.Parameter(fixation_token))

    def forward(self, sensory_response: T, add_fixation: T):
        "sensory_response comes in as shape [num stimuli, neuron population size]"
        # Check for nasty effects from Poisson sampling
        assert not sensory_response.requires_grad

        # Project and apply nonlinearity to each response
        stimulus_responses = self.nonlin(self.projection_weights(sensory_response))

        # Mean over functions, with a uniform likelihood over tuning curves for now
        stimulus_response = stimulus_responses.mean(1)

        if add_fixation:
            stimulus_response += self.fixation_token

        ### Sample from Poisson then reduce
        ## return Poisson(stimulus_response).sample() * self.dt
        return stimulus_response



class QRSlotStimulusCombiner(nn.Module):

    def __init__(self, sensory_pop_size, output_size, num_stimuli: int) -> None:
        super(QRSlotStimulusCombiner, self).__init__()

        input_weight = nn.init.xavier_uniform(torch.zeros(sensory_pop_size, output_size))
        fixation_token = torch.randn(output_size) / output_size

        self.register_parameter(name='input_weight', param = torch.nn.parameter.Parameter(input_weight))
        self.register_parameter(name='fixation_token', param = torch.nn.parameter.Parameter(fixation_token))

        self.num_stimuli = num_stimuli
        assert output_size % num_stimuli == 0.0
        self.cols_per_stimulus = output_size // num_stimuli

    @property
    def orthogonalised_weight(self):
        qr = torch.linalg.qr(self.input_weight.T)
        return qr.Q, qr.R

    def mask_stim_weight(self, q, r, idx):
        slicer = slice(self.cols_per_stimulus * idx, self.cols_per_stimulus * (idx + 1))
        masked_q = torch.zeros_like(q)
        masked_q[:,slicer] = q[:,slicer]
        return (masked_q @ r).T

    def forward(self, sensory_response: T, add_fixation: bool):
        "Expect sensory_response of shape [batch, stim, sensory population]"
        assert sensory_response.shape[1] == self.num_stimuli
        total_network_input = 0.
        q, r = self.orthogonalised_weight
        for i in range(self.num_stimuli):
            masked_weight = self.mask_stim_weight(q, r, i)
            total_network_input += (sensory_response[:,i] @ masked_weight)  # [batch, hidden (output) size]
        if add_fixation:
            total_network_input += self.fixation_token
        return total_network_input



class SoftQRStimulusCombiner(nn.Module):

    def __init__(self, sensory_pop_size, output_size, num_stimuli: int) -> None:
        super(SoftQRStimulusCombiner, self).__init__()

        q = ortho_group(output_size)
        self.register_parameter(name='q', param = torch.nn.parameter.Parameter(q))
        
        r_full = torch.zeros(sensory_pop_size, output_size)
        r_full = torch.nn.init.xavier_uniform(r_full)
        self.register_parameter(name='r_full', param = torch.nn.parameter.Parameter(q))

    @property
    def r(self):
        return torch.triu(self.r_full)

    @property
    def weight(self):
        return self.q @ self.r
    
    def orth_reg_loss(self):
        return (self.q @ self.q.T - torch.eye(self.q.shape[0])).mean()

    def forward(self, sensory_input):
        return self.weight @ sensory_input



class StripyStimulusCombiner(nn.Module):
    """
    First combine the (fixed) N stimuli using the 'stripy' matrix (parameterised by alpha; alpha = 0 means identity)
    Then project to the output size with a freely parameterised projection matrix
        Alternatively, set identity_project = True to output just the stripy combined response
    """

    def __init__(self, sensory_pop_size, output_size, num_stimuli, alpha, identity_project = False) -> None:

        super(StripyStimulusCombiner, self).__init__()

        assert 0. <= alpha <= 1.

        fixation_token = torch.randn(output_size) / output_size
        self.register_parameter(name='fixation_token', param = torch.nn.parameter.Parameter(fixation_token))
        
        id_mat = np.eye(num_stimuli * sensory_pop_size)
        stripe_mat_block = np.eye(sensory_pop_size) / num_stimuli
        stripe_mat_block_row = np.concatenate([stripe_mat_block for _ in range(num_stimuli)], axis = 0)
        stripe_mat = np.concatenate([stripe_mat_block_row for _ in range(num_stimuli)], axis = 1)
        self.combining_matrix = torch.tensor(alpha * stripe_mat + (1. - alpha) * id_mat).float()
        self.alpha = alpha

        if identity_project:
            self.input_weight = torch.eye(num_stimuli * sensory_pop_size)
            assert output_size == sensory_pop_size
        else:
            input_weight = nn.init.xavier_uniform(torch.zeros(num_stimuli * sensory_pop_size, output_size))
            self.register_parameter(name='input_weight', param = torch.nn.parameter.Parameter(input_weight))

    def forward(self, sensory_input, add_fixation: bool):
        "sensory_input should come in [batch, stim, pop]"
        
        # Stack individual stimuli responses for each case -> [batch, stim * pop]
        stacked_inputs = torch.stack(
                [torch.concat(list(batch_item), dim=0
            ) for batch_item in sensory_input],
            0
        )

        # Combine using slider
        combined_inputs = stacked_inputs @ self.combining_matrix.float()

        # Project to hidden space
        projection = combined_inputs @ self.input_weight

        if add_fixation:
            projection += self.fixation_token
        
        return projection