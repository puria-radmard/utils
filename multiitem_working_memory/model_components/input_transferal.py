"""
After finding sensory neuron activation (see sensory_neurons.py), which is done per stimulus on the board,
this file offers some ways to project these activations into the network model

This methods vary fairly dramatically...
"""

import torch
from torch import nn
from torch import Tensor as _T
from torch.distributions import Poisson
from scipy.stats import ortho_group
import numpy as np
from typing import Callable

from purias_utils.rnn.layers.scalar_layers import ScalarLayer



class PalimpsestStimulusCombiner(nn.Module):
    """
    Like a DDC, with p(m) = uniform, as noted in comments

    combiner_types [where s is sensory output]:
        # a) agg ( sigma ( W @ s ) ) where s is sensory output (this is the original)
        # b) sigma ( agg ( W @ s ) )
        # c) agg ( W @ s )
    """
    def __init__(self, sensory_pop_size, output_size, combiner_type, agg_method = 'mean'):
        super(PalimpsestStimulusCombiner, self).__init__()

        assert combiner_type in ['a', 'b', 'c'], combiner_type
        self.combiner_type = combiner_type

        self.input_proj: Callable[[_T], _T] = nn.Linear(sensory_pop_size, output_size, bias = False)

        self.agg_method = agg_method
        assert agg_method in ['mean', 'sum']

        fixation_token = torch.randn(output_size) / output_size
        self.register_parameter(name='fixation_token', param = torch.nn.parameter.Parameter(fixation_token))

    def tanh(self, input: _T) -> _T:
        "Annoyingly have to do this myself"
        exp_two_input = (2 * input).exp()
        return (exp_two_input - 1) / (exp_two_input + 1)

    # def masked_mean(self, stimulus_seperated: _T, stimulus_mask: _T):
    #     if stimulus_mask is not None:
    #         stimulus_mask = stimulus_mask.to(stimulus_seperated.device)
    #         total = (stimulus_seperated * stimulus_mask).sum(-2)
    #         counts = stimulus_mask.sum(-2)
    #         return total/counts
    #     else:
    #         return stimulus_seperated.mean(-2)

    def agg(self, stimulus_seperated: _T):
        "Comes in as [batch, ... stim, n_h]"
        return stimulus_seperated.mean(-2) if self.agg_method == 'mean' else stimulus_seperated.sum(-2)

    def forward(self, sensory_response: _T, add_fixation: bool):
        """
        sensory_response comes in as shape [..., num stimuli, sensory size]
        """

        # Projection 
        projected_sensory_response = self.input_proj(sensory_response)

        # Non-linearity and meaning
        if self.combiner_type == 'a':
            stimulus_response = self.agg(self.tanh(projected_sensory_response))
        elif self.combiner_type == 'b':
            stimulus_response = self.tanh(self.agg(projected_sensory_response))
        elif self.combiner_type == 'c':
            stimulus_response = self.agg(projected_sensory_response)

        if add_fixation:
            stimulus_response += self.fixation_token

        ### Sample from Poisson then reduce
        ## return Poisson(stimulus_response).sample() * self.dt
        return stimulus_response
