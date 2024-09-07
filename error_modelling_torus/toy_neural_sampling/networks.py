from torch import nn
import torch
from torch.nn.functional import softplus, sigmoid, relu
from torch import Tensor as _T

from purias_utils.util.error_modelling import div_norm



class ExplicitMixtureModelNetwork(nn.Module):
    
    "Receive multimodal distribution parameters and output samples from it"

    non_lins_dict = {
        'relu': relu,
        'softplus': softplus,
        'sigmoid': sigmoid,
        'div_norm': div_norm,
        'identity': lambda x: x
    }

    def __init__(self, N: int, hidden_layer_sizes: list[int], non_lins: list[str], output_scaler: float) -> None:
        super().__init__()

        self.W_in_pi = nn.Linear(N+1, hidden_layer_sizes[0], bias = False)
        self.W_in_mu = nn.Linear(N, hidden_layer_sizes[0], bias = False)
        self.W_in_kp = nn.Linear(N, hidden_layer_sizes[0], bias = True)

        self.feedforward_layers = nn.ModuleList(
            [nn.Linear(a, b, bias = True) 
            for a, b in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])]
        )
        self.L = len(self.feedforward_layers)

        self.noise_input_layers = nn.ModuleList(
            [nn.Linear(a, a, bias = False) for a in hidden_layer_sizes[1:]]
        )

        # assert len(non_lins) == len(hidden_layer_sizes) - 1
        self.non_lins = non_lins

        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias = True)
        self.output_scaler = output_scaler

    def noisy_layer_forward(self, l, x_l_minus_1: _T) -> _T:
        "Expect x_l_minus_1 of shape [batch, trials, layer size]"
        feed_forward = self.feedforward_layers[l](x_l_minus_1)
        noise_in = torch.randn_like(feed_forward).to(feed_forward.device)
        random = self.noise_input_layers[l](noise_in)
        non_lin = self.non_lins_dict[self.non_lins[l]]
        pre_ac = feed_forward + random
        return non_lin(pre_ac), pre_ac

    def input_forward(self, pis, mus, kps):
        x = self.W_in_pi(pis) + self.W_in_mu(mus) + self.W_in_kp(kps) 
        return x

    def forward(self, pis, mus, kps, num_trials, return_pre_acs = False) -> _T:
        "All parameters of shape [batch, N+1]"
        pre_acs = []
        x = self.input_forward(pis, mus, kps)
        x = x.unsqueeze(1).repeat(1, num_trials, 1)
        for l in range(self.L):
            x, pre_ac = self.noisy_layer_forward(l, x)
            if return_pre_acs:
                pre_acs.append(pre_ac)
        x = self.output_layer(x).squeeze(-1)
        return (x, pre_acs) if return_pre_acs else x



class CircularExplicitMixtureModelNetwork(ExplicitMixtureModelNetwork):

    def __init__(self, N: int, hidden_layer_sizes: list[int], non_lins: list[str]) -> None:
        super().__init__(N, hidden_layer_sizes, non_lins, output_scaler = 1.0)
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 2, bias = True)

    def forward(self, pis, mus, kps, num_trials, return_pre_acs = False) -> _T:
        all_outputs = super().forward(pis, mus, kps, num_trials, return_pre_acs)
        if return_pre_acs:
            twod_output = all_outputs[0]
            return torch.arctan2(*twod_output.movedim(-1, 0)), all_outputs[1]
        else:
            return torch.arctan2(*all_outputs.movedim(-1, 0))



class CircularPalimpsestMixtureModelNetwork(ExplicitMixtureModelNetwork):
    
    def __init__(self, input_size, hidden_layer_sizes: list[int], non_lins: list[str]) -> None:
        super().__init__(0, hidden_layer_sizes, non_lins, output_scaler = 1.0)

        del self.W_in_pi
        del self.W_in_mu
        del self.W_in_kp
        self.W_in = nn.Linear(input_size, hidden_layer_sizes[0], bias = True)
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 2, bias = True)

    def input_forward(self, sensory_input):
        x = self.W_in(sensory_input)
        return x

    def forward(self, sensory_input, num_trials, return_pre_acs = False) -> _T:
        "sensory_input of shape [batch, input size]"
        pre_acs = []
        x = self.input_forward(sensory_input)
        x = x.unsqueeze(1).repeat(1, num_trials, 1)
        for l in range(self.L):
            x, pre_ac = self.noisy_layer_forward(l, x)
            if return_pre_acs:
                pre_acs.append(pre_ac)
        x = self.output_layer(x).squeeze(-1)
        x = torch.arctan2(*x.movedim(-1, 0))
        return (x, pre_acs) if return_pre_acs else x


class CircularPPCMixtureModelNetwork(CircularExplicitMixtureModelNetwork):

    def __init__(
        self, N: int, d_r: int, hidden_in_layer_sizes: list[int], hidden_out_layer_sizes: list[int],
        non_lins_in: list[str], combination_function: str, non_lins_out: list[str]
    ):

        super(ExplicitMixtureModelNetwork, self).__init__()

        self.N = N
        self.combination_function = combination_function

        # (shared) input network
        self.feedforward_layers_in = nn.ModuleList(
            [nn.Linear(d_r, hidden_in_layer_sizes[0], bias = True)] +
            [nn.Linear(a, b, bias = True) 
            for a, b in zip(hidden_in_layer_sizes[:-1], hidden_in_layer_sizes[1:])]
        )
        self.L_in = len(self.feedforward_layers_in)
        assert len(non_lins_in) == len(hidden_in_layer_sizes)
        self.non_lins_in = non_lins_in

        # (single) output network
        self.feedforward_layers = nn.ModuleList(
            [nn.Linear(hidden_in_layer_sizes[-1], hidden_out_layer_sizes[0], bias = True)] +
            [nn.Linear(a, b, bias = True) 
            for a, b in zip(hidden_out_layer_sizes[:-1], hidden_out_layer_sizes[1:])]
        )
        self.L = len(self.feedforward_layers)
        self.noise_input_layers = nn.ModuleList(
            [nn.Linear(a, a, bias = False) for a in hidden_out_layer_sizes]
        )
        assert len(non_lins_out) == len(hidden_out_layer_sizes)
        self.non_lins = non_lins_out

        self.output_layer = nn.Linear(hidden_out_layer_sizes[-1], 2, bias = True)

        self.div = 1.0

    def input_forward(self, ppc_reprs):
        "ppc_means of shape [batch, d_r, N], sample from them M times then, output of shape [batch, trials = M, d_0, N]"
        x = ppc_reprs.movedim(-1, -2)
        for l in range(self.L_in):
            x = self.feedforward_layers_in[l](x)
            non_lin = self.non_lins_dict[self.non_lins_in[l]]
            x = non_lin(x)
        x = x.movedim(-1, -2)
        return x

    def forward(self, ppc_means, num_trials) -> _T:
        "ppc_means of shape [batch, d_r, N]"
        assert ppc_means.shape[-1] == self.N
        ppc_means = ppc_means.unsqueeze(1).repeat(1, num_trials, 1, 1)
        ppc_reprs = torch.poisson(ppc_means) / self.div
        x_by_N = self.input_forward(ppc_means)              # [batch, trials, d_0, N]
        x = self.combination_function(x_by_N)               # [batch, trials, d_0]
        for l in range(self.L):
            x, preac = self.noisy_layer_forward(l, x)
        twod_output = self.output_layer(x).squeeze(-1)                # [batch, trials, (1)]
        return {
            'ppc_reprs': ppc_reprs,
            'samples': torch.arctan2(*twod_output.movedim(-1, 0)),
        }



