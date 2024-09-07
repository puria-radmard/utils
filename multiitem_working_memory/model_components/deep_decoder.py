import torch
from torch import nn, Tensor as _T


class CueDecoder(nn.Module):
    """
    See archive for old input types - currently only using hidden - any var
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_reprs: _T, cue_reprs: _T):
        """
        Inputs shapes:s
            :hidden_reprs   [batch, trials, time, size]
            :cue_reprs      [batch, num stim, size]

        Final size for both: [batch, trials, time, num stim, size]
        """
        _, tr, T, _ = hidden_reprs.shape
        N = cue_reprs.shape[1]
        
        hidden_reprs = hidden_reprs.unsqueeze(3).repeat(1, 1, 1, N, 1)
        cue_reprs = cue_reprs.unsqueeze(1).unsqueeze(1).repeat(1, tr, T, 1, 1)

        return self.forward_pass(hidden_reprs, cue_reprs)

    def forward_pass(self, stim, cue):
        raise NotImplementedError

        

class HadamardCueDecoder(CueDecoder):
    """
    Starts with an elementwise multiplication of (projections of) the stimulus array 
    and cue representations together
    """
    def __init__(
        self, 
        rnn_representation_size: int,
        hidden_layer_sizes: list[int] = [],
        estimated_feature_size: int = 2,
    ) -> None:
        
        super().__init__()

        ext_hidden_layer_sizes = [efs for efs in hidden_layer_sizes]
        ext_hidden_layer_sizes.append(estimated_feature_size)

        layers = [
            nn.Linear(rnn_representation_size, ext_hidden_layer_sizes[0]),
            nn.Sigmoid()
        ]
        for i, hls in enumerate(ext_hidden_layer_sizes[:-1], 1):
            layers.append(nn.Linear(hls, ext_hidden_layer_sizes[i]))
            layers.append(nn.Sigmoid())

        # Remove last sigmoid
        self.architecture = nn.Sequential(*layers[:-1])

    def forward_pass(self, stim, cue):
        hadamard = stim * cue
        return self.architecture(hadamard)  # [batch, trials, time, stim, output (2)]



class ParallelCueDecoder(CueDecoder):
    """
    Starts with a single parallel layer for stim and for cue, then concatenates them
    and passes through some hidden layers
    """
    def __init__(
        self, 
        rnn_representation_size: int,
        hidden_layer_sizes: list[int] = [],
        estimated_feature_size: int = 2,
        **kwargs
    ) -> None:
        
        super().__init__()

        hidden_layer_sizes = hidden_layer_sizes + [estimated_feature_size]  # Dont use append here!

        layers = [
            nn.Linear(2 * rnn_representation_size, hidden_layer_sizes[0]),
            nn.Softplus() # nn.LeakyReLU()
        ]
        for i, hls in enumerate(hidden_layer_sizes[:-1], 1):
            layers.append(nn.Linear(hls, hidden_layer_sizes[i]))
            layers.append(nn.Softplus())

        # Remove last sigmoid
        self.architecture = nn.Sequential(*layers[:-1])

    def forward_pass(self, stim, cue):
        shared_rep = torch.concat([stim, cue], -1)  # [batch, trial, time, num_stim, 2 * nh]
        return self.architecture(shared_rep)        # [batch, trial, time, num_stim, 2]

