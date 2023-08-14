import torch
from torch import nn, Tensor as T


class CueDecoder(nn.Module):
    """
    Takes in a combined representation of the full stimulus array, and a representation of
    the feature cue, and outputs an estimate of the cued object's relevant feature

    :input_type
        'sensory' - takes in two :sensory_size-sized vectors and projects them to :rnn_representation_size
                i.e. the inputs are shaped [batch, rnn_representation_size]
        'input' - takes in two :rnn_representation_size-sized vectors,
                which should be the same ones provided by the combiner to the RNN itself
                i.e. the inputs are shaped [batch, rnn_representation_size]
        'hidden' - takes in two :rnn_representation_size-sized vectors, 
                which should be the hidden states of the network during the delay period.
                Expects input with an added temporal tensor dimension.
                i.e. the inputs are shaped [batch, time, rnn_representation_size]

        (*) In all cases, cue is shaped like the sensory size, and a linear + tanh layer is applied
            to make it the same size as the RNN representation, if needed

    :rnn_representation_size
        The name of this parameter only makes sense if input_type is not sensory
        It is just the output size of the projections described above
        Named so purely for the (*) sentence
    """

    allowed_input_types = ['sensory', 'input', 'hidden']

    def __init__(self, input_type: str, rnn_representation_size: int, sensory_size: int = 0) -> None:
        super().__init__()

        assert input_type in self.allowed_input_types
        self.input_type = input_type
        if input_type == 'sensory':
            self.cue_input_proj = nn.Linear(sensory_size, rnn_representation_size)
            self.stim_input_proj = nn.Linear(sensory_size, rnn_representation_size)
        else:
            self.stim_input_proj = None
            self.cue_input_proj = nn.Linear(sensory_size, rnn_representation_size)
        
        self.n_h = rnn_representation_size
        self.sensory_size = sensory_size

    def stim_repr_entry(self, x: T):
        """
        See __init__ documentation
        A time dimension is added if it doesn't exist already
        """
        if self.input_type == 'sensory':
            assert list(x.shape) == [x.shape[0], self.sensory_size]
            return self.stim_input_proj(x).unsqueeze(1).tanh()
        elif self.input_type == 'input':
            assert list(x.shape) == [x.shape[0], self.n_h]
            return x.unsqueeze(1)
        elif self.input_type == 'hidden':
            assert list(x.shape) == [x.shape[0], x.shape[1], self.n_h]
            return x

    def cue_repr_entry(self, x: T):
        """
        See forward documentation.
        A time dimension is added if it doesn't exist already
        """
        if self.input_type == 'sensory':
            assert list(x.shape) == [x.shape[0], x.shape[1], self.sensory_size]
            return self.cue_input_proj(x).unsqueeze(2).tanh()
        elif self.input_type == 'input':
            assert list(x.shape) == [x.shape[0], x.shape[1], self.sensory_size]
            return x.unsqueeze(2)
        elif self.input_type == 'hidden':
            assert list(x.shape) == [x.shape[0], x.shape[1], x.shape[2], self.sensory_size]
            return x

    @staticmethod
    def flatten_stim(stim_reprs: T, num_stims: list[int]):
        "[batch, (time), dim] --> [batch * num_stim, (time), dim]"
        output = []
        for num_stim, stim_repr in zip(num_stims, stim_reprs):        # [(time), dim]
            new_set = torch.stack([stim_repr for _ in range(num_stim)], 0)    # [num_stim, (time), dim]
            output.append(new_set)      # i.e. same batch stimuli are next to each other
        return torch.concat(output, 0)
    
    @staticmethod
    def flatten_cue(cue_reprs: T, num_stims: list[int]):
        "[batch, max_stim, (time), dim] --> [batch * num_stim, (time), dim]"
        output = []
        for num_stim, cue_repr in zip(num_stims, cue_reprs):          # [max_stim, (time), dim]
            actual_set = cue_repr[:num_stim]                            # [num_stim, (time), dim]
            output.append(actual_set)
        return torch.concat(output, 0)

    @staticmethod
    def flatten_target(targets: T, num_stims: list[int]):
        "[batch, max_stim, ...] --> [batch * num_stim, 1, ...]"
        return CueDecoder.flatten_cue(targets, num_stims)   

    def forward(self, stim_reprs: T, cue_reprs: T, num_stims: list[int] = None):
        """
        Inputs shapes:
            :stim_reprs as in __init__ documentation
            :cue_reprs  [batch, maximum num stim, (time), rnn_representation_size or sensory_size]
            :num_stims  len = batch
        """
        if num_stims == None:
            num_stims = [cue_reprs.shape[1] for _ in range(cue_reprs.shape[0])]

        stim_reprs = self.stim_repr_entry(stim_reprs)
        cue_reprs = self.cue_repr_entry(cue_reprs)

        # Make both of them [batch * num_stim, time, dim], where num_stim depends on the task variable
        flattened_stim_reprs = self.flatten_stim(stim_reprs, num_stims)
        flattened_cue_reprs = self.flatten_cue(cue_reprs, num_stims)

        return self.forward_pass(flattened_stim_reprs, flattened_cue_reprs)

    def forward_pass(self, stim, cue):
        raise NotImplementedError

        

class HadamardCueDecoder(CueDecoder):
    """
    Starts with an elementwise multiplication of (projections of) the stimulus array 
    and cue representations together
    """
    def __init__(
        self, 
        input_type: str,
        rnn_representation_size: int,
        sensory_size: int = 0,
        hidden_layer_sizes: list[int] = [],
        estimated_feature_size: int = 2
    ) -> None:
        
        super().__init__(input_type, rnn_representation_size, sensory_size)

        hidden_layer_sizes.append(estimated_feature_size)

        layers = [
            nn.Linear(rnn_representation_size, hidden_layer_sizes[0]),
            nn.Sigmoid()
        ]
        for i, hls in enumerate(hidden_layer_sizes[:-1], 1):
            layers.append(nn.Linear(hls, hidden_layer_sizes[i]))
            layers.append(nn.Sigmoid())
        
        # Remove last sigmoid
        self.architecture = nn.Sequential(*layers[:-1])

    def forward_pass(self, stim, cue):
        hadamard = torch.sigmoid(stim * cue)
        return self.architecture(hadamard)  # [batch * stim, time, 1]


class ParallelCueDecoder(CueDecoder):
    """
    Starts with a single parallel layer for stim and for cue, then concatenates them
    and passes through some hidden layers
    """
    def __init__(
        self, 
        input_type: str,
        rnn_representation_size: int,
        sensory_size: int = 0,
        hidden_layer_sizes: list[int] = [],
        estimated_feature_size: int = 2
    ) -> None:
        
        super().__init__(input_type, rnn_representation_size, sensory_size)
        
        assert len(hidden_layer_sizes) >= 1, "Require at least the parallel layer size for ParallelCueDecoder"
        hidden_layer_sizes.append(estimated_feature_size)

        self.stim_branch = nn.Sequential(
            nn.Linear(rnn_representation_size, hidden_layer_sizes[0]),
            nn.Sigmoid()
        )

        self.cue_branch = nn.Sequential(
            nn.Linear(rnn_representation_size, hidden_layer_sizes[0]),
            nn.Sigmoid()
        )

        shared_layers = [
            nn.Linear(2 * hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.Sigmoid()
        ]
        for i, hls in enumerate(hidden_layer_sizes[1:-1], 2):
            shared_layers.append(nn.Linear(hls, hidden_layer_sizes[i]))
            shared_layers.append(nn.Sigmoid())

        # Remove last sigmoid
        self.shared_architecture = nn.Sequential(*shared_layers[:-1])

    def forward_pass(self, stim, cue):
        cue_rep = self.cue_branch(cue)
        stim_rep = self.stim_branch(stim).unsqueeze(1).repeat(1, cue_rep.shape[1], 1)
        shared_rep = torch.concat([stim_rep, cue_rep], 2)
        return self.shared_architecture(shared_rep)
