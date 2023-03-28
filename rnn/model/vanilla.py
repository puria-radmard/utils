import torch
from torch import nn

raise Exception('deprecated')

class BasicRNN(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int, output_size: int, lag_constant: float):
        super(BasicRNN, self).__init__()
        self.input_size = observation_size
        self.hidden_size = hidden_size

        self.input_layer = torch.nn.Linear(observation_size, hidden_size)
        self.hidden_weight = torch.nn.Linear(hidden_size, hidden_size)

        self.out_layer = torch.nn.Linear(hidden_size, output_size)

        self.relu = torch.nn.ReLU()

        # Alpha in Yang et al. paper
        self.lag_constant = lag_constant

    def forward(self, obs, h):
        rnn_h_pre_act = self.input_layer(obs) + self.hidden_weight(h)
        rnn_h_addition = self.lag_constant * self.relu(rnn_h_pre_act)
        rnn_h = rnn_h_addition + ((1 - self.lag_constant) * h)
        out = self.out_layer(rnn_h)
        # Torch adds in a dimension at the start
        return out.squeeze(), rnn_h.squeeze()

    def forward_sequence(self, obs_seq, h0, seq_dim = 1):
        out_history, h_history = [], []
        h = h0
        for t in range(obs_seq.shape[seq_dim]):
            t = torch.tensor(t)
            obs = torch.index_select(obs_seq, seq_dim, t).squeeze()
            out, h = self(obs=obs, h=h)
            out_history.append(out)
            h_history.append(h)
        return (
            torch.stack(out_history, seq_dim),    # [batch, time, output]
            torch.stack(h_history, seq_dim),      # [batch, time, hidden]
        )



class NNWithTaskSwitcher(BasicRNN):
    def __init__(self, observation_size, hidden_size, output_size, num_tasks):
        super(NNWithTaskSwitcher, self).__init__(
            observation_size=observation_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )
        self.switcher_layer = torch.nn.Linear(num_tasks, hidden_size)
        self.relu = torch.nn.ReLU()

    def forward(self, obs, h, task_idx_ohe):
        rnn_h_pre_act = (
            self.input_layer(obs)
            + self.hidden_weight(h)
            + self.switcher_layer(task_idx_ohe)
        )
        rnn_h = self.relu(rnn_h_pre_act)
        out = self.out_layer(rnn_h)
        return out, rnn_h

    def forward_sequence(self, obs_seq, h0, task_idx_ohe):
        out_history, h_history = [], []
        h = h0
        for obs in obs_seq:
            out, h = self(obs=obs, h=h, task_idx_ohe=task_idx_ohe)
            out_history.append(out)
            h_history.append(h)
        import pdb; pdb.set_trace()
        return out_tensor, h_tensor
