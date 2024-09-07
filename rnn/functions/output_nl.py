from torch import Tensor as _T


def linear_output_projection(self, h: _T):
    "misnomer"
    return self.W_feedback(h)
