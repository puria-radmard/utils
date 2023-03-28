from torch import Tensor as T


def linear_output_projection(self, h: T):
    "misnomer"
    return self.W_feedback(h)
