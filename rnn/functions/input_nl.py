import torch
from torch import Tensor as _T


def make_funky_repeated_input_nonlinearity():
    raise Exception('make_funky_repeated_input_nonlinearity deprecated - do it yourself')
    def funky_repeated_input_nonlinearity(self, h: _T):
        thetas = self.input_thetas.masked_weight
        assert torch.all(thetas >= 0.0)
        exc_input = (thetas[0]) * torch.exp((thetas[2]) * torch.log(h + (thetas[1])))
        return exc_input.repeat(1, 2, 1)    # i.e. same input provided to both E and I
    return funky_repeated_input_nonlinearity

def make_linear_input_projection():
    raise Exception('make_linear_input_projection deprecated - just use self.W_input')
    def linear_input_projection(self, h: _T):
        return self.W_input(h)
    return linear_input_projection
