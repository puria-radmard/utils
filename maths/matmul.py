from typing import Callable
from torch import Tensor as _T

def multiply_2d_state_safely(func: Callable, act: _T):
    """
        input state of shape [batch, trial, image1, image2, channels]
        need to combine batch and trials, multiply, then undo combination
    """
    B, tr, Y, X, C = act.shape
    u_reshaped = act.reshape(B * tr, Y, X, C)
    output = func(u_reshaped)
    output = output.reshape(B, tr, Y, X, C)
    return output
