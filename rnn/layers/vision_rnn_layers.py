from torch import nn

class VisionRNNReadoutLayer(nn.Module):

    """
    Expects x of shape: [B, T, P_dim, C_dim, num_channels (1)]
    Reshape to [B, T, P_dim * C_dim * num_channels]
    Project to [B, T, num_classes]
    Apply softmax
    """

    def __init__(self, P_dim: int, C_dim: int, num_classes: int, num_channels: int, bias = False):
        super(VisionRNNReadoutLayer, self).__init__()
        self.input_size = P_dim * C_dim * num_channels
        self.linear = nn.Linear(self.input_size, num_classes, bias = False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        "See docstring"
        B, T, *_ = x.shape
        x_flattened = x.reshape(B, T, self.input_size)
        return self.softmax(self.linear(x_flattened))


class VisionRNNInputLayer(nn.Module):

    """
    Inputs come in in shape [batch, image_size, image_size, num_channels]
    We need to output to size [batch, P, C, 1 (for channels)]

    This is done by a projection:
        image_size * image_size * num_channels --> dimC * dimP
    with a bias
    """

    def __init__(self, image_size: int, num_channels: int, P_dim: int, C_dim: int, hscale: float):
        super(VisionRNNInputLayer, self).__init__()
        self.linear = nn.Linear(
            image_size * image_size * num_channels,
            P_dim * C_dim,
            bias = True
        )
        self.P, self.C, self.hscale = P_dim, C_dim, hscale

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.linear(x)
        return self.hscale * x.reshape(B, self.P, self.C, 1)
