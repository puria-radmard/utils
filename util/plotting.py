import numpy as np
from numpy import ndarray as _A
import matplotlib.pyplot as plt


THETA = np.arctan(np.sqrt(2))
PROJECTION_MATRIX = np.array(
    [
        [-1/np.sqrt(2),               1/np.sqrt(2),                0.0          ],
        [-np.sin(THETA) / np.sqrt(2), -np.sin(THETA) / np.sqrt(2), np.cos(THETA)],
    ]
)


SIMPLEX_OUTLINES = [
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
]

def project_to_simplex(vectors: _A):
    "Expecting vectors of shape [..., 3] to project to simplex"
    return vectors @ PROJECTION_MATRIX.T


PROJECTED_SIMPLEX_OUTLINES = [
    project_to_simplex(so) for so in SIMPLEX_OUTLINES
]



def prepare_axes_for_simplex(ax: plt.Axes, vertex_labels, plot_kwargs = {}, label_kwargs = {}):
    ax.axis('off')
    label_has = ['right', 'left', 'center']
    label_vas = ['top', 'top', 'bottom']
    for i, pso in enumerate(PROJECTED_SIMPLEX_OUTLINES):
        ax.plot(*pso.T, color = 'black', **plot_kwargs)
        ax.text(*pso[0], vertex_labels[i], ha=label_has[i], va=label_vas[i], **label_kwargs)


def scatter_to_simplex(ax, vectors, **kwargs):
    projected_vectors = project_to_simplex(vectors)
    ax.scatter(*projected_vectors.T, **kwargs)




def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    
    