import numpy as np
from numpy import ndarray as _A
import matplotlib.pyplot as plt


from typing import Optional
from matplotlib.pyplot import Axes


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
    
    
def legend_without_repeats(*axes):
    for ax in axes:
        # get legend handles and their corresponding labels
        handles, labels = ax.get_legend_handles_labels()

        # zip labels as keys and handles as values into a dictionary, ...
        # so only unique labels would be stored 
        dict_of_labels = dict(zip(labels, handles))

        # use unique labels (dict_of_labels.keys()) to generate your legend
        ax.legend(dict_of_labels.values(), dict_of_labels.keys())




def standard_swap_model_simplex_plots(all_prior_vectors: _A, ax: Axes, all_posteriors_vectors: Optional[_A] = None, ax_posthoc_posterior: Optional[Axes] = None, ax_no_u: Optional[Axes] = None, ax_no_u_posthoc_posterior: Optional[Axes] = None):

    print('Fix simplex_plots to handle >1 models at the same time!, i.e. Q axis exists...')

    prepare_axes_for_simplex(ax, ['Uniform', 'Correct', 'All swaps'], label_kwargs = {})
    if ax_no_u is not None:
        prepare_axes_for_simplex(ax_no_u, ['Correct', 'Largest\nSwap', 'Everything Else'], label_kwargs = {})
    if ax_posthoc_posterior is not None:
        prepare_axes_for_simplex(ax_posthoc_posterior, ['Uniform', 'Correct', 'All swaps'], label_kwargs = {})
    if ax_no_u_posthoc_posterior is not None:
        prepare_axes_for_simplex(ax_no_u_posthoc_posterior, ['Correct', 'Largest\nSwap', 'Everything Else'], label_kwargs = {})

    N = all_prior_vectors.shape[-1] -1

    ##### Make prior and posterior simplex-plottable
    if N == 1:
        swap_combined_pi_vectors = np.concatenate([all_prior_vectors, np.zeros([all_prior_vectors.shape[0], 1])], 1)
        maximum_swap_pi_vectors = swap_combined_pi_vectors[:,[1, 0, 2]]

        if all_posteriors_vectors is not None:
            swap_combined_post_hoc_posteriors = np.concatenate([all_posteriors_vectors, np.zeros(all_posteriors_vectors.shape[0], 1)], 1)
            maximum_swap_post_hoc_posteriors = swap_combined_post_hoc_posteriors[:,[1, 0, 2]]

    elif N > 1:
        pi_vectors_first = all_prior_vectors[:,:2]
        pi_vectors_second = all_prior_vectors[:,2:].sum(-1, keepdims = True)
        swap_combined_pi_vectors = np.concatenate([pi_vectors_first, pi_vectors_second], axis=1)

        maximum_swap_pi_vectors = np.zeros_like(swap_combined_pi_vectors)
        maximum_swap_pi_vectors[:,0] = all_prior_vectors[:,1]  # Correct
        maximum_swap_pi_vectors[:,1] = all_prior_vectors[:,2:].max(-1) # maximum swap
        maximum_swap_pi_vectors[:,2] = 1.0 - maximum_swap_pi_vectors.sum(-1)

        if all_posteriors_vectors is not None:
            post_hoc_posteriors_first = all_posteriors_vectors[:,:2]
            post_hoc_posteriors_second = all_posteriors_vectors[:,2:].sum(-1, keepdims = True)
            swap_combined_post_hoc_posteriors = np.concatenate([post_hoc_posteriors_first, post_hoc_posteriors_second], axis=1)

            maximum_swap_post_hoc_posteriors = np.zeros_like(swap_combined_post_hoc_posteriors)
            maximum_swap_post_hoc_posteriors[:,0] = all_posteriors_vectors[:,1]  # Correct
            maximum_swap_post_hoc_posteriors[:,1] = all_posteriors_vectors[:,2:].max(-1) # maximum swap
            maximum_swap_post_hoc_posteriors[:,2] = 1.0 - maximum_swap_post_hoc_posteriors.sum(-1)   


    ##### Plot batch to simplex
    scatter_to_simplex(ax, swap_combined_pi_vectors, alpha = 0.2, label = f'{N}')
    ax.set_title('Generative model priors\n$\pi_\\beta(Z)$')
    if ax_no_u is not None:
        scatter_to_simplex(ax_no_u, maximum_swap_pi_vectors, alpha = 0.2, label = f'{N}')
        ax_no_u.set_title('Generative model priors\n$\pi_\\beta(Z)$')
    if ax_posthoc_posterior is not None:
        scatter_to_simplex(ax_posthoc_posterior, swap_combined_post_hoc_posteriors, alpha = 0.2, label = f'{N}')
        ax_posthoc_posterior.set_title('Posteriors $\pi_\\beta(y, Z)$')
    if ax_no_u_posthoc_posterior is not None:
        scatter_to_simplex(ax_no_u_posthoc_posterior, maximum_swap_post_hoc_posteriors, alpha = 0.2, label = f'{N}')
        ax_no_u_posthoc_posterior.set_title('Posteriors $\pi_\\beta(y, Z)$')
    
