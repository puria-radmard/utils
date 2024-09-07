from torch.distributions import Normal
from torch import Tensor as _T
from typing import List
import torch

from purias_utils.population.sensory_neurons import MultiFeatureNeuronPopulationSet
from purias_utils.error_modelling_torus.ideal_observers.base import ToroidalFeaturesIdealObserverBase
from purias_utils.multiitem_working_memory.model_components.input_transferal import DDCStyleStimulusCombiner


class SingleDimensionIdealObserver(ToroidalFeaturesIdealObserverBase):
    """
    Replicating the single dimension tuning curve of Wei and Woodford 2023, sort of...

    Need to have a single dimensional sensory population set! 

    In Wei and Woodford 2023, we have that "the manifold has a centroid, and its distance to m(theta) is 1 for every stimulus."
        XXX: Because the tuning curves are constrained to be von Mises shaped, constant distance from centroid should hold anyway?

    In this model, population code is shared between all conditions (e.g. stimulus coherence, set size), but sigma_noise (the only noise parameter)
        is private to each condition. This should be fed in at the right place in full_cued_response_generation
    """

    def __init__(
        self,
        sensory_population: MultiFeatureNeuronPopulationSet,
        target_feature_idx: int,
        feature_margins: list[float],
        combiner: DDCStyleStimulusCombiner = None
    ) -> None:

        super().__init__(sensory_population, target_feature_idx, feature_margins, combiner)

        assert self.D == 1

    def apply_cue_to_stim_representations(self, *args, **kwargs):
        raise TypeError('This should not be called')

    def add_cue_representation_noise(self, *args, **kwargs):
        raise TypeError('This should not be called')

    def sensory_cue_representation(self, feature_set: List[_T], cued_index = 0):
        raise TypeError('This should not be called')

    def add_stim_representation_noise(self, stim_response, sigma_stim, **other_likelihood_kwargs):
        noise = sigma_stim * torch.randn_like(stim_response)
        return stim_response + noise

    def full_cued_response_generation(self, feature_set, cued_index, likelihood_kwargs):
        stim_response = self.sensory_stim_representation(feature_set)
        noisy_stim_response = self.add_stim_representation_noise(stim_response, **likelihood_kwargs)
        return noisy_stim_response

    def log_p_cued_stim_repr_given_features(self, noisy_cued_representation, sampled_feature_set: List[_T], cued_index, **likelihood_kwargs):
        """
        Easiest one to find...
        """
        assert len(sampled_feature_set) == 1, "Cannot have more than one feature dimension for SingleDimensionIdealObserver"
        import pdb; pdb.set_trace(header = "Make sure there's only one stimulus sampled here - recall that presence of distractors is all rolled into the noise term!")

        stim_response = self.sensory_stim_representation(sampled_feature_set)
        stim_dist = Normal(loc = stim_response, scale = likelihood_kwargs['sigma_stim'])
        stim_log_prob = stim_dist.log_prob(noisy_cued_representation)

        return stim_log_prob
    
