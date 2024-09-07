"""
This file is for all cases where the cue is also encoded by a tuning curve set
"""

from torch.distributions import Normal
from torch import Tensor as _T
import torch

from purias_utils.population.sensory_neurons import MultiFeatureNeuronPopulationSet
from purias_utils.multiitem_working_memory.model_components.input_transferal import DDCStyleStimulusCombiner

from purias_utils.error_modelling_torus.ideal_observers.base import ToroidalFeaturesIdealObserverBase



class FullCueRepresentationIdealObserver(ToroidalFeaturesIdealObserverBase):

    """
    Full cue representation, where the cue and stimulus response are not combined, but are kept as two different objects,
        with their likelihood considered seperately
    Additive independent noise everywhere
    """

    def apply_cue_to_stim_representations(self, stim_repr: _T, cue_repr: _T, **other_likelihood_kwargs):
        """
        No combination
        In the initial case, we have them be the same neural size, however this gets across a lot of information
            to the cue representation, because you have many independently noisy versions of the same cue tuning.
        So maybe in the future we cut down (~sqrt) the cue neural size
        As such, we just present them as two seperate entities
        """
        return [stim_repr, cue_repr]

    def add_stim_representation_noise(self, stim_response, sigma_stim, **other_likelihood_kwargs):
        noise = sigma_stim * torch.randn_like(stim_response)
        return stim_response + noise

    def add_cue_representation_noise(self, cue_response, sigma_cue, **other_likelihood_kwargs):
        noise = sigma_cue * torch.randn_like(cue_response)
        return cue_response + noise

    def log_p_cued_stim_repr_given_features(self, noisy_cued_representation, sampled_features: _T, cued_index, **likelihood_kwargs):
        """
        noisy_cued_representation is a list of size 2 - first item is a stimulus representation, second is a cue representation

        Log likelihoods are easy to interpret, just the reversal of add_stim_representation_noise and add_cue_representation_noise
        """
        
        # Gaussian means
        stim_response = self.sensory_stim_representation(sampled_features)
        cue_response = self.sensory_cue_representation(sampled_features, cued_index)
        
        # Gaussian objects
        stim_dist = Normal(loc = stim_response, scale = likelihood_kwargs['sigma_stim'])
        cue_dist = Normal(loc = cue_response, scale = likelihood_kwargs['sigma_cue'])

        # Log likelihood of whole thing
        stim_log_prob = stim_dist.log_prob(noisy_cued_representation[0])
        cue_log_prob = cue_dist.log_prob(noisy_cued_representation[0])

        import pdb; pdb.set_trace(header = 'check shapes here!')

        return stim_log_prob + cue_log_prob



class HadamardCueIdealObserver(ToroidalFeaturesIdealObserverBase):
    """
    Hadamard decoder, where the cued stim representation is element-wise multiplied by the combined sensory
    
    Additive independent noise everywhere again
    """

    def apply_cue_to_stim_representations(self, stim_repr: _T, cue_repr: _T, sigma_comb, **other_likelihood_kwargs):
        noise = sigma_comb * torch.randn_like(stim_repr)
        return (stim_repr * cue_repr) + noise

    def add_stim_representation_noise(self, stim_response, sigma_stim, **other_likelihood_kwargs):
        noise = sigma_stim * torch.randn_like(stim_response)
        return stim_response + noise

    def add_cue_representation_noise(self, cue_response, sigma_cue, **other_likelihood_kwargs):
        noise = sigma_cue * torch.randn_like(cue_response)
        return cue_response + noise

    def log_p_cued_stim_repr_given_features(self, noisy_cued_representation, sampled_features: _T, **likelihood_kwargs):
        """
        noisy_cued_representation is this time a single representation

        loglikehood now scales with likelihood_kwargs and the unnoised cue response... see maths
        """
        
        # Used to generate moments
        stim_response = self.sensory_stim_representation(sampled_features)
        cue_response = self.sensory_cue_representation(sampled_features)

        # Gaussian mean
        mean = stim_response * cue_response

        # Gaussian variance - diagonal so we only have to take those terms
        import pdb; pdb.set_trace(header = 'Check shapes are valid here')
        pure_term_multiplier = likelihood_kwargs['sigma_comb']**2 + (likelihood_kwargs['sigma_stim']**2 * likelihood_kwargs['sigma_cue']**2)
        cross_term = (
            (likelihood_kwargs['cue_stim']**2 * (stim_response * stim_response)) + 
            (likelihood_kwargs['sigma_stim']**2 * (cue_response * cue_response))
        ) * torch.eye()

        dist = Normal(loc = mean, scale = pure_term_multiplier + cross_term)
        log_prob = dist.log_prob(noisy_cued_representation)
        return log_prob



class ConcatenationCueIdealObserver(ToroidalFeaturesIdealObserverBase):
    """
    Concat decoder, where the cued stim representation a projection of the concatenation of the stim and cue representations

    Generates two random nxn matrices and an n-sized column vector, where n is the SHARED size of the stim and cue represntations
        This might want to be changed in the future...
    
    
    Additive independent noise everywhere again
    """

    def __init__(self, sensory_population: MultiFeatureNeuronPopulationSet, target_feature_idx: int, feature_margins: list[float], combiner: DDCStyleStimulusCombiner = None) -> None:
        super().__init__(sensory_population, target_feature_idx, feature_margins, combiner)
        n = sensory_population.population_size
        self.W_stim = torch.randn(n, n) / n
        self.W_cue = torch.randn(n, n) / n
        self.b = torch.randn(n) / (n ** 0.5)

    def apply_cue_to_stim_representations(self, stim_repr: _T, cue_repr: _T, sigma_comb, **other_likelihood_kwargs):
        import pdb; pdb.set_trace(header = 'shapes here!')
        comb_repr = (self.W_stim @ stim_repr) + (self.W_cue @ cue_repr) + self.b
        noise = sigma_comb * torch.randn_like(stim_repr)
        return comb_repr + noise

    def add_stim_representation_noise(self, stim_response, sigma_stim, **other_likelihood_kwargs):
        noise = sigma_stim * torch.randn_like(stim_response)
        return stim_response + noise

    def add_cue_representation_noise(self, cue_response, sigma_cue, **other_likelihood_kwargs):
        noise = sigma_cue * torch.randn_like(cue_response)
        return cue_response + noise

    def log_p_cued_stim_repr_given_features(self, noisy_cued_representation, sampled_features: _T, **likelihood_kwargs):
        """
        noisy_cued_representation is this time a single representation

        loglikehood now scales with likelihood_kwargs and the unnoised cue response... see maths
        """
        
        # Used to generate moments
        stim_response = self.sensory_stim_representation(sampled_features)
        cue_response = self.sensory_cue_representation(sampled_features)

        # Gaussian moments
        mean = self.apply_cue_to_stim_representations(stim_response, cue_response, sigma_comb=0.0)
        total_variance = (
            (likelihood_kwargs['sigma_stim'] ** 2) + 
            (likelihood_kwargs['sigma_cue'] ** 2) + 
            (likelihood_kwargs['sigma_comb'] ** 2)
        )
        var = torch.ones_like(mean) * (total_variance ** 0.5)
        import pdb; pdb.set_trace(header = 'check shapes here!')
        
        dist = Normal(loc = mean, scale = var)
        log_prob = dist.log_prob(noisy_cued_representation)
        return log_prob
