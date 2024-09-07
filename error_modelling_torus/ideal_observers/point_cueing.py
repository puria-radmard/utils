"""
This file is for all cases where the cue is provided as a point value - i.e. the forms in Matthey et al., 2015
"""

from torch.distributions import Normal
from tqdm import tqdm
from torch import Tensor as _T
import torch, math
from purias_utils.error_modelling_torus.ideal_observers.gibbs_utils import gibbs_sample_new_circular_value
from purias_utils.multiitem_working_memory.util.circle_utils_parallel import generate_circular_feature_list

from purias_utils.population.sensory_neurons import MultiFeatureNeuronPopulationSet
from purias_utils.multiitem_working_memory.model_components.input_transferal import PalimpsestStimulusCombiner

from purias_utils.error_modelling_torus.ideal_observers.base import ToroidalFeaturesIdealObserverBase



class PointCueIdealObserver(ToroidalFeaturesIdealObserverBase):

    """
    Similar to non-hierarchical code in Matthey 2015.

    The key differences:
        1. We take beta_i to be the same for all stimuli - no difference between items means we can simplify maths greatly (eq 24-27 now absorbed)
        2. We take beta_i = 1/N i.e. averaging not summing on palimpsest, with noise added once rather than once per stimulus (eq 22, 23: sigmax and sigmay now absorbed)

    By default, target feature is index 1, and there are only D=2 dimensions

    Importantly, we need to make changes to the iterate_stimulus_gibbs_sampling method
        In the base class, we assume that cuing is roled into the encoding process, e.g. with our homegrown decoder forms
        In this case, however, cuing only happens in the decoder, when a point value of the cue is given
        As such, given the simplifed maths, we can do MCMC to sample many valid stimulus arrays, as in the base case
        However, in this case, the cuing feature value must be pinned at the cuing value

        That means that initial_features[:,:,0,0] is the point stim value, and should be the same along the second (num_chains) dimension
    """

    def __init__(
        self,
        sensory_population: MultiFeatureNeuronPopulationSet,
        feature_margins: list[float],
        num_grid_points = 64,
        combiner: PalimpsestStimulusCombiner = None
    ) -> None:

        super().__init__(sensory_population, 1, feature_margins, num_grid_points, combiner)

        assert self.D == 2

    def sensory_cue_representation(self, *args, **kwargs):
        raise TypeError('This should not be called -  see iterate_stimulus_gibbs_sampling docstring for how to present cue point value at decoding time')

    def apply_cue_to_stim_representations(self, *args, **kwargs):
        raise TypeError('This should not be called -  see iterate_stimulus_gibbs_sampling docstring for how to present cue point value at decoding time')

    def add_stim_representation_noise(self, stim_response, sigma_stim, **other_likelihood_kwargs):
        noise = sigma_stim * torch.randn_like(stim_response)
        return stim_response + noise

    def add_cue_representation_noise(self, cue_response, **noise_kwargs):
        raise TypeError('This should not be called')

    def full_cued_response_generation(self, features, likelihood_kwargs):
        stim_response = self.sensory_stim_representation(features)
        noisy_stim_response = self.add_stim_representation_noise(stim_response, **likelihood_kwargs)
        return noisy_stim_response

    def log_p_cued_stim_repr_given_features(self, noisy_cued_representation, sampled_features: _T, **likelihood_kwargs):
        """
        Given simplified maths (uniform beta_i), this is fairly easy to understand

        noisy_cued_representation comes in shape [batch, encoding size]
        sampled_features comes in shape [batch, steps, test points, stim, feature dimensions]

        output of shape [batch, steps, test points, stim]
        """
        
        var = likelihood_kwargs['sigma_stim'] ** 2
        log_scale = math.log(likelihood_kwargs['sigma_stim'])
        batch_size = noisy_cued_representation.shape[0]
        
        all_scaled_total_item_stim_log_probs = []
        # all_stepwise_log_prob_scaler = []

        for b in tqdm(range(batch_size)):
            item_stim_response = self.sensory_stim_representation(sampled_features[[b]])    # [1, steps, testpoints, encoding size]
            
            item_noisy_repr = noisy_cued_representation[[b]].unsqueeze(1).unsqueeze(1)
            item_stim_log_prob = -((item_noisy_repr - item_stim_response) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
            total_item_stim_log_prob = item_stim_log_prob.sum(-1)
            stepwise_log_prob_scaler = total_item_stim_log_prob.min(-1, keepdims = True).values
            scaled_total_item_stim_log_prob = total_item_stim_log_prob - stepwise_log_prob_scaler
            all_scaled_total_item_stim_log_probs.append(scaled_total_item_stim_log_prob)
            # all_stepwise_log_prob_scaler.append(scaled_total_item_stim_log_prob)

        all_scaled_total_item_stim_log_probs = torch.concat(all_scaled_total_item_stim_log_probs, 0)
        # all_stepwise_log_prob_scaler = torch.concat(all_stepwise_log_prob_scaler, 0)
        return all_scaled_total_item_stim_log_probs

    def initialise_stimulus_gibbs_sampling(self, batch_size: int, num_stimuli: int, cuing_features: _T):
        """
        Including batch size allows us to decode multiple example x vectors at once
        Output is of shape [batch_size, num grid points, stim count, D],
            where output[:,:,0,1] is self.test_points repeated                  target_feature_index = 1 by default in this class

        Inherted version you to give it a cue set, of shape [batch size], shared by all grid points of course
            So output[:,:,0,0] is uniform for batch items (the cued value)
        """
        assert list(cuing_features.shape) == [batch_size]
        cuing_features = cuing_features.unsqueeze(-1).numpy()

        all_feature_lists = []
        for d in range(self.D):
            testpoint_feature_list = []
            for tp in self.test_points:
            
                if d == self.target_feature_idx:
                    existing_features = (tp * torch.ones([batch_size, 1])).numpy()
                else:
                    existing_features = cuing_features  # only change from base function

                new_feature_set = generate_circular_feature_list(
                    batch_size = batch_size,
                    num_stim = num_stimuli, 
                    feature_border = self.feature_borders[d],
                    existing_features = existing_features
                )
                testpoint_feature_list.append(torch.tensor(new_feature_set))
            testpoint_feature = torch.stack(testpoint_feature_list, 1)
            all_feature_lists.append(testpoint_feature)
        initial_features = torch.stack(all_feature_lists, -1)

        return initial_features

    def iterate_stimulus_gibbs_sampling_with_target_fixed(self, initial_feature_set: _T, num_iterations=10_000):
        """
        As noted in docstring, the cuing value is also pegged.

        Make sure initial_feature_set contains the cuing information by using this inherited version of initialise_stimulus_gibbs_sampling
        """
        all_sampled_features = [initial_feature_set]
        B, gp, N, D = initial_feature_set.shape
        assert gp == self.num_grid_points
        assert D == self.D

        for t in tqdm(range(num_iterations)):
            
            new_feature_set = all_sampled_features[-1].clone()  # After this, everything is in place to save memory
            
            for n in range(N):

                for d in range(D):

                    if n == 0:
                        continue    # These values are pegged

                    else:
                        new_sample = gibbs_sample_new_circular_value(new_feature_set[...,[d]], n, self.feature_borders[d], inplace = False)
                        new_feature_set[:,:,n,d] = new_sample[:,:,n,0]
            
            all_sampled_features.append(new_feature_set)
        all_sampled_features = torch.stack(all_sampled_features, 1)

        return all_sampled_features



class HierarchicalPointCueIdealObserver(PointCueIdealObserver):
    
    "Again, see Matthey 2015. TODO: Need to construct W - see appendix equations 18,19, TODO: FINISH IMPLEMENTATION - REQUIRES REVISING SENSORY REPRESENTATION, appendix equations 15, 16. as such - the log_p_cued_stim_repr_given_features should also be adjusted"

    def __init__(self, sensory_population: MultiFeatureNeuronPopulationSet, feature_margins: list[float], combiner: PalimpsestStimulusCombiner = None) -> None:
        super().__init__(sensory_population, feature_margins, combiner)

        self.W = ...
