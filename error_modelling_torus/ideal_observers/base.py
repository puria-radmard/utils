from torch import Tensor as _T
from typing import List
import torch
from tqdm import tqdm

from purias_utils.multiitem_working_memory.util.circle_utils_parallel import generate_circular_feature_list
from purias_utils.error_modelling_torus.ideal_observers.gibbs_utils import old_full_gibbs_iteration, old_bin_colour_samples_samples, gibbs_sample_new_circular_value
from purias_utils.population.sensory_neurons import MultiFeatureNeuronPopulationSet
from purias_utils.multiitem_working_memory.model_components.input_transferal import PalimpsestStimulusCombiner

from abc import ABC, abstractmethod


class ToroidalFeaturesIdealObserverBase(ABC):

    """
    19.4.24 update: unless specified, feature_set comes in shaped [batch_size, num grid points, stim count, D]
        where num grid points is the number of test points (\hat\zeta) for the estimated value.
        Note: we have replaced num_chains with grid points! Implementation-wise this doesn't matter!

        As before, wehn multiple MCMC steps are stacked, the shape becomes [batch_size, num steps, num grid points, stim count, D]

    OLD: Unless specified, feature_set comes in as a D-length list with each entry of shape [batch_size, num_chains, stim count, 1]
    """

    def __init__(
        self,
        sensory_population: MultiFeatureNeuronPopulationSet,
        target_feature_idx: int,
        feature_borders: list[float],
        num_grid_points: int = 64,
        combiner: PalimpsestStimulusCombiner = None,
    ) -> None:
        
        self.D = sensory_population.num_features
        self.sensory_population = sensory_population
        self.target_feature_idx = target_feature_idx

        assert len(feature_borders) == self.D
        self.feature_borders = feature_borders
        
        self.combiner = combiner or self.default_combiner

        self.num_grid_points = num_grid_points
        self.test_points = torch.linspace(0.0, 2*torch.pi, num_grid_points + 1)[:-1]
        self.dzeta = (self.test_points[1] - self.test_points[0]).item()
    
    @staticmethod
    def default_combiner(x: _T):
        return x.mean(-2)

    def sensory_stim_representation(self, features: _T):
        "Input [batch, stim, D]. output [batch, encoding size]"
        feature_set = list(features.movedim(-1, 0).unsqueeze(-1))
        seperated_repr = self.sensory_population.forward_from_features(feature_set)
        return self.combiner(seperated_repr)

    def sensory_cue_representation(self, features: _T):
        """
        As always, first item is cued
        """
        for d in range(self.D):
            if d == self.target_feature_idx:
                import pdb; pdb.set_trace(header = 'deal with size here!')
                features[...,d] = torch.nan * features[d][:,0]
            else:
                import pdb; pdb.set_trace(header = 'deal with cued_index here!')
                features[...,d] = features[d][:,0]
        feature_set = list(features.movedim(-1, 0).unsqueeze(-1))
        seperated_repr = self.sensory_population.forward_from_features(feature_set)
        import pdb; pdb.set_trace('Is default combiner correct here?')
        return self.combiner(seperated_repr)

    @abstractmethod
    def apply_cue_to_stim_representations(self, stim_repr: _T, cue_repr: _T, **noise_kwargs):
        raise NotImplementedError('Depends on specific observer!')

    @abstractmethod
    def add_stim_representation_noise(self, stim_response, **noise_kwargs):
        raise NotImplementedError('Depends on specific observer!')

    @abstractmethod
    def add_cue_representation_noise(self, cue_response, **noise_kwargs):
        raise NotImplementedError('Depends on specific observer!')

    def full_cued_response_generation(self, feature_set, cued_index, likelihood_kwargs):
        """
        This is very generic and likely to be overwtitten, e.g. with PointCueIdealObserver
        """
        stim_response = self.sensory_stim_representation(feature_set)
        noisy_stim_response = self.add_stim_representation_noise(stim_response, **likelihood_kwargs)
        cue_response = self.sensory_cue_representation(feature_set, cued_index)
        noisy_cue_response = self.add_cue_representation_noise(cue_response, **likelihood_kwargs)
        cued_stim_response = self.apply_cue_to_stim_representations(noisy_stim_response, noisy_cue_response, **likelihood_kwargs)
        return cued_stim_response

    @abstractmethod
    def log_p_cued_stim_repr_given_features(self, noisy_cued_representation, sampled_features: _T, **likelihood_kwargs):
        """
        log likelihood function on the output of self.full_cued_response_generation given a sampled feature set
        this time expecting each of the D entries of sampled_feature_set to be of shape [batch size, num steps, num test points, num stimuli, feature dimension]

        likelihood_kwargs should include the stim_noise_kwargs, cue_noise_kwargs arguments above!
        """
        raise NotImplementedError('Depends on specific observer!')

    def log_p_features(self, noisy_cued_representation, sampled_features: _T, **prior_kwargs):
        """
        prior function, e.g. for case where the uniform gibbs sampling is actually part of an importance sampling scheme
        this time expecting each of the D entries of sampled_feature_set to be of shape [batch size, num steps, num test points, num stimuli, feature dimension]

        As a default, we assume uniformality, which can be implemented as a zero addition
        """
        return 0.0

    def initialise_stimulus_gibbs_sampling(self, batch_size: int, num_stimuli: int):
        """
        Including batch size allows us to decode multiple example x vectors at once
        Output is of shape [batch_size, num grid points, stim count, D],
            where output[:,:,0,target_feature] is self.test_points repeated
        """
        all_feature_lists = []
        for d in range(self.num_features):
            testpoint_feature_list = []
            for tp in self.test_points:
            
                if d == self.target_feature_idx:
                    existing_features = tp * torch.ones([batch_size, 1]).numpy()
                else:
                    existing_features = None

                new_feature_set = generate_circular_feature_list(
                    batch_size = batch_size,
                    num_stim = num_stimuli, 
                    feature_border = self.feature_borders[d],
                    existing_features = existing_features
                )
                testpoint_feature_list.append(new_feature_set)
            all_feature_lists.append(testpoint_feature_list)
        initial_features = torch.tensor(all_feature_lists)
        import pdb; pdb.set_trace(header = 'make sure all_feature_lists of right size and correct slice is the testpoints')
        return initial_features

    def iterate_stimulus_gibbs_sampling_with_target_fixed(self, initial_feature_set: _T, num_iterations=10_000):
        """
        initial_feature_set is output of initialise_stimulus_gibbs_sampling - check that docstring

        We know that we are pegging the testpoint values, i.e. output[:,:,0,target_feature] to their original values
        """
        all_sampled_features = [initial_feature_set]
        B, gp, N, D = initial_feature_set.shape
        assert gp == self.num_grid_points
        assert D == self.D

        for t in tqdm(range(num_iterations)):
            
            new_feature_set = all_sampled_features[-1].clone()  # After this, everything is in place to save memory
            
            for n in range(N):

                for d in range(D):

                    if n == 0 and d == self.target_feature_idx:
                        continue    # This value is pegged

                    else:
                        new_sample = gibbs_sample_new_circular_value(new_feature_set[...,[d]], n, self.feature_borders[d], inplace = False)
                        new_feature_set[:,:,n,d] = new_sample[:,:,n,0]
                        import pdb; pdb.set_trace(header = 'make sure it changes here!')
            
            all_sampled_features.append(new_feature_set)
        all_sampled_features = torch.stack(all_sampled_features, 1)
        import pdb; pdb.set_trace(header = 'make sure all_sampled_features of right size and correct slice is the testpoints, and that they actually change')
        return all_sampled_features
    
    def generate_scaled_posterior_curve_at_testpoints(
        self,
        sampled_features: _T,
        noisy_cued_representation,
        likelihood_kwargs: dict,
        prior_kwargs: dict,
        add_jitter = True
    ):
        """
        19.4.24 - This is where the power of the power of the 'prebinning' i.e. only sampling at testpoints for the target feature comes out
            Add jitter will add uniform noise to the fixed features, i.e. the testpoints while ensuring they stay in their bins around the unit circle
        
        sampled_features output of iterate_stimulus_gibbs_sampling_with_target_fixed, of shape [batch_size, num steps, num grid points, stim count, D]
        """
        if add_jitter:
            assert (self.D == 2) and (self.target_feature_idx == 1)
            sampled_features[:,:,:,0,1] += self.dzeta * torch.rand_like(sampled_features[:,:,:,0,1]) - (0.5 * self.dzeta)
            
        log_likelihood_grid = self.log_p_cued_stim_repr_given_features(noisy_cued_representation, sampled_features, **likelihood_kwargs)
        log_prior_grid = self.log_p_features(noisy_cued_representation, sampled_features, **prior_kwargs)
        log_posterior_grid = log_likelihood_grid + log_prior_grid   # [batch, steps, testpoints]
        scaled_posteriors = log_posterior_grid.exp().mean(1) # Finally, montecarlo integration -> [batch, testpoints]

        return scaled_posteriors


    def old_generate_full_binned_scaled_posterior_curve(
        self, 
        sampled_feature_sets: List[_T],
        noisy_cued_representation, 
        num_bins: int, 
        likelihood_kwargs: dict,
        prior_kwargs: dict,
        cued_index: int = 0
    ):
        """
        19.4.24 update:
            This method has been labelled old because it is relatively inefficient
            We know which item is getting binned later (i.e. sampled_feature_sets[self.target_feature_idx][...,cued_index,:])
            Therefore, we can just iterate over a grid of values for this individual feature, and do gibbs over the rest of the features.
            This ensures we have roughly the same number of samples for each candidate value too, so similar variance for the posterior estimate.

        Full decoder pipeline, for generating an (unnormalised) posterior function over the binned colour value,
            which can be maximised over
        
        sampled_feature_sets is outputs of iterate_stimulus_gibbs_sampling,
            i.e. D-length list of tensors shaped [batch_size, num_iterations, num_chains, stim count, 1]

        noisy_cued_representation is the output of self.full_cued_response_generation XXX: SHAPE
        """

        assert len(sampled_feature_sets) == self.D

        # masks of shape [num_bins, batch_size, total samples (num steps * num chains)]
        all_target_features = sampled_feature_sets[self.target_feature_idx]
        bin_boundaries, masks = old_bin_colour_samples_samples(all_target_features, num_bins, cued_index)  # masks of shape [batch, steps, chains, 1]
        bins, batch, steps, chains = masks.shape
        masks = masks.reshape(bins, batch, steps * chains)

        import pdb; pdb.set_trace(header = "XXX shapes!")
        log_likelihood_grid = self.log_p_cued_stim_repr_given_features(
            noisy_cued_representation, sampled_feature_sets, cued_index, **likelihood_kwargs
        )
        log_prior_grid = self.log_p_features(
            noisy_cued_representation, sampled_feature_sets, cued_index, **prior_kwargs
        )
        log_posterior = log_likelihood_grid + log_prior_grid

        underflow_helpers = log_posterior.max(-1).values.unsqueeze(-1)
        scaled_posterior_grid = (log_posterior - underflow_helpers).exp().unsqueeze(0)

        masked_scaled_posterior_grids = (scaled_posterior_grid * masks).reshape(num_bins, batch, -1)    # inter -> [num bins, batch_size, num steps * num chains]
        scaled_posteriors = masked_scaled_posterior_grids.sum(-1) / masks.sum(-1)
        scaled_posteriors = torch.nan_to_num(scaled_posteriors, 0.0)

        return bin_boundaries, scaled_posteriors, underflow_helpers

    def old_iterate_stimulus_gibbs_sampling(self, initial_feature_set: List[_T], num_iterations=10_000):
        """
        19.4.24 update: see old_generate_full_binned_scaled_posterior_curve docstring

        Run num_chains MCMC simulations for num_iterations steps each
        initial_feature_set comes in as described in class docstring
        output comes out as D-length list with each entry of shape [batch_size, num_iterations, num_chains, stim count, 1]
        """

        all_sampled_features = [initial_feature_set]

        for t in tqdm(range(num_iterations)):
            new_feature_set = full_gibbs_iteration(all_sampled_features[-1], self.D, self.feature_borders)
            all_sampled_features.append(new_feature_set)
            import pdb; pdb.set_trace(header = 'ensure different!')

        return [torch.stack(asf, 1) for asf in all_sampled_features]

    def old_generate_estimates_from_scaled_posterior_grid(self, scaled_posteriors: _T, bin_boundaries: _T):
        "19.4.24 update: see old_generate_full_binned_scaled_posterior_curve docstring"

        mid_points = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
        estimate_idxs = scaled_posteriors.argmax(0)
        estimates = mid_points[estimate_idxs]
        return estimates
