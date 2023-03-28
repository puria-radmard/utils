from typing import List
from purias_utils.population.stimuli import StimulusBasisFunctionSetBase

import math

import torch
from torch import Tensor as T

class TuningCurveBase:

    def __init__(self, stimulus_dimension: int) -> None:
        self.stimulus_dimension = stimulus_dimension

    def _r(self, stimulus: T):
        raise NotImplementedError

    def mean_firing_rate(self, stimulus: T):
        assert stimulus.shape[1] == self.stimulus_dimension
        assert len(stimulus.shape) == 2
        response = self._r(stimulus)
        return response


class AngularThresholdedTuningCurve(TuningCurveBase):

    """
    Can stack tuning curves here remember!
    """

    def __init__(self, tuning_curve: T, thresholds: T) -> None:
        stimulus_dimension = tuning_curve.shape[-1]
        super().__init__(stimulus_dimension=stimulus_dimension)
        self.tuning_curve = tuning_curve
        self.thresholds = thresholds

    def _r(self, stimulus: T):
        all_subpopulation_responses = []
        for threshold in self.thresholds:
            all_subpopulation_responses.append(
                torch.relu(stimulus @ self.tuning_curve.T - threshold)
            )
        stacked = torch.stack(all_subpopulation_responses, dim=2)
        interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)
        return interleaved


class IndependentPoissonPopulationResponse:
    """
        Assuming Poisson process with tuning curve mean, e.g. in rate_loglikelihoods
    """

    def __init__(self, tuning_curves: List[TuningCurveBase], stimulus_dimension: int = None) -> None:
        self.stimulus_dimension = stimulus_dimension if stimulus_dimension is not None else tuning_curves[0].stimulus_dimension
        assert all([tc.stimulus_dimension == self.stimulus_dimension for tc in tuning_curves])
        self.tuning_curves = tuning_curves

    def population_mean_firing_rate(self, stimulus: T):
        """Finer combinations should be done at tuning curve lever"""
        return torch.vstack([tc.mean_firing_rate(stimulus) for tc in self.tuning_curves])

    def population_empirical_rates(self, stimulus: T, duration: float):
        mean_firing_rates = self.population_mean_firing_rate(stimulus)
        return torch.poisson(mean_firing_rates * duration) / duration

    def empirical_rates_from_encoded_rates(self, rates: T, duration: float):
        return torch.poisson(rates * duration) / duration

    def uncertain_response(self, stimulus_function_base: StimulusBasisFunctionSetBase, probabilities: T):
        assert probabilities.sum() == 1.0
        return probabilities @ self.population_mean_firing_rate(stimulus_function_base.full_basis().T)

    def rate_loglikelihoods(self, rates: torch.Tensor, tuning_curve_rates: torch.Tensor, duration: float):
        """
            TODO: ADD TO NEURONS.PY
            This is specific the independent Poisson form of equation 4 in Ma, et al. 2006
            rates is the output of empirical_rates_from_encoded_rates, for one stimulus
            tuning_curve_rates is the output of population_empirical_rates, for many stimuli
        """
        counts = rates * duration
        #assert (counts % 1).sum() == 0., "Require natural number events"
        assert rates.shape[-1] == tuning_curve_rates.shape[-1]
        assert rates.shape[0] == 1, "No batching at the moment"
        counts = counts.round().int()

        return (
            ((tuning_curve_rates * duration) ** counts).log() + 
            (- tuning_curve_rates * duration) -
            torch.lgamma(counts+1)
        )
