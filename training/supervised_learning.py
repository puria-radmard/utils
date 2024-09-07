from typing import Optional

import torch
from torch import Tensor as _T
from torch.nn import Module, CrossEntropyLoss

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.util import inference_mean_only

from purias_utils.multiitem_working_memory.stimulus_design.stimulus_board import MultiOrientationStimulusBoard

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


class WeightedSequenceCEL(CrossEntropyLoss):

    def __init__(self, weight: Optional[_T] = None) -> None:
        super().__init__(weight, reduction='none')

    def forward(self, input: _T, target: _T, seq_weight: _T) -> _T:
        input = input.permute(0, 2, 1)
        # [batch, time]
        unweighted = super(WeightedSequenceCEL, self).forward(input=input, target=target)
        return (seq_weight * unweighted).mean()


class AngleLoss(Module):
    "Given an R^2 fixation and a (scalar) angle, gets the weighted cosine difference + magnitude difference to 1"

    def __init__(self, magnitude_regulariser_weight: float = 0.01) -> None:
        super().__init__()
        self.magnitude_regulariser_weight = magnitude_regulariser_weight

    def forward(self, input: _T, target: _T) -> tuple[_T]:
        assert input.shape == (target.shape[0], 2)
        assert target.shape == (input.shape[0], 1)
        
        angles = torch.arctan2(*input.T)
        angle_target = target[:,0]
        dec_loss = 1 - (angles - angle_target).cos().mean()

        if self.magnitude_regulariser_weight > 0:
            mags = (input[:,0]**2 +  input[:,1]**2)**0.5
            mag_loss = self.magnitude_regulariser_weight * torch.mean((mags - 1.0)**2)
        else:
            mag_loss = torch.tensor(0.0)

        return {'dec_loss': dec_loss, 'mag_loss': mag_loss, 'angles': angles}


class FixationPowerLoss(Module):

    def __init__(self, p=2) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: _T, target: _T, agg = True) -> tuple[_T]:
        assert input.shape == (target.shape[0], 2)
        assert target.shape == (input.shape[0], 2)
        loss_grid = (target - input).float_power(self.p).sum(-1)
        if agg:
            loss_grid = loss_grid.mean()
        return {'dec_loss': loss_grid}


class NonParametricModelDrivenMultiModalComponentLoss(Module):

    def __init__(self, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, kernel_noise_sigma: float, device, inverse_temperature: float = 1.0) -> None:
        super().__init__()
        self.generative_model = generative_model.eval().to(device)
        self.variational_model = variational_model.eval().to(device)
        self.kernel_noise_sigma = kernel_noise_sigma
        self.to(device)
        self.inverse_temperature = inverse_temperature
        assert self.inverse_temperature == 1.0

    def generate_pi_vectors(self, all_features: _T, inverse_temperature, device):
        "all_features of shape [M (batch), N (set_size), D (2)]"
        with torch.no_grad():
            deltas = rectify_angles(all_features - all_features[:,[0],:])
            mu = inference_mean_only(
                generative_model=self.generative_model, variational_model=self.variational_model, 
                deltas=deltas, kernel_noise_sigma = self.kernel_noise_sigma, device = device
            )

            f_mean = self.variational_model.reparameterised_sample(
                num_samples = 1, mu = mu, 
                sigma_chol = torch.zeros(mu.shape[0], mu.shape[0], dtype=mu.dtype, device=mu.device),         # i.e. just the mean!
                M = all_features.shape[0], N = all_features.shape[1]
            )                                                           # [batch, N]

            # pi_vectors = self.generative_model.generate_pi_vectors(model_evaulations = f_mean)[0]    # [batch, N+1], with [:,0] being uniform probs
            _, exp_grid = self.generative_model.generate_pi_vectors(model_evaulations = f_mean, return_exp_grid = True)    # [batch, N+1], with [:,0] being uniform probs
            exp_grid *= inverse_temperature
            denominator = exp_grid.sum(-1, keepdim=True)                                        # [I, M, 1]
            pi_vectors = exp_grid / denominator
        return pi_vectors

    def generate_componentwise_likelihoods(self, estimates, all_features, target_feature_idx):
        "estimates of shape [batch, trial], all_features of shape [batch, stim, dim (2)]"
        component_target_angles = all_features[:,:,target_feature_idx]                                          # [batch, N]
        estimate_deviations = rectify_angles(estimates.unsqueeze(2) - component_target_angles.unsqueeze(1))     # [batch, trial, N, 2]
        individual_component_log_likelihoods = self.generative_model.individual_component_log_likelihoods_from_estimate_deviations(estimate_deviations).squeeze(0)
        return individual_component_log_likelihoods.exp()   # [batch, trial, N+1]
    
    @staticmethod
    def infer_component_responsibilities(lhs, pis):
        scaled_lhs = lhs * pis
        norm_term = (scaled_lhs).sum(-1, keepdim=True)
        component_posteriror = (scaled_lhs) / norm_term
        return component_posteriror

    @staticmethod
    def kl(p1, p2):
        log_grid = (p1 * (p1 / p2).log())
        log_grid[p1 == 0.] = 0.0
        return log_grid.sum(-1).mean(0)

    def forward(self, estimates: _T, boards: list[MultiOrientationStimulusBoard], target_feature_idx, inverse_temperature = None, device = 'cuda'):
        """
        estimates of shape [batch, trial]
        boards of length batch, and all of the same set size

        Order of events:
            0. Extract all features from the biards, and get the ones that are being estimated
            1. Generate pi_vectors of shape [batch, N+1], with [:,0] being uniform probs
            2. Get component-wise likelihoods for each estimate, and get errors
            3. Invert with Bayes rule to get posterior of estimates belonging to each component
            4. MC marginalise posterior to get MC marginals over components (pi^p)
            5. Do reverse KL on the two vectors (KL[pi^p || pi])

        TODO: Also return random_kl, which is the loss for this batch if the XXX: if what? what is the lower bound case here?

        Also return an example distribution from the first item in the batch, evaluated on a grid of points
        """
        if estimates.shape[1] < 64:
            print("Too few trials for NonParametricModelDrivenMultiModalComponentLoss - require at least 64")

        if inverse_temperature == None:
            inverse_temperature = self.inverse_temperature

        # Main loss calculation
        all_features = torch.stack([board.full_feature_batch() for board in boards], 0).to(estimates.device)     # [M (batch), N (set_size), D (2)]
        pi_vectors = self.generate_pi_vectors(all_features, inverse_temperature, device).squeeze(0).unsqueeze(1)                                                    # [batch, 1, N+1]
        componentwise_likelihoods_of_estimates = self.generate_componentwise_likelihoods(estimates, all_features, target_feature_idx)     # [batch, trial, N+1]
        component_posteriors = self.infer_component_responsibilities(componentwise_likelihoods_of_estimates, pi_vectors)    # [batch, trial, N+1]
        estimate_pis = component_posteriors.mean(1)     # [batch, N+1]
        kl = self.kl(estimate_pis, pi_vectors.squeeze(1))

        # Loss upper bound calculation
        #import pdb; pdb.set_trace()
        #random_kl = self.kl(estimate_pis, pi_vectors.squeeze(1))

        # Example distribution
        example_distribution_x = torch.linspace(-torch.pi, torch.pi, 51)[:-1].unsqueeze(0).to(estimates.device)
        with torch.no_grad():
            example_componentwise_likelihoods = self.generate_componentwise_likelihoods(example_distribution_x, all_features[[0]], target_feature_idx)     # [1, 50, N+1]
        example_pi_vectors = pi_vectors[[0]].detach()
        example_distribution_y = (example_pi_vectors * example_componentwise_likelihoods).sum(-1).squeeze(0)

        return {
            'dec_loss': kl,
            # 'random_selection_kl': random_kl,
            'example_distribution_x': example_distribution_x.squeeze(0),
            'example_distribution_y': example_distribution_y,
            'pis': estimate_pis,
            'mus': all_features[:,:,target_feature_idx],
            'kps': torch.ones_like(all_features[:,:,target_feature_idx]) * self.generative_model.concentration.data
        }
