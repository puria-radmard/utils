raise Exception

import torch

from torch import Tensor as _T

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel
from purias_utils.error_modelling_torus.non_parametric_error_model.util import inference_mean_only

from purias_utils.multiitem_working_memory.wm_tasks.multiple_orientation.base import MultipleOrientationDelayedSingleEstimationTask

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from purias_utils.util.api import return_as_obj



class NonParametricModelDrivenMultipleOrientationDelayedSingleEstimationTask(MultipleOrientationDelayedSingleEstimationTask):
    """
    Same as parent task, always cuing the first item, but also providing variables from the GP model
    These should be used by the loss function downstream
    """

    def __init__(self, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, kernel_noise_sigma: float, epoch_kwargs: dict, board_kwargs: dict, batch_size: int, cue_during_response: bool = True, visualiser = None) -> None:
        super().__init__(epoch_kwargs, board_kwargs, batch_size, cue_during_response, visualiser)
        self.generative_model = generative_model.eval()
        self.variational_model = variational_model.eval()
        self.kernel_noise_sigma = kernel_noise_sigma

    def select_components(self, start_of_resp_saccade_locations: _T, all_features: _T, target_feature_idx):
        """
        At the start of the response period, whichever component of the N+1 has the highest likelihood is selected to be the target.
        
        This is done by taking the angle of the end of cue period saccade and treating that as an angular estimate.
        This is fed into the generative model, which gives the loglikelihood of each component (including the uniform)

        start_of_resp_saccade_locations of shape [batch, trial, 2]

        Unfortunately saccade location = [sin(zeta), cos(zeta)] rather than the other way around due to the PyTorch API...
        """
        component_target_angles = all_features[:,:,target_feature_idx].to(start_of_resp_saccade_locations.device)               # [batch, N]
        start_of_resp_saccade_angles = torch.arctan2(*start_of_resp_saccade_locations.movedim(-1, 0))                           # [batch, trial]
        estimate_deviations = rectify_angles(start_of_resp_saccade_angles.unsqueeze(2) - component_target_angles.unsqueeze(1))  # [batch, trial, N, 2]
        individual_component_log_likelihoods = self.generative_model.individual_component_log_likelihoods_from_estimate_deviations(estimate_deviations).squeeze(0)  # [batch, trial, N+1]
        selected_components = individual_component_log_likelihoods.argmax(-1)   # [batch, trial]
        return selected_components

    def selected_response_time_targets_and_weightings(self, all_features, selected_components, target_feature_idx):
        """
        After a component is selected, the weighting on the loss function is the pi vector
        M should come in as {0,...,N}^batch where 0 means the uniform mode.
        
        In the case of a uniform mode, there should be no weighting, i.e. no supervision at all - the network is guessing, so it's free to roam!

        selected_components of shape [batch, trial], as is final weighting...
        all_features is [batch, N, D], of which the first is always cued!
        """
        deltas = rectify_angles(all_features - all_features[:,[0],:]).to('cuda')

        mu = inference_mean_only(
            generative_model=self.generative_model, variational_model=self.variational_model, 
            deltas=deltas, kernel_noise_sigma = self.kernel_noise_sigma, device = 'cuda'
        )

        f_mean = self.variational_model.reparameterised_sample(
            num_samples = 1, mu = mu, 
            sigma_chol = torch.zeros(mu.shape[0], mu.shape[0], dtype=mu.dtype, device=mu.device),         # i.e. just the mean!
            M = self.batch_size, N = all_features.shape[1]
        )                                                           # [batch, N]

        pi_vectors = self.generative_model.generate_pi_vectors(model_evaulations = f_mean)[0].cpu()    # [batch, N+1], with [:,0] being uniform probs
        pi_vectors[:, 0] = 0.0                                      # See docstring - no supervision in this case...

        weightings = torch.zeros_like(selected_components).float()          # [batch, trial]
        target_angles = torch.zeros_like(selected_components).float()       # [batch, trial]

        for m in range(selected_components.shape[0]):
            per_trial_selected_components = selected_components[m]          # [trial]
            weightings[m] = (pi_vectors[m, per_trial_selected_components])    # Selecting from the same pi vector many times for each trial
            target_angles[m] = all_features[m, per_trial_selected_components-1, target_feature_idx]  # 0-1=-1 doesn't matter - weighting will take care of that
        
        target_locations = torch.stack([target_angles.sin(), target_angles.cos()], -1)

        return target_angles, target_locations, weightings

    @return_as_obj
    def generate_response_targets_and_weightings(self, original_boards, start_of_resp_saccade_locations, target_feature_idx):
        "See individual functions' docstrings!"
        all_features = torch.stack([board.full_feature_batch() for board in original_boards], 0)     # [M (batch), N (set_size), D (2)]
        selected_components = self.select_components(start_of_resp_saccade_locations, all_features, target_feature_idx)
        target_angles, target_locations, target_weightings = self.selected_response_time_targets_and_weightings(
            all_features, selected_components, target_feature_idx
        )
        cued_targets = all_features[:,0,target_feature_idx]
        return {
            "selected_components": selected_components,
            "target_locations": target_locations,
            "target_angles": target_angles,
            "target_weightings": target_weightings,
            "cued_targets": cued_targets
        }

    def generate_targets(self, boards, epoch_name: str, target_feature_idx: int):
        "Definitely avoid using original targets in the response period!"
        if epoch_name == 'resp':
            angles = torch.tensor([[torch.nan, torch.nan] for _ in boards])
            targets = torch.stack([angles.sin(), angles.cos()], -1)
        else:
            angles =  torch.tensor([torch.nan for _ in boards])
            targets =  torch.tensor([[0., 0.] for _ in boards])
        return {'angles': angles, 'saccade_targets': targets}
