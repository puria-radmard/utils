from __future__ import annotations

import torch
import numpy as np
from torch import Tensor as _T

from purias_utils.util.error_modelling import kl, I0
from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.non_parametric_error_model.util import inference_mean_only

from purias_utils.multiitem_working_memory.util.circle_utils import generate_circular_feature_list

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel
from purias_utils.error_modelling_torus.non_parametric_error_model.variational_approx import NonParametricSwapErrorsVariationalModel



def generate_gp_distribution(features: _T, generative_model: NonParametricSwapErrorsGenerativeModel, variational_model: NonParametricSwapErrorsVariationalModel, kernel_noise_sigma: float, target_feature_idx: int):
    """
    Pass display into GP model and get a set of mixture distribution parameters
    
    features of shape [batch, stimuli, dimensionality] as always
    """
    kp = generative_model.concentration(features.shape[1]).detach()  # No need to change until model itself is changed

    mus = features[:,:,target_feature_idx]
    kps = torch.ones_like(mus).float().to(kp.device) * kp

    ## XXX Lifted from training.supervised_learning.NonParametricModelDrivenMultiModalComponentLoss.generate_pi_vectors
    deltas = rectify_angles(features - features[:,[0],:])
    with torch.no_grad():
        feature_mu = inference_mean_only(
            generative_model=generative_model, variational_model=variational_model, 
            deltas=deltas, kernel_noise_sigma = kernel_noise_sigma, device = 'cuda'
        )

        f_mean = variational_model.reparameterised_sample(
            num_samples = 1, mu = feature_mu, 
            sigma_chol = torch.zeros(feature_mu.shape[0], feature_mu.shape[0], dtype=feature_mu.dtype, device=feature_mu.device),         # i.e. just the mean!
            M = features.shape[0], N = features.shape[1]
        )                                                           # [batch, N]

        pis = generative_model.generate_pi_vectors(model_evaulations = f_mean, return_exp_grid = False)[0]    # [batch, N+1], with [:,0] being uniform probs

    return {
        'pis': pis, # [batch, N+1] of which first mode is the uniform, and second is cued
        'kps': kps, # [batch, N] of which first mode is cued
        'mus': rectify_angles(mus), # [batch, N] of which first mode is cued
    }


def generate_calibration_curves(samples: _T, batch_data: dict, num_samples: int, behavioural_model: ExplicitMixtureModel, num_pdf_eval_points = 1024, include_shuffled = True):
    "Expecting samples of shape [batch size, num samples]"

    u = behavioural_model.cdf_transform(samples[:,:num_samples], **batch_data, num_pdf_eval_points=num_pdf_eval_points)
    u = np.sort(u.flatten().cpu().numpy())  # [batch * trial]
    calibration_x = [0.0] + u.tolist() + [1.0]                                  # [batch * trial + 2]
    calibration_y = [0.0] + np.linspace(0.0, 1.0, len(u)).tolist() + [1.0]      # [batch * trial + 2], strict inquality
    
    if include_shuffled:
        shuff_calibration_x, shuff_calibration_y, _, _ = generate_calibration_curves(
            samples[torch.randperm(len(samples))], batch_data, num_samples, behavioural_model, num_pdf_eval_points, include_shuffled = False
        )
    else:
        shuff_calibration_x, shuff_calibration_y = None, None
        
    return np.array(calibration_x), np.array(calibration_y), shuff_calibration_x, shuff_calibration_y






class ExplicitMixtureModel:

    """
    Generates truncated multimodal distribution parameters in a range and do inference on estimates
    NB: assumes truncation range is sufficiently 
    """

    def __init__(self, N: int, conc_lower: float, conc_upper: float, mu_half_range: float, data_half_range: float, uniform_scale=1.0, component_scale=1.0) -> None:
        self.N = N
        self.conc_lower = conc_lower
        self.conc_upper = conc_upper
        self.mu_half_range = mu_half_range
        self.data_half_range = data_half_range
        self.uniform_scale = uniform_scale
        self.component_scale = component_scale

    def generate_batch(self, batch_size, **kwargs):
        N = self.N
        pi_tilde = torch.rand(batch_size, N+1)
        pi_tilde[:,0] *= self.uniform_scale
        pi_tilde[:,1:] *= self.component_scale
        return dict(
            mus = 2 * self.mu_half_range * torch.rand(batch_size, N) - self.mu_half_range,
            kps = self.conc_lower + (self.conc_upper - self.conc_lower) * torch.rand(batch_size, N),
            pis = pi_tilde.softmax(-1)
        )

    def component_likelihood(self, x: _T, mus: _T, kps: _T):
        "x of shape [batch, something, 1]"
        exponent_term = - 0.5 * (x - mus).square() * kps
        normalisation_terms = (kps / (2 * torch.pi)).sqrt()
        normal_likelihoods = normalisation_terms * exponent_term.exp()   # [batch, trials, N]
        return normal_likelihoods

    def attach_uniform_likelihood_and_scale(self, component_likelihoods, pis):
        batch, trials, _ = component_likelihoods.shape
        uniform_likelihoods = torch.ones(batch, trials, 1).to(component_likelihoods.device) / (2 * self.data_half_range)
        likelihoods = torch.concat([uniform_likelihoods, component_likelihoods], dim=-1)  # [batch, trials, N+1]
        return likelihoods * pis

    def bayes_rule(self, estimates: _T, mus: _T, kps: _T, pis: _T, **kwargs) -> _T:
        "See do_marginal_inference"
        estimates = estimates.unsqueeze(-1)
        mus = mus.unsqueeze(1)
        kps = kps.unsqueeze(1)
        pis = pis.unsqueeze(1)
        
        # scaled_likelihoods = Q(beta) * q(y | beta) of shape [batch, N+1]
        normal_likelihoods = self.component_likelihood(estimates, mus, kps)
        scaled_likelihoods = self.attach_uniform_likelihood_and_scale(normal_likelihoods, pis)

        # Bayes' rule
        norm_term = (scaled_likelihoods).sum(-1, keepdim=True)   # [batch, trials, 1]
        component_posteriror = (scaled_likelihoods) / norm_term          # [batch, trials, N+1] aka Q(beta | y)
        return component_posteriror
    
    def do_marginal_inference(self, estimates: _T, mus: _T, kps: _T, pis: _T, **kwargs) -> _T:
        """
        Expecting of sizes:
            estimates [batch, trials]
            mus [batch, N]
            kps [batch, N]
            pis [batch, N+1], where pis[:,0] is for the uniform component!

        Returns "pi^p"
        """
        component_posteriror = self.bayes_rule(estimates, mus, kps, pis, **kwargs)

        # Marginalisation
        component_estimated_marginal = component_posteriror.mean(1)  # [batch, N+1] aka pi^p
        return component_estimated_marginal

    def generate_example_target_dist(self, mus: _T, kps: _T, pis: _T, num_points: int, **kwargs):
        "Shaping throughout: [batch, xaxis, N or N + 1]"
        mus, kps, pis = [thing.unsqueeze(1) for thing in [mus, kps, pis]]
        x_axis = torch.linspace(-self.data_half_range, +self.data_half_range, num_points).reshape(1, -1, 1)
        normal_likelihoods = self.component_likelihood(x_axis.to(mus.device), mus, kps)
        scaled_likelihoods = self.attach_uniform_likelihood_and_scale(normal_likelihoods, pis).sum(-1)
        return x_axis.squeeze(), scaled_likelihoods

    def data_space_kl(self, samples: _T, mus: _T, kps: _T, pis: _T, num_points: int, backwards: bool, **kwargs):
        """
        For a batch of items, do KL between the system samples drawn and the actual mixture model, in the space of the data
        
        backwards = False gives KL[target || samples]
        backwards = True gives KL[samples || target]
        """
        # [num_points], [batch size, num_points]
        x_axis, likelihoods = self.generate_example_target_dist(mus, kps, pis, num_points)

        bin_boundaries = 0.5 * (x_axis[1:] + x_axis[:-1])
        min_margin = x_axis[[0]] - (bin_boundaries[[0]] - x_axis[[0]])
        max_margin = x_axis[[-1]] + (x_axis[[-1]] - bin_boundaries[[-1]])
        bin_boundaries = torch.concat([min_margin, bin_boundaries, max_margin])
        assert torch.isclose(0.5 * (bin_boundaries[1:] + bin_boundaries[:-1]), x_axis).all()

        _min, _max = bin_boundaries[0].item(), bin_boundaries[-1].item()
        clipped_samples = samples.clip(min=_min, max=_max)
        if (clipped_samples != samples).any():
            print('warning: clipping samples to calculate KL divergence')
            
        densities = bin_boundaries.diff().to(likelihoods.device) * likelihoods

        sample_densities = torch.zeros(densities.shape)
        for i, (a, b) in enumerate(zip(bin_boundaries[:-2], bin_boundaries[1:-1])):
            in_bin = torch.logical_and(a <= clipped_samples, clipped_samples < b)
            sample_densities[:,i] = in_bin.float().mean(-1)

        last_in_bin = torch.logical_and(bin_boundaries[-2] <= clipped_samples, clipped_samples <= bin_boundaries[-1])
        sample_densities[:,-1] = last_in_bin.float().mean(-1)
        
        sample_densities = sample_densities.to(densities.device)

        if backwards:
            return kl(sample_densities, densities)
        else:
            return kl(densities, sample_densities)

    def cdf_transform(self, *args, **kwargs):
        raise NotImplementedError

    def ideal_samples(self, *args, **kwargs):
        raise NotImplementedError


class CircularExplicitMixtureModel(ExplicitMixtureModel):

    "Multimodal vMs + uniform"

    def __init__(self, N: int, conc_lower: float, conc_upper: float, uniform_scale=1.0, component_scale=1.0) -> None:
        super().__init__(N, conc_lower, conc_upper, mu_half_range = torch.pi, data_half_range = torch.pi, uniform_scale=uniform_scale, component_scale=component_scale)

    def component_likelihood(self, x: _T, mus: _T, kps: _T):
        "x of shape [batch, something, 1]"
        exponent_term = (x - mus).cos() * kps
        normalisation_terms = (1 / (2 * torch.pi * I0(kps)))
        normal_likelihoods = normalisation_terms * exponent_term.exp()   # [batch, trials, N]
        return normal_likelihoods

    def cdf_transform(self, x: _T, mus: _T, kps: _T, pis: _T, num_pdf_eval_points: int = None, **kwargs):
        """
        See 7/4/2024 report - this is for a posterior calibration check.
        Take CDF lower bound convention as -pi.
        HOWEVER - CDF is not analytic so we do a very tight pdf evaluation...

        As always:
            x [batch, trials]
            mus [batch, N]
            kps [batch, N]
            pis [batch, N+1], where pis[:,0] is for the uniform component

        output should be flattened first then passed to some calibration curve plotting
        """
        x = rectify_angles(x)
        bs, num_trials = x.shape
        num_pdf_eval_points = num_pdf_eval_points or (5 * num_trials)
        pdf_eval_points = torch.linspace(-torch.pi, torch.pi, num_pdf_eval_points+1)[:-1]   # [num_pdf_eval_points]
        pdf_eval_points = pdf_eval_points.to(mus.device)
        dy = pdf_eval_points[1] - pdf_eval_points[0]

        mus=mus.unsqueeze(1)
        kps=kps.unsqueeze(1)
        pis=pis.unsqueeze(1)
        rep_pdf_eval_points = pdf_eval_points.reshape(1, -1, 1).repeat(bs, 1, 1)

        pdf_comp_eval = self.component_likelihood(x=rep_pdf_eval_points, mus = mus, kps = kps)   # [batch, num_pdf_eval_points, N]

        pdf_eval = self.attach_uniform_likelihood_and_scale(pdf_comp_eval, pis).sum(-1)                 # [batch, num_pdf_eval_points]
        cdf_mask_counts = (x.unsqueeze(1) >= rep_pdf_eval_points).sum(1) - 1                            # [batch, trials]
        
        full_cdf_eval = pdf_eval.cumsum(-1)[:,:-1] * dy
        full_cdf_eval = torch.concat([torch.zeros(bs, 1).to(full_cdf_eval.device), full_cdf_eval], dim = 1)  # [batch, num_pdf_eval_points]

        cdf_eval = []
        for b in range(bs):
            cdf_eval.append(full_cdf_eval[b].index_select(0, cdf_mask_counts[b]))
        cdf_eval = torch.stack(cdf_eval, 0)

        return cdf_eval         # "u" in report notation

    def ideal_samples(self, num_samples: int, mus: _T, kps: _T, pis: _T, N = None, **kwargs):
        "parameters as above, return is of shape [batch, num_samples]"

        if N is None:
            N = self.N

        pi_cdf = pis.cumsum(-1).unsqueeze(1).repeat(1, num_samples, 1)     # [batch, num_samples, N+1]
        bs = pi_cdf.shape[0]
        u = torch.rand(bs, num_samples, 1).to(pis.device)
        sample_selected = (u >= pi_cdf).sum(-1) # check distribution!      # [batch, num_samples]
        
        samples = torch.zeros(bs, num_samples).to(pis.device)
        uniform_samples = torch.rand_like(samples).to(pis.device) * 2 * torch.pi
        vonmises_object = torch.distributions.VonMises(mus, kps)
        vonmises_samples = vonmises_object.sample([num_samples]).transpose(0, 1)

        samples[sample_selected == 0] = uniform_samples[sample_selected == 0]
        for n in range(N):
            indexer = (sample_selected == (n+1))
            samples[indexer] = vonmises_samples[:,:,n][indexer]

        return rectify_angles(samples)
        


class CircularExplicitMixtureModelNPModelStats(CircularExplicitMixtureModel):

    def __init__(
        self, 
        N: int, 
        generative_model: NonParametricSwapErrorsGenerativeModel, 
        variational_model: NonParametricSwapErrorsVariationalModel,
        kernel_noise_sigma: float,
        feature_borders = [torch.pi / 8, torch.pi / 8],
        target_feature_idx = 1
        ) -> None:
        super().__init__(N, 0.0, 0.0)
        self.generative_model = generative_model
        self.variational_model = variational_model
        self.kernel_noise_sigma = kernel_noise_sigma
        
        assert self.generative_model.num_features == len(feature_borders)
        self.num_features = self.generative_model.num_features
        self.feature_borders = feature_borders
        assert 0 <= target_feature_idx < self.num_features
        self.target_feature_idx = target_feature_idx

    def generate_displays(self, batch_size):
        all_feature_lists = []
        for b in range(batch_size):
            all_feature_lists.append([
                generate_circular_feature_list(num_stim = self.N, feature_border = self.feature_borders[i])
                for i in range(self.num_features)
            ]) # Makes sure feature_border is not violated
        return torch.tensor(all_feature_lists).transpose(1, 2)  # [batch size, N, D (2)]

    def generate_batch(self, batch_size, features = None, return_features = False, shuffle = True, device = 'cuda'):

        if features is None:
            features = self.generate_displays(batch_size).to(device)   # [B, N, D]
        else:
            assert features.shape[0] == batch_size

        N = features.shape[1]

        mixture_model_parameters = generate_gp_distribution(features, self.generative_model, self.variational_model, self.kernel_noise_sigma, self.target_feature_idx)

        if shuffle:
            perm = torch.randperm(N)
        else:
            perm = torch.arange(N)

        mus = mixture_model_parameters['mus']
        kps = mixture_model_parameters['kps']
        pis = mixture_model_parameters['pis']

        shuffled_mus = mus[:,perm]
        shuffled_kps = kps[:,perm]
        shuffled_pis = torch.concat([pis[:,[0]], pis[:,perm + 1]], dim = 1)

        ret_dict = dict(
            mus = shuffled_mus.float(), # [B, N]
            kps = shuffled_kps.float(), # [B, N]
            pis = shuffled_pis.float(), # [B, N+1]
        )

        if return_features:
            ret_dict['features'] = features[:,perm]
        
        return ret_dict




class CircularPPCMixtureModel(CircularExplicitMixtureModel):

    "Encode component mean and conc as PPC (i.e. scaled von Mises tuning curves) mean. pi = 1/N and no uniform component"

    def __init__(self, k: int, d_r: int, N: int, conc_lower: float, conc_upper: float, device = 'cuda') -> None:
        super().__init__(N, conc_lower, conc_upper)
        assert k > 0.0  # XXX: maybe tighten this bound
        self.k = torch.tensor(k)
        self.pref_orientations = torch.linspace(-torch.pi, torch.pi, d_r + 1)[1:].reshape(1, -1, 1).to(device)
        self.tuning_curve_norm_factor = 2.0 * torch.pi * I0(self.k)
        self.device = device

    def attach_uniform_likelihood_and_scale(self, component_likelihoods, pis):
        return component_likelihoods * pis

    def enc_f(self, orientations: _T):
        "orientations come in as [batch, N]. Tuning curves leave as [batch, d_r, N]"
        orientations = orientations.unsqueeze(1)
        orientations_differences = self.pref_orientations - orientations
        exponent = self.k * (orientations_differences).cos()
        return exponent.exp() / self.tuning_curve_norm_factor

    def generate_batch(self, batch_size):
        "Add scaled_f_vector of shape [batch, d_r, N] - NB no Poisson yet"
        batch = super().generate_batch(batch_size)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        mus, kps, pis = batch['mus'], batch['kps'], batch['pis']
        pis = torch.ones(batch_size, self.N) / self.N
        gains = kps / self.k    # [batch, N]
        f_vector = self.enc_f(mus)
        scaled_f_vector = gains.unsqueeze(1) * f_vector
        return {'mus': mus, 'kps': kps, 'pis': pis, 'scaled_f_vector': scaled_f_vector}

    def cdf_transform(self, *args, **kwargs):
        raise NotImplementedError

    def ideal_samples(self, *args, **kwargs):
        raise NotImplementedError
