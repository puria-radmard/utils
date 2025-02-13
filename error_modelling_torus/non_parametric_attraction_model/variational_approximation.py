import torch
from torch import nn
from torch import Tensor as _T

from dataclasses import dataclass

import warnings

from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes

from typing import List, Optional, Dict, Callable

from purias_utils.error_modelling_torus.non_parametric_attraction_model.parameters_gp_prior import KernelEvaluationInfo

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles



@dataclass
class InferenceEvaluationInfo:
    """
    Gaussian moments for Q (num_models) different functions evaluated at 
        B (num_data) different points of the input
    """

    mu: _T
    sigma: _T
    sigma_chol: _T

    def __post_init__(self):
        num_models, num_data = self.mu.shape
        self.num_data = num_data
        self.num_models = num_models
        assert tuple(self.sigma.shape) == (num_models, num_data, num_data)
        assert tuple(self.sigma_chol.shape) == (num_models, num_data, num_data)

    def take_samples(self, num_samples: int):
        """
        Output of shape [Q, I, B]
            I is the number of samples of the function we want to take
            i.e. function_evals[q,i,:] gives smooth function, 
                function_evals[q,:,b] gives you all the function evaluations sampled at one point
        """
        num_data = self.mu.shape[-1]
        eps = torch.randn(self.num_models, num_samples, num_data, dtype = self.mu.dtype, device = self.mu.device) # [Q, I, B]
        function_evals = self.mu.unsqueeze(1) + torch.bmm(eps, self.sigma_chol.transpose(-1, -2))   # [Q, I, B]
        return function_evals

    @staticmethod
    def prepare_samples_for_ancestor_sampling(samples: _T) -> _T:
        """
        Takes in shape [Q, I, B] with I == B
        Spits out shape [Q, K], K == B, i.e. one valid sample from each model-(data input) pair
        """
        _, I, B = samples.shape
        assert B == I, \
            """
            InferenceEvaluationInfo.prepare_samples_for_ancestor_sampling requires that you have the same
                number of function samples as you have datapoints, i.e. samples of shape [Q, I, B] and I == B
            """
        extracted_samples = samples.diagonal(0, 1, 2)
        return extracted_samples

    def plot_to_axes(self, evaluation_locations: _T, model_idx: int, axes: Axes, num_samples: int = 0, mean_line_kwargs: Dict = {}, std_fill_kwargs: Dict = {}, link_function: Optional[Callable] = None):
        std_line = self.sigma[model_idx].diagonal().sqrt()
        mu_line = self.mu[model_idx]
        upper_line = mu_line + std_line
        lower_line = mu_line - std_line
        if link_function is not None:
            mu_line = link_function(mu_line)
            upper_line = link_function(upper_line)
            lower_line = link_function(lower_line)
        upper_line = upper_line.detach().cpu().numpy()
        lower_line = lower_line.detach().cpu().numpy()
        mu_line = mu_line.detach().cpu().numpy()
        axes.plot(
            evaluation_locations.cpu().numpy(),
            mu_line,
            **mean_line_kwargs
        )
        axes.fill_between(
            evaluation_locations.cpu().numpy(),
            lower_line,
            upper_line,
            **std_fill_kwargs
        )
        if num_samples > 0:
            samples = self.take_samples(num_samples)[model_idx]
            if link_function is not None:
                samples = link_function(samples)
            samples = samples.detach().cpu().numpy()
            for sample in samples:
                axes.plot(evaluation_locations.cpu().numpy(), sample, color = 'black', alpha = 0.3)
        

@dataclass
class NoInputInferenceEvaluationInfo(InferenceEvaluationInfo):

    def __post_init__(self):
        super().__post_init__()
        for q in range(self.num_models):
            try:
                assert len(self.mu[q].unique()) == 1
            except:
                import pdb; pdb.set_trace()
            assert len(self.sigma[q].unique()) == 1
            assert len(self.sigma_chol[q].unique()) == 1

    def take_samples(self, num_samples: int):
        """
        Under one sample (axis 1), the samples are the same for all locations (axis 2) for each individual model (axis 0)
        """
        jitter = torch.randn(self.num_models, num_samples).to(self.mu.device).unsqueeze(-1)
        parameters_evals = self.mu.unsqueeze(1) + self.sigma_chol.diagonal(0, 1, 2).unsqueeze(1) * jitter        # Not function_evals!
        return parameters_evals

    def plot_to_axes(
        self, evaluation_locations: _T, 
        model_idx: int, axes: Axes, num_samples: int = 0, mean_line_kwargs: Dict = {}, std_fill_kwargs: Dict = {},
        link_function: Optional[Callable] = None
    ):
        """
        evaluation_locations are ignored!
        """
        mean = self.mu[model_idx,0]
        std = self.sigma_chol[model_idx,0,0]
        if std == 0.0:
            grid = torch.tensor([mean - 0.1, mean, mean + 0.1])
            pdf = torch.tensor([0.0, 1.0, 0.0])
            if link_function is not None:
                grid = link_function(grid)
        else:
            upper = self.mu[model_idx,0] + std * 5
            lower = self.mu[model_idx,0] - std * 5
            num_grid = 256
            grid = torch.linspace(lower, upper, num_grid).to(mean.device)
            pdf = torch.distributions.Normal(loc = mean, scale=std + 1e-9).log_prob(grid).exp()
            if link_function is not None:
                grid = link_function(grid)
            axes.fill_between(
                grid.cpu(),
                0.0 * grid.cpu(),
                pdf.cpu(),
                **std_fill_kwargs
            )
        axes.plot(
            grid.cpu(),
            pdf.cpu(),
            **mean_line_kwargs
        )

        if num_samples > 0:
            samples = self.take_samples(num_samples)[model_idx,:,0].detach().cpu()
            if link_function is not None:
                samples = link_function(samples)
            axes.hist(samples.numpy(), density = True, bins = int(num_samples/5), alpha = 0.3)



class SVGPApproximation(nn.Module):
    """
    Sparse approximation along the target report value dimension

    R: int - num inducing points
    """

    def __init__(self, num_models: int, R: int, fix_inducing_point_locations: bool) -> None:
        super().__init__()

        self.num_models = num_models
        self.R = R

        initial_inducing_locations = torch.linspace(0, 2*torch.pi, R + 1)[:-1].unsqueeze(0).repeat(num_models, 1)
        self.register_parameter('inducing_points_tilde', nn.Parameter(initial_inducing_locations, requires_grad = not fix_inducing_point_locations))    # [Q, R (unless if symmetric)]
        self.register_parameter('m_u_raw', nn.Parameter(0.1 * torch.randn(num_models, R, dtype = torch.float64), requires_grad = True))       # [Q, R (unless if symmetric)]
        self.register_parameter('S_uu_log_chol', nn.Parameter(torch.tril((0.01 * torch.randn(num_models, self.R, self.R, dtype = torch.float64))), requires_grad = True))    # [Q, R (always), R]

    @property
    def Z(self) -> _T:            # [Q, R]
        return rectify_angles(self.inducing_points_tilde)

    @property
    def m_u(self) -> _T:              # [Q, R]
        return self.m_u_raw

    @property
    def S_uu_chol(self) -> _T:         # [Q, R, R]
        chol = torch.tril(self.S_uu_log_chol)
        diag_index = range(self.R)
        chol[:, diag_index, diag_index] = torch.nn.functional.softplus(chol[:,diag_index, diag_index])           # [Q, R, R]  
        return chol      

    @property
    def S_uu(self) -> _T:         # [Q, R, R]
        chol = self.S_uu_chol
        S_uu = torch.bmm(chol, chol.transpose(1, 2))
        return S_uu

    def batched_inner_product_matrix(self, M, X):
        """
        TODO: move to some utils!!!

        X.T M X
            X of shape [Q, T, S]
            M of shape [Q, T, T] --> MX of shape [Q, T, S] --> X^T MX of shape [Q, S, S]
            output of shape [Q, S, S]
        """
        return torch.einsum('qtc,qtd->qcd', X, torch.einsum('qab,qbs->qas', M, X))

    def batched_inner_product_mix(self, M, X, x):
        """
        TODO: move to some utils!!!

        X.T M x
        X of shape [Q, T, U]
        M of shape [Q, T, T]
        x of shape [Q, T]
        output of shape [Q, U]
        """
        return torch.einsum('qtu,qt->qu', X, torch.einsum('qab,qb->qa', M, x))

    def batched_inner_product(self, M, x, unsqueeze_x = True):
        """
        x.T M x
            x of shape [Q, T]
            M of shape [Q, T, T]
            output of shape [Q]
        """
        return torch.einsum('qa,qa->q', x, torch.einsum('qab,qb->qa', M, x))

    def variational_gp_inference(self, k_ud: _T, K_dd: _T, K_uu_inv: _T):
        """
            k_ud is the kernel evaluated on the inducing points against the real data -> [Q, R, B]
            K_dd is the kernel evaluated on the real data -> [Q, B, B]
            K_uu is the kernel evaluated on the inducing points -> [Q, R, R]
                We need the inverse of this!

            Infer GP parameters for q(f)

            sigma: [Q, B, B]
            mu: [Q, B]
        """
        diff_S = self.batched_inner_product_matrix(self.S_uu, K_uu_inv)
        sigma = K_dd - self.batched_inner_product_matrix(K_uu_inv, k_ud) + self.batched_inner_product_matrix(diff_S, k_ud)  # [Q, B, B]
        mu = self.batched_inner_product_mix(K_uu_inv, k_ud, self.m_u)   # [Q, B]
        
        sigma_perturb = torch.eye(sigma.shape[1], device = sigma.device, dtype = sigma.dtype).unsqueeze(0).repeat(self.num_models, 1, 1)
        try:
            sigma_chol = torch.linalg.cholesky(sigma + 1e-3 * sigma_perturb)
        except torch._C._LinAlgError:
            sigma_chol = torch.linalg.cholesky(sigma + 1e-2 * sigma_perturb)
        except:
            import pdb; pdb.set_trace()
        return InferenceEvaluationInfo(
            mu=mu, sigma=sigma, sigma_chol=sigma_chol
        )

    def variational_gp_inference_minibatched(
        self, 
        prior_kernel_infos: List[KernelEvaluationInfo],
    ):
        """
        NB: each arg input is a list!
        """
        return [
            self.variational_gp_inference(
                k_ud = pki.k_ud, K_dd = pki.K_dd, K_uu_inv = pki.K_uu_inv,
            ) for pki in prior_kernel_infos
        ]

    def kl_loss(self, K_uu: _T, K_uu_inv: _T) -> _T:    # [Q]
        """
            K_uu is the kernel evaluated on the inducing points -> [Q, R, R]
                We only need to inverse of this!
        """
        Suu_chol = self.S_uu_chol
        S_uu = self.S_uu                                                # [Q, R, R]
        assert K_uu.shape == S_uu.shape == K_uu_inv.shape
        log_det_S_uu = 2 * torch.diagonal(Suu_chol, dim1=-1, dim2=-2).log().sum(-1)                           # [Q]
        log_det_K_uu: _T = torch.logdet(K_uu)                           # [Q]
        log_det_term = (log_det_K_uu - log_det_S_uu)                          # [Q]
        trace_term = torch.diagonal(torch.bmm(K_uu_inv, S_uu), offset=0, dim1=-1, dim2=-2).sum(-1)  # [Q]
        mu_term = self.batched_inner_product(K_uu_inv, self.m_u)        # [Q]
        kl_term = 0.5 * (log_det_term + trace_term + mu_term - self.R)      # [Q]
        return kl_term
    


class NoInputSVGPApproximation(SVGPApproximation):
    """
    Simple case where we have no input (target response), just a single variational approximation
    of the posterior

    The prior learned is over a single point, and 
    """
    def __init__(self, num_models: int) -> None:
        super().__init__(num_models, 1, True)
        self.inducing_points_tilde.data = self.inducing_points_tilde.data * torch.nan    # NoInputAttractionErrorDistributionParametersPrior ignores locations anyway!

    @property
    def variance(self) -> _T:         # [Q]
        return self.log_variance_raw.exp()

    @staticmethod
    def invert_kernel_matrix(K_uu: _T) -> _T:
        return 1.0 / K_uu

    def verify_kernel_inputs(self, *k_matrices):
        error_message = "NoInputSVGPApproximation.variational_gp_inference requires you input kernel matrices with all the same values in all locations"
        for q in range(self.num_models):
            unique_entries = k_matrices[0][q].unique()
            assert len(unique_entries) == 1, error_message
            for km in k_matrices[1:]:
                try:
                    assert (km[q].unique() == unique_entries).all(), error_message
                except:
                    import pdb; pdb.set_trace()
            return unique_entries

    def variational_gp_inference(self, k_ud: _T, K_dd: _T, K_uu_inv: _T):
        """
        All kernel evals should be identical!
        The checks are literally just there t make sure everything works.
        The logic I commented out will lead to the same answer

        k_ud shaped [Q, 1, B]
        K_dd shaped [Q, B, B]
        K_uu_inv shaped [Q, 1, 1]
        """
        self.verify_kernel_inputs(k_ud, K_dd)
        assert tuple(K_uu_inv.shape) == (self.num_models, 1, 1)
        Q, R, B = k_ud.shape
        assert R == 1
        assert tuple(K_dd.shape) == (Q, B, B)
        assert tuple(K_uu_inv.shape) == (Q, 1, 1)

        # k_ud = k_ud[:,0,0]
        # K_uu_inv = K_uu_inv[:,0,0]
        # muz = self.m_u[:,0]
        # K_dd = K_dd[:,0,0]
        # S_uu = self.S_uu[:,0,0]
    
        # mu = (k_ud * K_uu_inv * muz).unsqueeze(-1).repeat(1, B)
        # sigma = (K_dd - (K_uu_inv * (k_ud**2)) + ((K_uu_inv**2) * (k_ud**2) * S_uu))[:,None,None].repeat(1, B, B)
        # sigma_chol = sigma.sqrt()

        mu = self.m_u.repeat(1, B)
        sigma = self.S_uu.repeat(1, B, B)
        sigma_chol = self.S_uu_chol.repeat(1, B, B)

        return NoInputInferenceEvaluationInfo(
            mu=mu, sigma=sigma, sigma_chol=sigma_chol
        )

    def kl_loss(self, K_uu: _T, K_uu_inv: _T) -> _T:    # [Q]
        assert tuple(K_uu.shape) == tuple(K_uu_inv.shape) == (self.num_models, 1, 1)
        return super().kl_loss(K_uu, K_uu_inv)
