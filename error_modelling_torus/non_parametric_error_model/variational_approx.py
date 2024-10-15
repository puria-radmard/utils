from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as _T

from math import prod, log as mathlog
from itertools import product
import matplotlib.pyplot as plt

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

import warnings

from numpy import ndarray as _A

from typing import Optional, List, Union, Dict


class NonParametricSwapErrorsVariationalModel(nn.Module):

    """
    Container for variational parameters:
        Z:      set of R inducing points, shaped [R, 2]
        m_u:   variational mean function evaluated at Z, shaped [R]
        S_uu:   variational covariance function evaluated at Z, shaped [R] if inducing_point_variational_parameterisation == 'gaussian' or zero matrix if 'vanilla'
    
    We do inference of the variational posterior q(f),
        and sampling from it, here.

    We also calculate the ELBO KL term here, for completeness

    Finally, we have some util functions to get rid of repeated data at the right times
        Repeated data will of couse come up when dealing with the delta data - every N^m
        delta will be completely zero!

        Therefore, we need to take the data out before calculating K_dd and k_ud
        Then, we need to reinsert the f(delta) in the right places in the evaulation tensor

    Universal indices:
        I = number of samples when doing MC
        M = number of trials
        N = number of stimuli in a trial
        R = number of inducing points
        Q = number of models being trained in parallel
    """

    def __init__(self, num_models: int, R_per_dim: int, num_features: int, fix_non_swap: bool, fix_inducing_point_locations: bool, symmetricality_constraint: bool, min_seps: Optional[_T], inducing_point_variational_parameterisation: str):

        super(NonParametricSwapErrorsVariationalModel, self).__init__()

        self.num_models = num_models    # Q
        self.num_features = num_features    # D
        self.fix_non_swap = fix_non_swap
        self.fix_inducing_point_locations = fix_inducing_point_locations
        self.symmetricality_constraint = symmetricality_constraint
        self.inducing_point_variational_parameterisation = inducing_point_variational_parameterisation

        if symmetricality_constraint:   # For inducing points
            self.all_quadrant_mults = []
            for quadrant_mult in list(product([-1.0, 1.0], repeat = self.num_features)):
                self.all_quadrant_mults.append(torch.tensor(quadrant_mult).unsqueeze(0).unsqueeze(0))   # [1, 1, D] to multiply rectify_angles(inducing_points_tilde)

        if min_seps is not None:
            assert list(min_seps.shape) == [num_features]
            initial_inducing_points_per_axis = [self.generate_points_around_circle_with_min_separation(R_per_dim, ms, symmetricality_constraint) for ms in min_seps]
        else:
            initial_inducing_points_per_axis = [self.generate_points_around_circle_with_min_separation(R_per_dim, None, symmetricality_constraint) for _ in range(num_features)]
        torus_points = self.generate_torus_points_from_circle_points(initial_inducing_points_per_axis)
        self.R = R_per_dim ** num_features

        self.min_seps = min_seps

        self.register_parameter('inducing_points_tilde', nn.Parameter(torus_points.unsqueeze(0).repeat(num_models, 1, 1), requires_grad = not fix_inducing_point_locations))    # [Q, R (unless if symmetric), D]
        self.register_parameter('m_u_raw', nn.Parameter(torch.zeros(num_models, self.inducing_points_tilde.shape[1], dtype = torch.float64), requires_grad = True))                 # [Q, R (unless if symmetric)]
        
        if inducing_point_variational_parameterisation == 'gaussian':
            
            # # if symmetricality_constraint:
            # #     # SEE THIS FOR ALL DETAILS ON NOTATION https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices
            # #     # XXX: WRITE ALL OF THIS UP
            # #     bs = self.R // len(self.all_quadrant_mults)  # block size
            # #     self.register_parameter('S_uu_L_a_log', nn.Parameter(torch.zeros(bs, bs, dtype = torch.float64), requires_grad = True))
            # #     self.register_parameter('S_uu_B', nn.Parameter(torch.zeros(bs, bs, dtype = torch.float64), requires_grad = True))

            # # else:
                self.register_parameter('S_uu_log_chol', nn.Parameter(torch.zeros(num_models, self.R, self.R, dtype = torch.float64), requires_grad = True))    # [Q, R (always), R]

    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.num_models = 1
        self.register_parameter('inducing_points_tilde', nn.Parameter(self.inducing_points_tilde[[model_index]], requires_grad = not self.fix_inducing_point_locations))
        self.register_parameter('m_u_raw', nn.Parameter(self.m_u_raw[[model_index]], requires_grad = True))
        if self.inducing_point_variational_parameterisation == 'gaussian':
            self.register_parameter('S_uu_log_chol', nn.Parameter(self.S_uu_log_chol[[model_index]], requires_grad = True)) 

    @classmethod
    def from_typical_args(
        cls, *_, 
        num_models: int,
        swap_type: str,
        R_per_dim: int,
        fix_non_swap: bool, 
        fix_inducing_point_locations: bool,
        all_min_seps: Optional[_T],
        inducing_point_variational_parameterisation_type: str,
        symmetricality_constraint: bool,
        shared_swap_function: bool,
        all_set_sizes: List[int],
        device = 'cuda',
        **kwargs
        ) -> Optional[Dict[int, NonParametricSwapErrorsVariationalModel]]:
        """
        This is to replace the old setup_utils.py function logic!
        
        If min_seps is provided, expecting it in shape [len(all_set_sizes), D], all > 0
            if shared_swap_function then take min along the first dimension

        Remember that when loading a model, there is no need to provide min_seps - it's just for inducing points initalisation, and loading a state dict with solve this!
            Even if fix_inducing_point_locations given, inducing_points_tilde is still a parameter, just doesn't require grad
        """
        assert swap_type in ['cue_dim_only', 'est_dim_only', 'full', 'spike_and_slab']
        if swap_type == 'spike_and_slab':
            return None, ..., 0

        delta_dimensions = [0] if swap_type == 'cue_dim_only' else [1] if swap_type == 'est_dim_only' else [0, 1]
        D = 1 if swap_type in ['cue_dim_only', 'est_dim_only'] else 2

        set_sizes = [0] if shared_swap_function else all_set_sizes

        if all_min_seps is None:
            all_min_seps_by_set_size = {ss: None for ss in set_sizes}
        else:
            assert list(all_min_seps.shape) == [len(all_set_sizes), 2], f"Not expecting all_min_seps of shape {list(all_min_seps.shape)}"
            assert (all_min_seps >= 0.0).all()
            assert fix_non_swap, "Should not have a min separation if delta = 0 is in the swap functions domain"
            assert swap_type != 'spike_and_slab', "Specifying min_seps does not make sense for spike_and_slab model!"
            all_min_seps = all_min_seps[:,delta_dimensions]
            all_min_seps_by_set_size = {ss: min_seps for ss, min_seps in zip(set_sizes, all_min_seps)}

        return {
            str(ss): cls(
                num_models = num_models,
                R_per_dim = R_per_dim,
                num_features = D,
                fix_non_swap = fix_non_swap,
                fix_inducing_point_locations = fix_inducing_point_locations,
                symmetricality_constraint = symmetricality_constraint,
                min_seps = all_min_seps_by_set_size[ss],
                inducing_point_variational_parameterisation = inducing_point_variational_parameterisation_type
            ).to(device)
            for ss in all_set_sizes
        }, delta_dimensions, D


    @staticmethod
    def generate_points_around_circle_with_min_separation(R_d: int, min_sep: Optional[float], symmetricality_constraint: bool):
        """
        if min sep = 0, code is self explanatory
        else:
            Generate evenly spaced points from (-pi, -min_sep] and [min_sep, +pi)
            Ensuring that first and last are also evenly separated as the rest of points.
                i.e. initial_inducing_points[0] - -pi == initial_inducing_points[1] - initial_inducing_points[0] == ... == pi - initial_inducing_points[-1]
            As such, this case requires even R_d
        """
        if min_sep == None or min_sep == 0.0:
            if symmetricality_constraint:
                assert R_d % 2 == 0.0
                initial_inducing_points = torch.linspace(0.0, +torch.pi, R_d + 1)[1:-1:2]  # XXX: none at zero here!

            else:
                initial_inducing_points = torch.linspace(-torch.pi, +torch.pi, R_d + 1)[:-1]

        else:
            assert R_d % 2 == 0.0
            neg_hemiphere_points = torch.linspace(-torch.pi, -min_sep, R_d)[1::2] # Double the points required so we can get the half diff at the start
            pos_hemiphere_points = torch.linspace(+min_sep, +torch.pi, R_d)[::2] # Double the points required so we can get the half diff at the start
            assert torch.isclose(neg_hemiphere_points[0] + torch.pi, torch.pi - pos_hemiphere_points[-1])
            assert torch.isclose(neg_hemiphere_points[0] + torch.pi, (neg_hemiphere_points[1] - neg_hemiphere_points[0]) / 2)
            assert torch.isclose(neg_hemiphere_points[0] + torch.pi, (pos_hemiphere_points[1] - pos_hemiphere_points[0]) / 2)
            initial_inducing_points = torch.concat([neg_hemiphere_points, pos_hemiphere_points])

            if symmetricality_constraint:
                assert R_d % 2 == 0.0
                initial_inducing_points = torch.linspace(min_sep, +torch.pi, R_d+1)[1::2]  # XXX: none at zero here!

        return initial_inducing_points

    @staticmethod
    def generate_torus_points_from_circle_points(ordered_circular_points: List[_T], dtype=torch.float64):
        D = len(ordered_circular_points)
        num_points = prod(map(len, ordered_circular_points))
        if D == 1:
            return ordered_circular_points[0].reshape(num_points, 1)
        else:
            grid_points = torch.meshgrid(*ordered_circular_points, indexing='ij')
            torus_points = torch.stack(grid_points, -1).reshape(num_points, D).to(torch.float64)
            return torus_points

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    @property
    def Z(self):            # [Q, R, D]
        if self.symmetricality_constraint:
            positive_quadrant = rectify_angles(self.inducing_points_tilde)
            # assert (positive_quadrant >= 0.0).all()                         # No actual constraint for this, just hoping it works!
            all_quadrants = []
            for quadrant_mult in self.all_quadrant_mults:
                new_quadrant = positive_quadrant * quadrant_mult.to(positive_quadrant.device)
                all_quadrants.append(new_quadrant)
            return torch.concat(all_quadrants, 1)
        else:
            return rectify_angles(self.inducing_points_tilde)

    @property
    def m_u(self):              # [Q, R]
        if self.symmetricality_constraint:
            positive_quadrant = self.m_u_raw
            all_quadrants = []
            for quadrant_mult in self.all_quadrant_mults:
                new_quadrant = positive_quadrant
                all_quadrants.append(new_quadrant)
            return torch.concat(all_quadrants, 1)
        else:
            return self.m_u_raw

    @property
    def S_uu_chol(self):         # [Q, R, R]
        if self.inducing_point_variational_parameterisation == 'vanilla':
            raise TypeError('Cannot access S_uu_chol for vanilla variational model')
        else:
            chol = torch.tril(self.S_uu_log_chol)
            diag_index = range(self.R)
            chol[:, diag_index, diag_index] = 1.0 + chol[:,diag_index, diag_index].exp()           # [Q, R, R]  
            return chol      

    @property
    def S_uu(self):         # [Q, R, R]
        if self.inducing_point_variational_parameterisation == 'vanilla':
            return 0.0

        else:
            chol = self.S_uu_chol
            S_uu = torch.bmm(chol, chol.transpose(1, 2))
            return S_uu
    
    def sample_from_variational_prior(self, num_samples: int) -> Dict[str, _T]:
        """
        Simply sample from N(m_u, S_uu)
        Returns:
            samples of shape [Q, K, R] where K is num_samples
            sample_likelihoods of shape [Q, K]
        """
        S_uu_chol = self.S_uu_chol                  # [Q, R, R]
        m_u = self.m_u.unsqueeze(1)                 # [Q, 1, R]
        eps = torch.randn(self.num_models, num_samples, self.R, dtype = m_u.dtype, device = m_u.device) # [Q, K, R]
        original_lhs = mathlog((2 * torch.pi)**(-self.R / 2.)) + (-0.5 * (eps * eps).sum(-1))                        # [Q, K]
        cholesky_determinant = S_uu_chol.diagonal(offset = 0, dim1 = -1, dim2 = -2).prod(-1,keepdim=True)        # [Q, 1]
        new_lhs = original_lhs / cholesky_determinant
        return {
            'samples': m_u + torch.bmm(eps, S_uu_chol.transpose(-1, -2)), # [Q, K, R]
            'sample_log_likelihoods': new_lhs   # [Q, K]
        }

    @staticmethod
    def batched_inner_product(M, x):
        """
        x^\intercal M x
            x of shape [Q, T]
            M of shape [Q, T, T]
            output of shape [Q]
        """
        return torch.bmm(x.unsqueeze(1), torch.bmm(M, x.unsqueeze(-1))).squeeze(-1).squeeze(-1)

    @staticmethod
    def batched_inner_product_matrix(M, X):
        """
        X^\intercal M X
            X of shape [Q, T, S]
            M of shape [Q, T, T]
            output of shape [Q, S, S]
        """
        return torch.bmm(X.transpose(-1, -2), torch.bmm(M, X))

    @staticmethod
    def batched_inner_product_mix(M, X, x):
        """
        X^\intercal M x
            X of shape [Q, T, S]
            M of shape [Q, T, T]
            x of shape [Q, T]
            output of shape [Q, S]
        """
        return torch.bmm(X.transpose(-1, -2), torch.bmm(M, x.unsqueeze(-1))).squeeze(-1)

    def kl_loss(self, K_uu: _T, K_uu_inv: _T) -> _T:    # [Q]
        """
            K_uu is the kernel evaluated on the inducing points -> [Q, R, R]
                We only need to inverse of this!
        """
        if self.inducing_point_variational_parameterisation == 'vanilla':
            warnings.warn('kl loss for self.inducing_point_variational_parameterisation = vanilla not evaluated')
            return torch.tensor(0.0)
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

    def variational_gp_inference(self, k_ud: _T, K_dd: _T, K_uu_inv: _T):
        """
            k_ud is the kernel evaluated on the inducing points against the real data -> [Q, R, MN]
            K_dd is the kernel evaluated on the real delta data -> [Q, MN, MN]
            K_uu is the kernel evaluated on the inducing points -> [Q, R, R]
                We need the inverse of this!

            Infer GP parameters for q(f)

            sigma: [Q, MN, MN]
            mu: [Q, MN]
        """
        if self.inducing_point_variational_parameterisation == 'vanilla':
            sigma = K_dd - self.batched_inner_product_matrix(K_uu_inv, k_ud)    # [Q, MN, MN]
        elif self.inducing_point_variational_parameterisation == 'gaussian':
            diff_S = self.batched_inner_product_matrix(self.S_uu, K_uu_inv)
            sigma = K_dd - self.batched_inner_product_matrix(K_uu_inv, k_ud) + self.batched_inner_product_matrix(diff_S, k_ud)  # [Q, MN, MN]
        mu = self.batched_inner_product_mix(K_uu_inv, k_ud, self.m_u)   # [Q, MN]
        
        sigma_perturb = torch.eye(sigma.shape[1], device = sigma.device, dtype = sigma.dtype).unsqueeze(0).repeat(self.num_models, 1, 1)
        sigma_chol = torch.linalg.cholesky(sigma + 1e-3 * sigma_perturb)
            # eigval, eigvec = torch.linalg.eig(sigma)
            # eigval[eigval.real < 0.0] = 1e-5
            # sigma_recon = (eigvec @ torch.diag(eigval) @ eigvec.T)
            # print('Reconstruction error:', (sigma_recon - sigma).abs().max().item())
            # sigma_chol = torch.linalg.cholesky(sigma_recon + 1e-6 * torch.eye(sigma.shape[0], device = sigma.device, dtype = sigma.dtype)).real
        # try:
        #     sigma_chol = torch.linalg.cholesky(sigma)
        # except:
        #     sigma_chol = torch.linalg.cholesky(sigma + 1e-3 *torch.eye(sigma.shape[0], device = sigma.device, dtype = sigma.dtype))
        # # try:
        # #     sigma_chol = torch.linalg.cholesky(sigma)
        # # except torch._C._LinAlgError as e:
        # #     print(e)
        # #     eigval, eigvec = torch.linalg.eig(sigma)
        # #     eigval, eigvec = eigval.real, eigvec.real
        # #     eigval[eigval < 0.0] = 1e-5
        # #     sigma_recon = (eigvec @ torch.diag(eigval) @ eigvec.T)
        # #     print('Reconstruction error:', (sigma_recon - sigma).abs().max().item())
        # #     try:
        # #         sigma_chol = torch.linalg.cholesky(sigma_recon)
        # #     except torch._C._LinAlgError as e2:
        # #         print(e2)
        # #         import pdb; pdb.set_trace()
        return mu, sigma, sigma_chol   # [Q, MN], [Q, MN, MN], [Q, MN, MN]

    def variational_gp_inference_mean_only(self, k_ud: _T, K_uu_inv: _T):
        """
            k_ud is the kernel evaluated on the inducing points against the real data -> [Q, R, MN]
            K_uu is the kernel evaluated on the inducing points -> [Q, R, R]
                We also need to inverse of this!

            Infer GP mean parameter for q(f)

            Returns [Q, MN]
        """
        mu = self.batched_inner_product_mix(K_uu_inv, k_ud, self.m_u)
        return mu

    def variational_gp_inference_conditioned_on_inducing_points_function(self, u: _T, k_ud: _T, K_dd: _T, K_uu_inv: _T):
        """
        See variational_gp_inference for logic and most shapes
        This time, rather than integrating out the variational function distribution over inducing point, we fix it at u
        There is a key difference in the logic here:
            
            Shape of u: [Q, K, R] where Q and R are defined above, and K is number of samples (not the same as I, the number of samples when estimating the ELBO)
            
            We are conditioning at many points at once, so we generate many mus and sigmas at once
            
            However, we only reflect this in the shape of mu, which is given a new dimension -> [Q, K, MN]
        """
        assert self.inducing_point_variational_parameterisation == 'gaussian', "Cannot condition on u vector if inducing_point_variational_parameterisation is 'vanilla'"
        assert tuple(u.shape) == (self.num_models, u.shape[1], self.R)

        sigma = K_dd - self.batched_inner_product_matrix(K_uu_inv, k_ud)    # [Q, MN, MN]
        
        mu_first = torch.bmm(u, K_uu_inv.transpose(1, 2)) # [Q, K, R] @ [Q, R, R] -> [Q, K, R]
        mu = (mu_first.unsqueeze(-1) * k_ud.unsqueeze(1)).sum(-2) # [Q, K, R, 1] * [Q, 1, R, MN] -> [Q, K, R, MN] -> [Q, K, MN]
        
        sigma_perturb = torch.eye(sigma.shape[1], device = sigma.device, dtype = sigma.dtype).unsqueeze(0).repeat(self.num_models, 1, 1)
        sigma_chol = torch.linalg.cholesky(sigma + 1e-3 * sigma_perturb)
        return mu, sigma, sigma_chol   # [Q, K, MN], [Q, MN, MN], [Q, MN, MN]
    
    def reparameterised_sample(self, num_samples: int, mu: _T, sigma_chol: _T, M: int, N: int, unsqueeze_mu = True):
        """
            if unsqueeze_mu: mu and sigma_chol come from variational_gp_inference: [Q, MN] and [Q, MN, MN]
            else: mu and sigma_chol come from variational_gp_inference_conditioned_on_inducing_points_function: [Q, K, MN] and [Q, MN, MN]
            
            num_samples is I!

            return is of shape [Q, I, M, N]
        """
        if unsqueeze_mu:
            mu = mu.unsqueeze(1)
            num_samples_reshape = num_samples
        else:
            assert num_samples == 1, "Cannot sample on reparameterised_sample"
            num_samples_reshape = mu.shape[1]
        deduped_MN = mu.shape[-1]
        eps = torch.randn(self.num_models, num_samples, deduped_MN, dtype = mu.dtype, device = mu.device) # [Q, I, MN (dedup size)]
        model_evals = mu + torch.bmm(eps, sigma_chol.transpose(-1, -2))   # [Q, I, MN]
        readded_model_evals = self.reinclude_model_evals(model_evals, M, N, num_samples_reshape)    # [Q, I or K, M, N]
        return readded_model_evals

    def reinclude_model_evals(self, model_evals: _T, num_displays: int, set_size: int, num_mc_samples: int):
        """
        if self.fix_non_swap:
            Input in shape [Q, I, M*(N-1)]
        else:
            Input in shape [Q, I, 1 + M*(N-1)]

        Last shape axis is called "MN" or "dedup size" above

        Returns [Q, I, M, N]
        """
        if self.fix_non_swap:
            zero_f_eval = -float('inf')*torch.ones(self.num_models, num_mc_samples, num_displays, 1, dtype = model_evals.dtype, device = model_evals.device)   # [Q, I, M, 1]
            if set_size > 1:
                regrouped_model_evals = model_evals.reshape(self.num_models, num_mc_samples, num_displays, set_size-1)    # [Q, I, M, N-1]
                readded_model_evals = torch.concat([zero_f_eval, regrouped_model_evals], -1)
                return readded_model_evals  # [Q, I, M, N]
            else:
                return zero_f_eval  # [Q, I, M, N (1)]
        else:
            zero_f_eval = model_evals[...,[0]].unsqueeze(1).repeat(1, num_displays, 1)   # [Q, I, M, 1]
            if set_size > 1:
                regrouped_model_evals = model_evals[...,1:].reshape(self.num_models, num_mc_samples, num_displays, set_size-1)    # [Q, I, M, N-1]
                readded_model_evals = torch.concat([zero_f_eval, regrouped_model_evals], -1)
                return readded_model_evals  # [Q, I, M, N]
            else:
                return zero_f_eval  # [Q, I, M, N (1)]

    def deduplicate_deltas(self, deltas: _T, batch_size: int = 0) -> Union[_T, List[_T]]:
        """
        Every Mth delta will be a 0 by design
        If the non-swapped item (i.e. cued item) is included in the swap function, we remove this to ensure a correct kernel calculation downstream
        If not (i.e. self.fix_non_swap = True), then we don't include it in the delta

        Input: [Q, M, N, D], i.e. have to assume that data unique/repeated for each model
        Output = 
            1. list containing a single element of size [Q, ~MN, D] if batch_size = 0, otherwise a ~M/batch_size length list of entries of shape [Q, ~N*batch_size, D]
            2. list containing all the sizes of batches
        """
        Q, M, N, D = deltas.shape
        assert Q == self.num_models
        if batch_size < 1:
            flattened_deltas = deltas.reshape(Q, M * N, D)
            if self.fix_non_swap:
                unique_indices = [i for i in range(M * N) if i % N != 0]
            else:
                unique_indices = [i for i in range(M * N) if i == 0 or i % N != 0]
            dedup_deltas = [flattened_deltas[:,unique_indices]] # i.e. index M axis
            Ms = [M]                  
        else:
            num_batches = (M // batch_size) + (1 if M % batch_size else 0)
            dedup_deltas_and_Ms = [self.deduplicate_deltas(deltas[:,j*batch_size:(j+1)*batch_size], 0) for j in range(num_batches)]
            dedup_deltas, Ms = zip(*dedup_deltas_and_Ms)
            Ms = [a[0] for a in Ms]
            dedup_deltas = [a[0] for a in dedup_deltas]
        return dedup_deltas, Ms
