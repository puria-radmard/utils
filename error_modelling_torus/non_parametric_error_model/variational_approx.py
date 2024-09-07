import torch
from torch import nn
from torch import Tensor as _T

from math import prod
from itertools import product

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from torch.linalg import det

import warnings

from typing import Optional, List


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
    """

    def __init__(self, R_per_dim: int, num_features: int, fix_non_swap: bool, fix_inducing_point_locations: bool, symmetricality_constraint: bool, min_seps: Optional[_T], inducing_point_variational_parameterisation: str):

        super(NonParametricSwapErrorsVariationalModel, self).__init__()

        self.num_features = num_features
        self.fix_non_swap = fix_non_swap
        self.fix_inducing_point_locations = fix_inducing_point_locations
        self.symmetricality_constraint = symmetricality_constraint
        self.inducing_point_variational_parameterisation = inducing_point_variational_parameterisation

        if symmetricality_constraint:   # For inducing points
            self.all_quadrant_mults = []
            for quadrant_mult in list(product([1.0, -1.0], repeat = self.num_features)):
                self.all_quadrant_mults.append(torch.tensor(quadrant_mult).unsqueeze(0))

        if min_seps is not None:
            assert list(min_seps.shape) == [num_features]
        
        initial_inducing_points_per_axis = [self.generate_points_around_circle_with_min_separation(R_per_dim, ms, symmetricality_constraint) for ms in min_seps]
        torus_points = self.generate_torus_points_from_circle_points(initial_inducing_points_per_axis)
        self.R = R_per_dim * num_features

        if fix_inducing_point_locations:
            self.inducing_points_tilde = torus_points
        else:
            self.register_parameter('inducing_points_tilde', nn.Parameter(torus_points, requires_grad = True))
        self.register_parameter('m_u_raw', nn.Parameter(torch.zeros(len(self.inducing_points_tilde), dtype = torch.float64), requires_grad = True))
        
        if inducing_point_variational_parameterisation == 'gaussian':
            self.register_parameter('S_uu_log_chol', nn.Parameter(torch.zeros(self.R, self.R, dtype = torch.float64), requires_grad = True))

    @staticmethod
    def generate_points_around_circle_with_min_separation(R_d: int, min_sep: float, symmetricality_constraint: bool):
        """
        if min sep = 0, code is self explanatory
        else:
            Generate evenly spaced points from (-pi, -min_sep] and [min_sep, +pi)
            Ensuring that first and last are also evenly separated as the rest of points.
                i.e. initial_inducing_points[0] - -pi == initial_inducing_points[1] - initial_inducing_points[0] == ... == pi - initial_inducing_points[-1]
            As such, this case requires even R_d
        """
        if min_sep == 0.0:
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
            initial_inducing_points = initial_inducing_points[-R_d//2:]

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
        if self.fix_inducing_point_locations:
            self.inducing_points_tilde = self.inducing_points_tilde.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @property
    def Z(self):
        if self.symmetricality_constraint:
            positive_quadrant = rectify_angles(self.inducing_points_tilde)
            all_quadrants = []
            for quadrant_mult in self.all_quadrant_mults:
                all_quadrants.append(positive_quadrant * quadrant_mult.to(positive_quadrant.device))
            return torch.concat(all_quadrants)
        else:
            return rectify_angles(self.inducing_points_tilde)

    @property
    def m_u(self):
        if self.symmetricality_constraint:
            positive_quadrant = self.m_u_raw
            return torch.concat([positive_quadrant for _ in self.all_quadrant_mults])
        else:
            return self.m_u_raw

    @property
    def S_uu(self):
        if self.inducing_point_variational_parameterisation == 'vanilla':
            return 0.0
        chol = torch.tril(self.S_uu_log_chol)
        diag_index = range(self.R)
        chol[diag_index, diag_index] = chol[diag_index, diag_index].exp()
        S_uu = chol @ chol.T
        return S_uu

    def kl_loss(self, K_uu: _T, K_uu_inv: _T) -> _T:
        """
            K_uu is the kernel evaluated on the inducing points -> [R, R]
                We only need to inverse of this!
        """
        if self.inducing_point_variational_parameterisation == 'vanilla':
            warnings.warn('kl loss for self.inducing_point_variational_parameterisation = vanilla not evaluated')
            return torch.tensor(0.0)
        S_uu = self.S_uu
        det_S_uu = det(S_uu)
        det_K_uu = det(K_uu)
        det_term = (det_K_uu / det_S_uu).log()
        trace_term = torch.diag(K_uu_inv @ S_uu).sum()
        mu_term = self.m_u @ (K_uu_inv @ self.m_u)
        return 0.5 * (det_term + trace_term + mu_term - self.R)

    def variational_gp_inference(self, k_ud: _T, K_dd: _T, K_uu: _T, K_uu_inv: _T):
        """
            k_ud is the kernel evaluated on the inducing points against the real data -> [R, MN]
            K_dd is the kernel evaluated on the real delta data -> [M, M]
            K_uu is the kernel evaluated on the inducing points -> [R, R]
                We also need to inverse of this!

            Infer GP parameters for q(f)
        """
        if self.inducing_point_variational_parameterisation == 'vanilla':
            sigma = K_dd - (k_ud.T @ (K_uu_inv @ k_ud))
        elif self.inducing_point_variational_parameterisation == 'gaussian':
            sigma = K_dd - (k_ud.T @ (K_uu_inv @ ((K_uu - self.S_uu) @ (K_uu_inv @ k_ud))))
        mu = k_ud.T @ (K_uu_inv @ self.m_u)
        sigma_chol = torch.linalg.cholesky(sigma)
        # try:
        #     sigma_chol = torch.linalg.cholesky(sigma)
        # except torch._C._LinAlgError as e:
        #     print(e)
        #     eigval, eigvec = torch.linalg.eig(sigma)
        #     eigval, eigvec = eigval.real, eigvec.real
        #     eigval[eigval < 0.0] = 1e-5
        #     sigma_recon = (eigvec @ torch.diag(eigval) @ eigvec.T)
        #     print('Reconstruction error:', (sigma_recon - sigma).abs().max().item())
        #     try:
        #         sigma_chol = torch.linalg.cholesky(sigma_recon)
        #     except torch._C._LinAlgError as e2:
        #         print(e2)
        #         import pdb; pdb.set_trace()
        return mu, sigma, sigma_chol   # [MN], [MN, MN], [MN, MN]

    def variational_gp_inference_mean_only(self, k_ud: _T, K_uu_inv: _T):
        """
            k_ud is the kernel evaluated on the inducing points against the real data -> [R, MN]
            K_uu is the kernel evaluated on the inducing points -> [R, R]
                We also need to inverse of this!

            Infer GP parameters for q(f)
        """
        mu = k_ud.T @ (K_uu_inv @ self.m_u)
        return mu
    
    def reparameterised_sample(self, num_samples: int, mu: _T, sigma_chol: _T, M: int, N: int):
        deduped_MN = mu.shape[0]
        eps = torch.randn(num_samples, deduped_MN, dtype = torch.float64) # [I, dedup size]
        eps = eps.to(sigma_chol.device)
        model_evals = mu.unsqueeze(0) + (eps @ sigma_chol.T.abs()) # [I, dedup size]
        readded_model_evals = self.reinclude_model_evals(model_evals, M, N, num_samples)
        return readded_model_evals

    def reinclude_model_evals(self, model_evals: _T, M: int, N: int, I: int):
        """
        if self.fix_non_swap:
            Input in shape [I, 1 + M*(N-1)]
        else:
            Input in shape [I, M*(N-1)]
        """
        if self.fix_non_swap:
            zero_f_eval = -float('inf')*torch.ones(I, M, 1, dtype = model_evals.dtype, device = model_evals.device)   # [I, M, 1]
            if N > 1:
                regrouped_model_evals = model_evals.reshape(I, M, N-1)    # [I, M, N-1]
                readded_model_evals = torch.concat([zero_f_eval, regrouped_model_evals], -1)
                return readded_model_evals  # [I, M, N]
            else:
                return zero_f_eval
        else:
            zero_f_eval = model_evals[:,[0]].unsqueeze(1).repeat(1, M, 1)   # [I, M, 1]
            if N > 1:
                regrouped_model_evals = model_evals[:,1:].reshape(I, M, N-1)    # [I, M, N-1]
                readded_model_evals = torch.concat([zero_f_eval, regrouped_model_evals], -1)
                return readded_model_evals  # [I, M, N]
            else:
                return zero_f_eval

    def deduplicate_deltas(self, deltas: _T):
        """
        Every Mth delta will be a 0 by design
        If the non-swapped item (i.e. cued item) is included in the swap function, we remove this to ensure a correct kernel calculation downstream
        If not (i.e. self.fix_non_swap = True), then we don't include it in the deltas
        """
        M, N, D = deltas.shape
        flattened_deltas = deltas.reshape(M * N, D)
        if self.fix_non_swap:
            unique_indices = [i for i in range(M * N) if i % N != 0]
        else:
            unique_indices = [i for i in range(M * N) if i == 0 or i % N != 0]
        dedup_deltas = flattened_deltas[unique_indices]
        return dedup_deltas
        



class FixedCueLocationNonParametricSwapErrorsVariationalModel(NonParametricSwapErrorsVariationalModel):

    """
    Same as above, except the locations of inducing points on the torus are fixed along selected axes.

    Default parameters are for Bays 2009, where location is fixed to 8 points around the circle
    """

    def __init__(self, free_direction_Z_count: int, X: _T = None, num_features: int = 2, fixed_feature_axes = [0], fixed_features_location_counts = [8]):

        super(NonParametricSwapErrorsVariationalModel, self).__init__()

        assert 0 < len(fixed_feature_axes) == len(fixed_features_location_counts) < num_features
        if num_features != 2:
            raise NotImplementedError
        if fixed_feature_axes != [0]:
            raise NotImplementedError('Defining initial_inducing_points_free_direction and self.Z needs reworking in this case!')
        
        self.R = int(free_direction_Z_count * prod(fixed_features_location_counts))

        if X is None:
            initial_inducing_points_fixed_direction = torch.linspace(0, 2*torch.pi, fixed_features_location_counts[0]+1)[:-1]
            initial_inducing_points_free_direction = torch.linspace(0, 2*torch.pi, free_direction_Z_count+1)[:-1]
            grid_x, grid_y = torch.meshgrid(initial_inducing_points_fixed_direction, initial_inducing_points_free_direction, indexing='ij')
            
            torus_points = torch.stack([grid_x, grid_y], -1).reshape(self.R, 2).to(torch.float64)

            free_feature_axes = sorted(list(set(range(num_features)) - set(fixed_feature_axes)))
            fixed_axis_torus_points = torus_points[:,fixed_feature_axes]
            free_axis_torus_points = torus_points[:,free_feature_axes]
        else:
            raise ValueError('Providing inducing points is deprecated')

        self.fixed_axis_torus_points = fixed_axis_torus_points
        self.register_parameter('inducing_points_tilde', nn.Parameter(free_axis_torus_points, requires_grad = True))
        self.register_parameter('m_u', nn.Parameter(torch.zeros(self.R, dtype = torch.float64), requires_grad = True))
        self.register_parameter('S_uu_log_chol', nn.Parameter(torch.zeros(self.R, self.R, dtype = torch.float64), requires_grad = True))


    @property
    def Z(self):
        inducing_points = torch.concat([self.fixed_axis_torus_points, self.inducing_points_tilde], -1)
        return rectify_angles(inducing_points)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.fixed_axis_torus_points = self.fixed_axis_torus_points.to(*args, **kwargs)
        return self
