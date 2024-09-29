import torch
from torch import nn
from torch import Tensor as _T

from math import prod
from itertools import product
import matplotlib.pyplot as plt

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from purias_utils.util.plotting import standard_swap_model_simplex_plots, legend_without_repeats, lighten_color

import warnings

from numpy import ndarray as _A

from typing import Optional, List, Union


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
        if self.fix_inducing_point_locations:
            self.inducing_points_tilde = self.inducing_points_tilde.to(*args, **kwargs)
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
    def S_uu(self):         # [Q, R, R]
        if self.inducing_point_variational_parameterisation == 'vanilla':
            return 0.0

        # elif self.symmetricality_constraint:
            
        #     assert self.num_features == 1, "Havne't done this for D>1 yet!! needs complete rechanging"

        #     L_a = torch.tril(self.S_uu_L_a_log)
        #     L_a[range(len(L_a)),range(len(L_a))] = L_a[range(len(L_a)),range(len(L_a))].exp()
        #     L_a_inv = torch.linalg.inv(L_a)

        #     A = L_a @ L_a.T
        #     A_inv = torch.linalg.inv(A)
        #     B = self.S_uu_B
        #     S = A - B @ A_inv @ B.T
        #     try:
        #         L_s = torch.linalg.cholesky(S)
        #     except:
        #         import pdb; pdb.set_trace()

        #     chol = torch.zeros(self.R, self.R, dtype = L_s.dtype, device = L_s.device)
            
        #     bs = self.R // len(self.all_quadrant_mults) # block size
        #     chol[:bs,:bs] = L_a
        #     chol[bs:,bs:] = L_s
        #     chol[bs:,:bs] = B @ L_a_inv.T

        else:
            chol = torch.tril(self.S_uu_log_chol)
            diag_index = range(self.R)
            chol[:, diag_index, diag_index] = chol[:,diag_index, diag_index].exp()           # [Q, R, R]


        S_uu = torch.bmm(chol, chol.transpose(1, 2))

        return S_uu
    
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
        S_uu = self.S_uu                                                # [Q, R, R]
        assert K_uu.shape == S_uu.shape == K_uu_inv.shape
        det_S_uu: _T = torch.linalg.det(S_uu)                           # [Q]
        det_K_uu: _T = torch.linalg.det(K_uu)                           # [Q]
        det_term = (det_K_uu / det_S_uu).log()                          # [Q]
        trace_term = torch.diagonal(torch.bmm(K_uu_inv, S_uu), offset=0, dim1=-1, dim2=-2).sum(-1)  # [Q]
        mu_term = self.batched_inner_product(K_uu_inv, self.m_u)        # [Q]
        kl_term = 0.5 * (det_term + trace_term + mu_term - self.R)      # [Q]
        if kl_term.isnan().any():
            import pdb; pdb.set_trace()
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
    
    def reparameterised_sample(self, num_samples: int, mu: _T, sigma_chol: _T, M: int, N: int):
        """
            mu and sigma_chol come from variational_gp_inference: [Q, MN] and [Q, MN, MN]

            return is of shape [Q, I, M, N]
        """
        deduped_MN = mu.shape[1]
        eps = torch.randn(self.num_models, num_samples, deduped_MN, dtype = mu.dtype, device = mu.device) # [Q, I, MN (dedup size)]
        model_evals = mu.unsqueeze(1) + torch.bmm(eps, sigma_chol.transpose(-1, -2))   # [Q, I, MN]
        readded_model_evals = self.reinclude_model_evals(model_evals, M, N, num_samples)    # [Q, I, M, N]
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





        std_est = flat_sigma_est.diag().sqrt().cpu()
        flat_mu_est_numpy = flat_mu_est.cpu()

        upper_error_surface = (flat_mu_est_numpy + (2*std_est)).reshape(grid_x.shape)
        lower_error_surface = (flat_mu_est_numpy - (2*std_est)).reshape(grid_x.shape)
        surface = flat_mu_est_numpy.reshape(grid_x.shape)
        
        return flat_mu_est, flat_mu_est_numpy, upper_error_surface, lower_error_surface, full_grid, surface, grid_x.numpy(), grid_y.numpy(), sigma_chol



    def visualise_approximation(
        self, one_dimensional_grid: _A, all_grid_points: _A, mean_surface: _A, std_surface: _A, function_samples_on_grid: _A,
        pi_u_tildes: _A, pi_1_tildes: _A, all_deltas: _A, recent_component_priors: Optional[_A], true_mean_surface: Optional[_A], true_std_surface: Optional[_A],
        min_separation: float, max_separation: float, deltas_label: str
    ):
        """
        Input:
            All taken from util.inference_on_grid

        If D = 1:
            TODO: list everything!

        If D = 2:
            TODO: list everything!

        TODO: shapes!
        """

        if self.num_features > 1:
            raise NotImplementedError

        else:
            Q = self.num_models
            figsize = 8
            fig_surfaces = plt.figure(figsize = (figsize * 4, figsize * (Q+1)))

            # axes_kernel = fig_surfaces.add_subplot(Q+1,4,2*Q)
            axes_hist = fig_surfaces.add_subplot(Q+1,4,4*Q+1)
            axes_hist.hist(all_deltas[:,1:].flatten(), 1024, density=True)
            axes_hist.set_xlabel(deltas_label)

            all_inducing_points = self.Z.detach().cpu().squeeze(-1).numpy()
            all_inducing_points_means = self.m_u.detach().cpu().numpy()
            if self.inducing_point_variational_parameterisation == 'gaussian':
                all_inducing_points_covars = self.S_uu.detach().cpu().numpy()

            for q in range(Q):
                axes1D_linear = fig_surfaces.add_subplot(Q+1,4,q*4+1)
                # axes1D_exponentiated = fig_surfaces.add_subplot(2,3,2)
                axes_Suu = fig_surfaces.add_subplot(Q+1,4,q*4+2)
                axes_simplex = fig_surfaces.add_subplot(Q+1,4,q*4+3)
                axes_simplex_no_u = fig_surfaces.add_subplot(Q+1,4,q*4+4)

                surface_color = axes1D_linear.plot(one_dimensional_grid, mean_surface[q], color = 'blue')[0].get_color()
                lower_error_surface, upper_error_surface = mean_surface[q] - 2 * std_surface[q], mean_surface[q] + 2 * std_surface[q]
                axes1D_linear.fill_between(one_dimensional_grid, lower_error_surface, upper_error_surface, color = surface_color, alpha = 0.2)

                sample_colour = lighten_color(surface_color, 1.6)
                for sample_on_grid in function_samples_on_grid[q]:
                    axes1D_linear.plot(one_dimensional_grid, sample_on_grid, color = sample_colour, alpha = 0.4)

                axes1D_linear.scatter(all_inducing_points[q], all_inducing_points_means[q], color = 'black', marker = 'o', s = 20)
                axes1D_linear.plot([-torch.pi, torch.pi], [pi_u_tildes[q].item(), pi_u_tildes[q].item()], surface_color, linestyle= '-.', linewidth = 3)
                axes1D_linear.plot([-torch.pi, torch.pi], [pi_1_tildes[q].item(), pi_1_tildes[q].item()], surface_color, linewidth = 3)

                if true_mean_surface is not None:
                    flattened_true_mean = true_mean_surface[q].flatten()
                    axes1D_linear.scatter(all_deltas.flatten(), flattened_true_mean, color = 'red', alpha = 0.4, s = 5)
                    if true_std_surface is not None:
                        flattened_true_std = true_std_surface[q].flatten()
                        axes1D_linear.scatter(all_deltas.flatten(), flattened_true_mean + 2 * flattened_true_std, color = 'red', alpha = 0.01, s = 5)
                        axes1D_linear.scatter(all_deltas.flatten(), flattened_true_mean - 2 * flattened_true_std, color = 'red', alpha = 0.01, s = 5)
                
                if self.inducing_point_variational_parameterisation == 'gaussian':
                    axes_Suu.imshow(all_inducing_points_covars[q], cmap = 'gray')

                for sep in [min_separation, max_separation]:
                    y_bot, y_top = axes1D_linear.get_ylim()
                    axes1D_linear.plot([sep, sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes1D_linear.plot([-sep, -sep], [y_bot, y_top], color = 'black', linestyle = '--')
                    axes1D_linear.set_ylim(y_bot, y_top)
                    axes1D_linear.set_xlim(-torch.pi, torch.pi)

                if recent_component_priors is not None:
                    standard_swap_model_simplex_plots(recent_component_priors[q], axes_simplex, ax_no_u = axes_simplex_no_u)
                    legend_without_repeats(axes_simplex)
                    legend_without_repeats(axes_simplex_no_u)

            axes1D_linear.set_xlabel(deltas_label)

            return fig_surfaces
