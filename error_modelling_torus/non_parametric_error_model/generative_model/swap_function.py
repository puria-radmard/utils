import torch
from torch import nn
from torch import Tensor as _T

import numpy as np
from numpy import ndarray as _A

from typing import Dict, Union

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.helpers import KernelParameterHolder, PiTildeHolder, reduce_to_single_model

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

from abc import abstractmethod, ABC
from math import log as mathlog

GROUPING_THRES = 1e-3


class SwapFunctionBase(nn.Module, ABC):

    def __init__(self, num_models: int, function_set_sizes: list, remove_uniform: bool, include_pi_u_tilde: bool, normalisation_inner: str) -> None:
        super().__init__()

        self.num_models = num_models
        self.function_set_sizes = function_set_sizes
        self.remove_uniform = remove_uniform
        self.include_pi_u_tilde = include_pi_u_tilde
        self.normalisation_inner_function = normalisation_inner

        if include_pi_u_tilde:
            assert not self.remove_uniform, "Cannot remove uniform (remove_uniform) while specifying a learnable uniform pre-softmax (include_pi_u_tilde)!"

    @abstractmethod
    def reduce_to_single_model(self, model_index: int = 0) -> None:
        raise NotImplementedError

    def normalisation_inner(self, x: _T) -> _T:
        if self.normalisation_inner_function == 'exp':
            return x.exp()
        elif self.normalisation_inner_function == 'softplus':
            t = -1.0
            return (1. + (x - t).exp()).log()

    def normalisation_inner_numpy(self, x: _A) -> _A:
        if self.normalisation_inner_function == 'exp':
            return np.exp(x)
        elif self.normalisation_inner_function == 'softplus':
            t = -1.0
            return np.log(1. + np.exp(x - t))

    # def normalisation_inner_inverse_numpy(self, x: Union[float, _A]) -> Union[float, _A]:
    #     if isinstance(x, _A):
    #         zero_mask = (x == 0)
    #         x[zero_mask] = 1.0  # Avoid warnings!
    #     if self.normalisation_inner_function == 'exp':
    #         inverse = np.log(x)
    #     elif self.normalisation_inner_function == 'softplus':
    #         t = -1.0
    #         inverse = t + np.log(np.exp(x) - 1.0)
    #     if isinstance(x, _A):
    #         x[zero_mask] = 0.0
    #     return inverse

    def sample_betas(self, pi_vectors: _T):
        """
            pi_vectors of shape [Q, M, N+1]
            output [Q, M]
        """
        assert pi_vectors.shape[0] == self.num_models
        prob_cum = pi_vectors.cumsum(-1)    # [Q, M, N+1]
        u = torch.rand(*prob_cum.shape[:2], 1).to(pi_vectors.device)     # [Q, M, 1]
        selected_components = (u > prob_cum).sum(-1)                    # [Q, M, N+1] -> [Q, M]
        return selected_components                              # [Q, M]

    @abstractmethod
    def generate_pi_vectors(self, set_size: int) -> Dict[str, _T]:    # [Q, M, N+1]
        raise NotImplementedError

    @abstractmethod
    def evaluate_kernel(self, set_size: int, data_1: _T, data_2: _T = None) -> _T:
        raise NotImplementedError

    @abstractmethod
    def evaluate_kernel_inner(self, differences_matrix: _T) -> _T:
        raise NotImplementedError

    @abstractmethod
    def sample_from_prior(self, set_size: _T, eval_points: _T, num_samples: int) -> Dict[str, _T]:
        raise NotImplementedError

    def generate_exp_pi_u_tilde(self, set_size, I: int, M: int, dtype, device): # [Q, I, M, 1]
        Q = self.num_models
        if self.remove_uniform:
            exp_pi_u_tilde = torch.zeros(Q, I, M, 1).to(dtype=dtype, device=device)
        elif self.include_pi_u_tilde:
            pi_u_tilde = self.pi_u_tilde_holder[str(set_size)].pi_tilde.to(device).reshape(-1, 1, 1, 1).repeat(1, I, M, 1)
            exp_pi_u_tilde = self.normalisation_inner(pi_u_tilde)
        else:
            exp_pi_u_tilde = self.normalisation_inner(torch.zeros(Q, I, M, 1).to(dtype=dtype, device=device))
        return exp_pi_u_tilde

    def generate_exp_pi_1_tilde(self, set_size, I: int, M: int, dtype, device): # [Q, I, M, 1]
        assert self.fix_non_swap
        Q = self.num_models
        if self.include_pi_1_tilde:
            pi_1_tilde = self.pi_1_tilde_holder[str(set_size)].pi_tilde.to(device).reshape(-1, 1, 1, 1).repeat(1, I, M, 1)
            exp_pi_1_tilde = self.normalisation_inner(pi_1_tilde)
        else:
            exp_pi_1_tilde = self.normalisation_inner(torch.ones(Q, I, M, 1).to(dtype=dtype, device=device))
        return exp_pi_1_tilde


class NonParametricSwapFunctionBase(SwapFunctionBase):
    """
    Original delta calculation, where cued item is 'removed' without noise!
    """

    kernel_holder: Union[KernelParameterHolder, nn.ModuleDict]

    def __init__(self, num_models: int, num_features: int, kernel_set_sizes: list, remove_uniform: bool, include_pi_u_tilde: bool, fix_non_swap: bool, include_pi_1_tilde: bool, normalisation_inner: bool) -> None:
        super().__init__(num_models, kernel_set_sizes, remove_uniform, include_pi_u_tilde, normalisation_inner)

        self.fix_non_swap = fix_non_swap
        self.include_pi_1_tilde = include_pi_1_tilde
        self.num_features = num_features

        if include_pi_u_tilde:
            self.pi_u_tilde_holder = (
                PiTildeHolder(0.0, num_models) if kernel_set_sizes is None 
                else nn.ModuleDict({str(N): PiTildeHolder(0.0, num_models) for N in kernel_set_sizes})
            )

        if include_pi_1_tilde:
            self.pi_1_tilde_holder = (
                PiTildeHolder(2.0, num_models) if kernel_set_sizes is None 
                else nn.ModuleDict({str(N): PiTildeHolder(2.0, num_models) for N in kernel_set_sizes})
            )

    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.num_models = 1
        reduce_to_single_model(self.kernel_holder, model_index)
        if self.include_pi_u_tilde:
            reduce_to_single_model(self.pi_u_tilde_holder, model_index)
        if self.include_pi_1_tilde:
            reduce_to_single_model(self.pi_1_tilde_holder, model_index)
    
    def generate_pi_vectors(self, set_size: int, model_evaulations: _T, make_spike_and_slab = False, mc_average = True) -> Dict[str, _T]:
        """
        model_evaulations: (samples of) f, shaped [Q, I, M, N] --  see NonParametricSwapErrorsVariationalModel.reparameterised_sample

        output of shape [Q, M, N+1], where output[q,m,0] is the relevant pi_u probability for qth model being trained
            OR [Q, I or K, M, N+1] if choosing not to perform the MC average
        """
        Q, I, M, N = model_evaulations.shape
        assert Q == self.num_models
        exp_pi_u_tilde = self.generate_exp_pi_u_tilde(set_size, I, M, model_evaulations.dtype, model_evaulations.device)    # [Q, I, M, 1]
        exp_grid = torch.concat([exp_pi_u_tilde, model_evaulations.exp()], dim=-1)          # [Q, I, M, N+1]
        if self.fix_non_swap:
            assert (exp_grid[...,1] == 0.0).all(), "To learn pi_1_tilde, swap variational model cannot generate it!"
            exp_grid[...,[1]] = self.generate_exp_pi_1_tilde(set_size, I, M, model_evaulations.dtype, model_evaulations.device)
        pis = exp_grid / exp_grid.sum(-1, keepdim=True)                                        # [Q, I, M, 1]
        if make_spike_and_slab:
            print("Flattening swap function to spike and slab! Only recommended if whole dataset is being passed!")
            pis = pis.mean(2, keepdim=True) # [Q, I, 1, N+1]
            pis[...,2:] = pis[...,2:].mean(keepdim=True)    # Replace [Q, I, 1, N-1]
            pis = pis.repeat(1, 1, M, 1)
            exp_grid = None # Not determined here...
        if mc_average:
            pis = pis.mean(1)   # Average over MC draws
        return {'pis': pis, 'exp_grid': exp_grid}

    def evaluate_kernel(self, set_size: int, data_1: _T, data_2: _T = None) -> _T:
        """
        data_i comes in shape [Q, N_i, D_i]

        This will be given any combination of inducing points and deltas (data)

        output of shape [Q, N1, N2]
        output[q,i,j] = k(data_1[i], data_2[j]; self.ells[q])   where q is the model index
        """

        Q, N1, D1 = data_1.shape   # NB: N1 != setsize here!
        assert Q == self.num_models, f"Data passed has shape {data_1.shape} - the first axis should have length num_models ({self.num_models})"

        if data_2 is None:
            sigma = self.kernel_holder[str(set_size)].kernel_noise_sigma    # 
            private_noise = sigma * torch.eye(N1).to(data_1.device).unsqueeze(0).repeat(Q, 1, 1)    # [Q, N1, N1]
            differences_matrix = rectify_angles(data_1.unsqueeze(2) - data_1.unsqueeze(1))  # [Q,N1,1,D] - [Q,1,N1,D] -> [Q,N1,N1,D]

        else:
            private_noise = 0.0
            Q2, _, D2 = data_2.shape
            assert D1 == D2 == self.num_features and Q == Q2
            differences_matrix = rectify_angles(data_1.unsqueeze(2) - data_2.unsqueeze(1))  # [Q,N1,1,D] - [Q,1,N2,D] -> [Q,N1,N2,D]

        covariance_term = self.evaluate_kernel_inner(set_size, differences_matrix) # [Q,N1,N2 or N1]
        
        total_kernal_eval = covariance_term + private_noise

        return total_kernal_eval

    def sample_from_prior(self, set_size, eval_points, num_samples, existing_samples = None):
        """
        Sample K times from prior at the points provided, and return log-likleihoods, as well as kernel matrix at those points
        """
        K_uu = self.evaluate_kernel(set_size, eval_points)  # [Q, R, R], eval points size checked here!
        K_uu_chol = torch.cholesky(K_uu)    # [Q, R, R]
        R = K_uu_chol.shape[-1]
        if existing_samples is None:
            eps = torch.randn(self.num_models, num_samples, R, dtype = K_uu_chol.dtype, device = K_uu_chol.device) # [Q, K, R]
            samples = torch.bmm(eps, K_uu_chol.transpose(-1, -2))
        else:
            assert tuple(samples.shape) == (self.num_models, num_samples, eval_points.shape[1])
        original_lhs = mathlog((2 * torch.pi)**(-R / 2.)) + (-0.5 * (eps * eps).sum(-1))                        # [Q, K]
        cholesky_log_determinant = K_uu_chol.diagonal(offset = 0, dim1 = -1, dim2 = -2).log().sum(-1,keepdim=True)        # [Q, 1]
        new_llhs = original_lhs - cholesky_log_determinant
        return {
            'samples': samples, # [Q, K, R]
            'sample_log_likelihoods': new_llhs,   # [Q, K]
        }


class NonParametricSwapFunctionExpCos(NonParametricSwapFunctionBase):

    def __init__(self, num_models: int, num_features: int, kernel_set_sizes: list, trainable_kernel_delta: bool, remove_uniform: bool, include_pi_u_tilde: bool, fix_non_swap: bool, include_pi_1_tilde: bool, normalisation_inner: str) -> None:
        super().__init__(num_models, num_features, kernel_set_sizes, remove_uniform, include_pi_u_tilde, fix_non_swap, include_pi_1_tilde, normalisation_inner)

        self.kernel_holder = (
            KernelParameterHolder(num_models, self.num_features, trainable_kernel_delta) if kernel_set_sizes is None 
            else nn.ModuleDict({str(N): KernelParameterHolder(num_models, self.num_features, trainable_kernel_delta) for N in kernel_set_sizes})
        )

    def evaluate_kernel_inner(self, set_size: int, differences_matrix: _T):
        "Input is [Q, N1, N2, D], output is [Q, N1, N2]"
        inverse_ells: _T = self.kernel_holder[str(set_size)].inverse_ells       # [Q,1,1,D]
        exp_cos_matrix = ((differences_matrix).cos() * inverse_ells).exp()      # [Q,N1,N2,D]
        scaled_exp_cos_matrix = (exp_cos_matrix - (-inverse_ells).exp()) / (inverse_ells.exp() - (-inverse_ells).exp())
        scaled_exp_cos_matrix_total = scaled_exp_cos_matrix.sum(-1)
        return self.kernel_holder[str(set_size)].scaler * scaled_exp_cos_matrix_total # [Q,N1,N2]


class NonParametricSwapFunctionWeiland(NonParametricSwapFunctionExpCos):

    def evaluate_kernel_inner(self, set_size: int, differences_matrix: _T):
        "Input is [Q, N1, N2, D], output is [Q, N1, N2]"
        inverse_ells = self.kernel_holder[str(set_size)].inverse_ells   # [Q,1,1,D]
        x = rectify_angles(differences_matrix).abs()                    # [Q,N1,N2,D]
        weiland_matrix: _T = (1 + inverse_ells * x / torch.pi) * (1.0 - x / torch.pi).relu().pow(inverse_ells)  # [Q,N1,N2,D]
        return self.kernel_holder[str(set_size)].scaler * weiland_matrix.prod(-1)   # [Q,N1,N2]


class SpikeAndSlabSwapFunction(SwapFunctionBase):

    def __init__(self, num_models: int, logits_set_sizes: list, remove_uniform: bool, include_pi_u_tilde: bool, include_pi_1_tilde: bool, normalisation_inner: str) -> None:
        
        super().__init__(num_models, logits_set_sizes, remove_uniform, include_pi_u_tilde = include_pi_u_tilde, normalisation_inner = normalisation_inner)
        if not remove_uniform:
            assert include_pi_u_tilde or include_pi_1_tilde, "Cannot fix both pi_u_tilde and pi_1_tilde in spike and slab model!"

        self.fix_non_swap = True    # of course
        self.include_pi_1_tilde = include_pi_1_tilde

        self.pi_swap_tilde_holder = (
            PiTildeHolder(1.0, num_models) if logits_set_sizes is None 
            else nn.ModuleDict({str(N): PiTildeHolder(1.0, num_models) for N in logits_set_sizes})
        )
        if logits_set_sizes is not None and 1 in logits_set_sizes:
            self.pi_swap_tilde_holder.pop('1')

        if include_pi_u_tilde:
            self.pi_u_tilde_holder = (
                PiTildeHolder(0.0, num_models) if logits_set_sizes is None 
                else nn.ModuleDict({str(N): PiTildeHolder(0.0, num_models) for N in logits_set_sizes})
            )

        if include_pi_1_tilde:
            self.pi_1_tilde_holder = (
                PiTildeHolder(2.0, num_models) if logits_set_sizes is None 
                else nn.ModuleDict({str(N): PiTildeHolder(2.0, num_models) for N in logits_set_sizes})
            )

    def sample_from_prior(self, set_size: _T, eval_points: _T, num_samples: int) -> Dict[str, _T]:
        raise TypeError('SpikeAndSlabSwapFunction has no kernel!')

    def evaluate_kernel(self, set_size: int, data_1: _T, data_2: _T = None) -> _T:
        raise TypeError('SpikeAndSlabSwapFunction has no kernel!')

    def evaluate_kernel_inner(self, differences_matrix: _T) -> _T:
        raise TypeError('SpikeAndSlabSwapFunction has no kernel!')

    def reduce_to_single_model(self, model_index: int = 0) -> None:
        self.num_models = 1
        reduce_to_single_model(self.pi_swap_tilde_holder, model_index)
        if self.include_pi_u_tilde:
            reduce_to_single_model(self.pi_u_tilde_holder, model_index)
        if self.include_pi_1_tilde:
            reduce_to_single_model(self.pi_1_tilde_holder, model_index)

    def generate_exp_pi_swap_tilde(self, set_size, I: int, M: int, dtype, device): # [Q, I, M, N-1]
        if set_size == 1:
            pi_swap_tilde = torch.empty(self.num_models, I, M, 0)
        else:
            pi_swap_tilde = self.pi_swap_tilde_holder[str(set_size)].pi_tilde.reshape(-1, 1, 1, 1).repeat(1, I, M, set_size - 1)
        exp_pi_swap_tilde = self.normalisation_inner(pi_swap_tilde)
        return exp_pi_swap_tilde.to(device=device, dtype=dtype)

    def generate_pi_vectors(self, set_size: int, batch_size: int, device = 'cuda') -> Dict[str, _T]:
        """
        Output is of shape [Q, M, N+1], where output[q,0,:,n] is the same for all data
        """
        exp_pi_swap_tilde = self.generate_exp_pi_swap_tilde(set_size, 1, 1, device=device, dtype=None)   # [Q, 1, M, N-1]
        exp_pi_u_tilde = self.generate_exp_pi_u_tilde(set_size, 1, 1, device=exp_pi_swap_tilde.device, dtype=exp_pi_swap_tilde.dtype)   # [Q, 1, M, 1]
        exp_pi_1_tilde = self.generate_exp_pi_1_tilde(set_size, 1, 1, device=exp_pi_swap_tilde.device, dtype=exp_pi_swap_tilde.dtype)   # [Q, 1, M, 1]
        exp_grid = torch.concat([exp_pi_u_tilde, exp_pi_1_tilde, exp_pi_swap_tilde], dim = -1).squeeze(1)  # [Q, 1, 1, N+1] -> [Q, 1, N+1]
        exp_grid = exp_grid.repeat(1, batch_size, 1)                                         # [Q, M, N+1]
        denominator = exp_grid.sum(-1, keepdim=True)                                            # [Q, M, 1]
        pis = exp_grid / denominator
        return {'pis': pis, 'exp_grid': exp_grid}

