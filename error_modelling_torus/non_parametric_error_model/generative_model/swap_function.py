import torch
from torch import nn
from torch import Tensor as _T

from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model.helpers import KernelParameterHolder, LogitsHolder, PiTildeHolder

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles

GROUPING_THRES = 1e-3


class SwapFunctionBase(nn.Module):

    def __init__(self, function_set_sizes: list, remove_uniform: bool, include_pi_u_tilde: bool) -> None:
        super().__init__()

        self.function_set_sizes = function_set_sizes
        self.remove_uniform = remove_uniform
        self.include_pi_u_tilde = include_pi_u_tilde

        if include_pi_u_tilde:
            assert not self.remove_uniform, "Cannot remove uniform (remove_uniform) while specifying a learnable uniform pre-softmax (include_pi_u_tilde)!"

        # if include_pi_u_tilde:
        #     pi_u_tilde_raw = ((0.10 * torch.randn(1)) + 0.15).to(torch.float64)
        #     self.register_parameter('pi_u_tilde', nn.Parameter(pi_u_tilde_raw, requires_grad = True))
        # else:
        #     self.pi_u_tilde = 0.0

    # def pi_u_tilde_vector(self, I: int, M: int):
    #     return self.pi_u_tilde * torch.ones([I, M, 1]).to(self.pi_u_tilde.device)

    def sample_betas(self, pi_vectors: _T):
        """
            pi_vectors of shape [I, M, N+1]
        """
        prob_cum = pi_vectors.cumsum(-1)
        u = torch.rand(*prob_cum.shape[:2], 1).to(pi_vectors.device)
        selected_components = (u > prob_cum).sum(-1)
        return selected_components                              # [I, M]

    def generate_pi_vectors(self, set_size: int, return_exp_grid: bool):
        raise NotImplementedError

    def evaluate_kernel(self, set_size: int, data_1: _T, data_2: _T = None):
        raise NotImplementedError

    def evaluate_kernel_inner(self, differences_matrix: _T) -> _T:
        raise NotImplementedError

    def generate_exp_pi_u_tilde(self, set_size, I: int, M: int, dtype, device):
        if self.remove_uniform:
            exp_pi_u_tilde = torch.zeros(I, M, 1).to(dtype=dtype, device=device)
        elif self.include_pi_u_tilde:
            pi_u_tilde = self.pi_u_tilde_holder[str(set_size)].pi_tilde.to(device) * torch.ones([I, M, 1]).to(dtype=dtype, device=device)
            exp_pi_u_tilde = pi_u_tilde.exp()
        else:
            exp_pi_u_tilde = torch.ones(I, M, 1).to(dtype=dtype, device=device)     # [I, M, 1]
        return exp_pi_u_tilde


class NonParametricSwapFunctionBase(SwapFunctionBase):
    """
    Original delta calculation, where cued item is 'removed' without noise!
    """

    def __init__(self, num_features: int, kernel_set_sizes: list, remove_uniform: bool, include_pi_u_tilde: bool, fix_non_swap: bool, include_pi_1_tilde: bool) -> None:
        super().__init__(kernel_set_sizes, remove_uniform, include_pi_u_tilde)

        self.fix_non_swap = fix_non_swap
        self.include_pi_1_tilde = include_pi_1_tilde
        self.num_features = num_features

        if include_pi_u_tilde:
            self.pi_u_tilde_holder = (
                PiTildeHolder() if kernel_set_sizes is None 
                else nn.ModuleDict({str(N): PiTildeHolder(0.0) for N in kernel_set_sizes})
            )

        if include_pi_1_tilde:
            self.pi_1_tilde_holder = (
                PiTildeHolder() if kernel_set_sizes is None 
                else nn.ModuleDict({str(N): PiTildeHolder(1.0) for N in kernel_set_sizes})
            )

    def generate_exp_pi_1_tilde(self, set_size, I: int, M: int, dtype, device):
        assert self.fix_non_swap
        if self.include_pi_1_tilde:
            pi_1_tilde = self.pi_1_tilde_holder[str(set_size)].pi_tilde.to(device) * torch.ones([I, M, 1]).to(dtype=dtype, device=device)
            exp_pi_1_tilde = pi_1_tilde.exp()
        else:
            exp_pi_1_tilde = torch.ones(I, M, 1).to(dtype=dtype, device=device).exp()
        return exp_pi_1_tilde
    
    def generate_pi_vectors(self, set_size: int, model_evaulations: _T, return_exp_grid = False):
        """
        model_evaulations: (samples of) f, shaped [I, M, N]

        output of shape [I, M, N+1], where output[i,m,0] is the relevant pi_u probability
        """
        I, M, N = model_evaulations.shape
        exp_pi_u_tilde = self.generate_exp_pi_u_tilde(set_size, I, M, model_evaulations.dtype, model_evaulations.device)
        exp_grid = torch.concat([exp_pi_u_tilde, model_evaulations.exp()], dim=-1)          # [I, M, N+1]
        if self.fix_non_swap:
            assert (exp_grid[:,:,1] == 0.0).all(), "To learn pi_1_tilde, swap variational model cannot generate it!"
            exp_grid[:,:,[1]] = self.generate_exp_pi_1_tilde(set_size, I, M, model_evaulations.dtype, model_evaulations.device)
        denominator = exp_grid.sum(-1, keepdim=True)                                        # [I, M, 1]
        pis = exp_grid / denominator
        return (pis, exp_grid) if return_exp_grid else pis

    def evaluate_kernel(self, set_size: int, data_1: _T, data_2: _T = None):
        """
        This will be given any combination of inducing points and deltas (data)
        output[i,j] = k(data_1[i], data_2[j]; self.ells)

        To alleviate some numerical problems, we also group together items that are really similar to each other,
            in the case that data_2 is None, i.e. data with itself

            The actual kernel matrix evaluated will be on a subset of the data (size N' < N) with near duplicates removed
            grouping_idxs is of size N, and indexes which item (0,1,2,...,N'-1) should be unpacked from that item.

            At downstream sampling time, i.e. at NonParametricSwapErrorsVariationalModel.reparameterised_sample, these can be unpacked
        """

        N1, D1 = data_1.shape   # NB: N1 != setsize here!

        if data_2 is None:
            sigma = self.kernel_holder[str(set_size)].kernel_noise_sigma
            private_noise = sigma * torch.eye(N1).to(data_1.device)
            differences_matrix = rectify_angles(data_1.unsqueeze(1) - data_1)  # [N1,1,D] - [N2,D] -> [N1,N2,D]

            if (D1 == 1):
                #import pdb; pdb.set_trace()
                #x = differences_matrix[...,0].triu().abs()
                #similar_pairs = torch.logical_and(0.0 < x, x<=GROUPING_THRES).argwhere()    # [(kept item, )]
                pass

            else:
                pass
                #import pdb; pdb.set_trace(header = 'perform grouping!')

        else:
            private_noise = 0.0
            _, D2 = data_2.shape
            assert D1 == D2 == self.num_features
            differences_matrix = rectify_angles(data_1.unsqueeze(1) - data_2)  # [N1,1,D] - [N2,D] -> [N1,N2,D]

        covariance_term = self.evaluate_kernel_inner(set_size, differences_matrix) # [N1,N2]
        
        total_kernal_eval = covariance_term + private_noise

        return total_kernal_eval# , grouping_idxs


class NonParametricSwapFunctionExpCos(NonParametricSwapFunctionBase):

    def __init__(self, num_features: int, kernel_set_sizes: list, trainable_kernel_delta: bool, remove_uniform: bool, include_pi_u_tilde: bool, fix_non_swap: bool, include_pi_1_tilde: bool) -> None:
        super().__init__(num_features, kernel_set_sizes, remove_uniform, include_pi_u_tilde, fix_non_swap, include_pi_1_tilde)

        self.kernel_holder = (
            KernelParameterHolder(self.num_features, trainable_kernel_delta) if kernel_set_sizes is None 
            else nn.ModuleDict({str(N): KernelParameterHolder(self.num_features, trainable_kernel_delta) for N in kernel_set_sizes})
        )

    def evaluate_kernel_inner(self, set_size: int, differences_matrix: _T):
        inverse_ells: _T = self.kernel_holder[str(set_size)].inverse_ells
        exp_cos_matrix = ((differences_matrix).cos() * inverse_ells).exp()   # [N1,N2,D]
        scaled_exp_cos_matrix = (exp_cos_matrix - (-inverse_ells).exp()) / (inverse_ells.exp() - (-inverse_ells).exp())
        scaled_exp_cos_matrix_total = scaled_exp_cos_matrix.prod(-1)
        return self.kernel_holder[str(set_size)].scaler * scaled_exp_cos_matrix_total # [N1,N2]



class NonParametricSwapFunctionWeiland(NonParametricSwapFunctionExpCos):

    def evaluate_kernel_inner(self, set_size: int, differences_matrix: _T):
        inverse_ells = self.kernel_holder[str(set_size)].inverse_ells
        x = rectify_angles(differences_matrix).abs() 
        weiland_matrix: _T = (1 + inverse_ells * x / torch.pi) * (1.0 - x / torch.pi).relu().pow(inverse_ells)  # [N1,N2,D]
        return self.kernel_holder[str(set_size)].scaler * weiland_matrix.prod(-1)



class SpikeAndSlabSwapFunction(SwapFunctionBase):

    def __init__(self, logits_set_sizes: list, remove_uniform: bool) -> None:
        
        self.remove_uniform = remove_uniform

        super().__init__(logits_set_sizes, remove_uniform, include_pi_u_tilde = False)

        self.logit_holder = (
            LogitsHolder() if logits_set_sizes is None 
            else nn.ModuleDict({str(N): LogitsHolder(N) for N in logits_set_sizes})
        )

    def generate_pi_vectors(self, set_size: int, batch_size: int, return_exp_grid = False):
        exp_pi_u_tilde = self.generate_exp_pi_u_tilde(set_size, 1, 1, None, None)
        component_exp_grid = self.logit_holder[str(set_size)].logit_vector(set_size)    # [1, 1, N]
        exp_grid = torch.concat([exp_pi_u_tilde.to(component_exp_grid.device), component_exp_grid.exp()], dim=-1)
        exp_grid = exp_grid.repeat(1, batch_size, 1)    # [1, M, N+1]
        denominator = exp_grid.sum(-1, keepdim=True)    # [1, M, 1]
        pis = exp_grid / denominator
        return (pis, exp_grid) if return_exp_grid else pis
