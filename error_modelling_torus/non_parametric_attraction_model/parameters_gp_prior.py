from abc import ABC, abstractmethod

from typing import List

import torch
from torch import nn
from torch import Tensor as _T

from dataclasses import dataclass

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


@dataclass
class KernelEvaluationInfo:
    K_uu: _T
    K_dd: _T
    k_ud: _T
    K_uu_inv: _T

    def __post_init__(self):
        num_models, num_ind, num_data = self.k_ud.shape
        assert tuple(self.K_uu.shape) == (num_models, num_ind, num_ind)
        assert tuple(self.K_uu_inv.shape) == (num_models, num_ind, num_ind)
        assert tuple(self.K_dd.shape) == (num_models, num_data, num_data)


class AttractionErrorDistributionParametersPrior(nn.Module, ABC):
    """
    We do not inherit from the other project, because there's a bunch of overhead from
    e.g. different set sizes, evaluating PMFs over swaps, sampling from them etc.

    Most inner working is the same though, and making a shared base class is on the books...
    """

    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.num_models = num_models
        self.kernel_noise_sigma = torch.tensor(0.0001).to(torch.float64)
    
    @abstractmethod
    def evaluate_kernel_inner(self, differences_matrix: _T) -> _T:
        raise NotImplementedError

    def evaluate_kernel(self, data_1: _T, data_2: _T = None) -> _T:
        """
        XXX: shared entirely with other project

        data_i comes in shape [Q, B_i]

        This will be given any combination of inducing points and mus (data)

        output of shape [Q, B1, B2]
        output[q,i,j] = k(data_1[i], data_2[j]; self.ells[q])   where q is the model index
        """

        Q, B1 = data_1.shape   # NB: B1 != setsize here!
        assert Q == self.num_models, f"Data passed has shape {data_1.shape} - the first axis should have length num_models ({self.num_models})"

        if data_2 is None:
            sigma = self.kernel_noise_sigma
            private_noise = sigma * torch.eye(B1).to(data_1.device).unsqueeze(0).repeat(Q, 1, 1)    # [Q, B1, B1]
            differences_matrix = rectify_angles(data_1.unsqueeze(2) - data_1.unsqueeze(1))  # [Q,B1,1] - [Q,1,B1] -> [Q,B1,B1]

        else:
            private_noise = 0.0
            Q2, _ = data_2.shape
            assert Q == Q2
            differences_matrix = rectify_angles(data_1.unsqueeze(2) - data_2.unsqueeze(1))  # [Q,B1,1] - [Q,1,B2] -> [Q,B1,B2]

        covariance_term = self.evaluate_kernel_inner(differences_matrix) # [Q,B1,B2 or B1]
        
        total_kernal_eval = covariance_term + private_noise

        return total_kernal_eval

    def generate_kernel_info(self, target_angle_minibatches: List[_T], inducing_locations: _T) -> List[KernelEvaluationInfo]:
        all_kernel_evals = []
        K_uu = self.evaluate_kernel(inducing_locations)
        K_uu_inv = torch.linalg.inv(K_uu)
        for target_angle_minibatch in target_angle_minibatches:
            repeated_target_angle_minibatch = target_angle_minibatch.unsqueeze(0).repeat(self.num_models, 1)
            new_kernel_eval = KernelEvaluationInfo(
                K_uu=K_uu, 
                K_uu_inv=K_uu_inv,
                K_dd=self.evaluate_kernel(repeated_target_angle_minibatch),
                k_ud=self.evaluate_kernel(inducing_locations, repeated_target_angle_minibatch)
            )
            all_kernel_evals.append(new_kernel_eval)
        return all_kernel_evals
    



class NoInputAttractionErrorDistributionParametersPrior(AttractionErrorDistributionParametersPrior):
    """
    Just learns a univariate prior, intended to be used by NoInputSVGPApproximation

    Therefore, the kernel evaluation is just the same number (the prior variance) everywhere
    """

    def __init__(self, num_models: int) -> None:
        super().__init__(num_models)
        log_scaler_raw = (1.0 + torch.relu(2.0 + (0.6 * torch.randn([num_models, 1, 1])))).log().to(torch.float64)
        self.register_parameter('log_scaler', nn.Parameter(log_scaler_raw, requires_grad = True))

        self.kernel_noise_sigma = 0.0

    @property
    def scaler(self):
        return self.log_scaler.exp()

    def evaluate_kernel_inner(self, differences_matrix: _T):
        "Input is [Q, B1, B2], output is [Q, B1, B2]"
        ret = self.scaler * torch.ones_like(differences_matrix)
        for q in range(self.num_models):
            if len(ret[q].unique()) != 1:
                import pdb; pdb.set_trace()
        return ret


class WeilandAttractionErrorDistributionParametersPrior(AttractionErrorDistributionParametersPrior):

    def __init__(self, num_models: int) -> None:
        super().__init__(num_models)

        log_scaler_raw = (1.0 + torch.relu(2.0 + (0.6 * torch.randn([num_models, 1, 1])))).log().to(torch.float64)
        log_inverse_ells_raw = (1.0 + torch.relu(5.0 + (1.5 * torch.randn([num_models, 1, 1])))).log().to(torch.float64)
        
        self.register_parameter('log_inverse_ells', nn.Parameter(log_inverse_ells_raw, requires_grad = True))
        self.register_parameter('log_scaler', nn.Parameter(log_scaler_raw, requires_grad = True))

    @property
    def inverse_ells(self):
        return self.log_inverse_ells.exp()

    @property
    def scaler(self):
        return self.log_scaler.exp()

    def evaluate_kernel_inner(self, differences_matrix: _T):
        "Input is [Q, B1, B2], output is [Q, B1, B2]"
        inverse_ells = self.inverse_ells   # [Q,1,1]
        x = rectify_angles(differences_matrix).abs()                    # [Q,B1,B2]
        weiland_matrix: _T = (1 + inverse_ells * x / torch.pi) * (1.0 - x / torch.pi).relu().pow(inverse_ells)  # [Q,B1,B2]
        return self.scaler * weiland_matrix   # [Q,B1,B2]



