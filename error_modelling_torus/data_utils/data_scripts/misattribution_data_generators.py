raise Exception('Deprecated/need to update')

import torch
import numpy as np
from scipy.io import loadmat

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import MultipleSetSizesActivitySetDataGeneratorEnvelopeBase

from purias_utils.util.error_modelling import zero_mean_von_mises_log_prob_with_kappa


class TargetFunctionDataGeneratorBase(EstimateDataLoaderBase, ABC):

    all_deltas: _T
    all_target_zetas: _T

    def __init__(self, N, torus_dimensionality, target_concentration, underlying_data_generator, output_layer = 'error', device = 'cuda') -> None:

        self.device = device
        self.output_layer = output_layer

        self.data_generator = NonParametricSwapErrorsGenerativeModel(torus_dimensionality, [N], [N], False, False).to(self.device)
        self.data_generator.concentration_holder[str(N)].log_concentration.data = torch.tensor(target_concentration).log().to(torch.float64).to(self.device)
        self.data_generator.concentration_holder[str(N)].log_concentration.requires_grad = False
        del self.data_generator.kernel_holder

        self.steal_M_bullshit_from_another_generator(underlying_data_generator)

        self.all_deltas = torch.tensor(underlying_data_generator.all_deltas).to(device)                    # [M, N, D (2)]
        self.all_target_zetas = torch.tensor(underlying_data_generator.all_target_zetas).to(device)                    # [M, N, D (2)]

        test_deltas = self.all_deltas[self.test_indices]
        test_zeta_t = self.all_target_zetas[self.test_indices]
        self.test_outputs = self.generate_from_batch(N, test_deltas, test_zeta_t)              # [M, N]

        train_deltas = self.all_deltas[self.train_indices]
        train_zeta_t = self.all_target_zetas[self.train_indices]
        self.train_outputs = self.generate_from_batch(N, train_deltas, train_zeta_t)

        self.set_size_to_M_train_each = {self.all_deltas.shape[1]: self.M_train}

    @abstractmethod
    def target_function(self, deltas: _T):
        raise NotImplementedError

    def generate_from_batch(self, set_size: int, all_deltas, zeta_t, return_all=False):
        """
        all_deltas: [M, N, D]
        zeta_t:     [M, N, 1]   (i.e. means for simulated samples)
        """
        assert (all_deltas[:,0] == 0.0).all(), "Convention requires that first item is always cued"
        function_evals = self.target_function(all_deltas)

        data_dict = self.data_generator.full_data_generation(
            set_size,
            vm_means = zeta_t.squeeze(-1).to(self.device),
            model_evaulations = function_evals.unsqueeze(0).to(self.device),    # Unsqueeze once for I axis (analogous to 1 monte carlo draw)
        )

        if return_all:
            return data_dict, function_evals

        else:

            if self.output_layer == 'error':
                estimates = data_dict['samples'].squeeze(0).unsqueeze(-1)   # [I (1 for sure), M] -> [M, 1]
                data = estimates - zeta_t.squeeze(-1).to(self.device)       # [M, N]
            elif self.output_layer == 'beta':
                data = data_dict['betas'].squeeze(0)                        # [M]
            elif self.output_layer == 'pi':
                data = data_dict['pi_vectors'].squeeze(0)                   # [M, N+1]

            return data

    def new_train_batch(self, *_, dimensions: list):
        raise Exception('EstimateDataLoaderBase.new_train_batch deprecated, use iterate_train_batches instead')
        batch_indices = random.sample(self.train_indices, self.M_batch)
        deltas_batch = self.all_deltas[batch_indices].to(self.device)
        train_output_indices = [self.train_indices.index(bi) for bi in batch_indices]
        output_batch = self.train_outputs[train_output_indices].to(self.device)
        return deltas_batch, output_batch

    def iterate_train_batches(self, *_, dimensions: list, shuffle, total = None, return_indices = False):
        raise Exception('Have not yet converted new_train_batch to iterate_train_batches for TargetFunctionDataGeneratorBase')

    def all_test_batches(self, *_, dimensions: list):
        for i in range(self.num_test_batches):
            if self.M_batch > 0:
                test_output_slicer = slice(i*self.M_batch, (i+1)*self.M_batch)
            else:
                test_output_slicer = slice(None, None)
            deltas_indices = self.test_indices[test_output_slicer]
            deltas_batch = self.all_deltas[deltas_indices].to(self.device)
            yield (
                deltas_batch[...,dimensions],
                self.test_outputs[test_output_slicer].to(self.device)
            )




class ZeroSwapTargetFunctionDataGenerator(TargetFunctionDataGeneratorBase):

    """
    No swaps occur under this model

    zero_delta_magnitude is f(0), which is relative to pi_u_tilde being 0!
    """

    def __init__(self, N, zero_delta_magnitude, torus_dimensionality, target_concentration, underlying_data_generator, device='cuda') -> None:
        self.zero_delta_magnitude = zero_delta_magnitude
        super().__init__(N, torus_dimensionality, target_concentration, underlying_data_generator, device = device)
    
    def target_function(self, deltas):
        at_zero = (deltas == 0.0)
        at_zero = at_zero[...,0].logical_and(at_zero[...,1])   # [M, N]
        canvas = torch.zeros_like(at_zero).to(deltas.dtype)
        canvas[at_zero] = self.zero_delta_magnitude
        canvas[~at_zero] = - float('inf')
        return canvas


class TargetModulatedTargetFunctionDataGenerator(TargetFunctionDataGeneratorBase):

    """
    Swap function only modulated along the target feature dimension

    zero_delta_magnitude is f(0), which is relative to pi_u_tilde being 0!
    """

    def __init__(self, N, surface_conc, surface_offset, surface_magnitude, target_feature_idx, torus_dimensionality, target_concentration, underlying_data_generator, device='cuda') -> None:
        self.surface_conc = torch.tensor(surface_conc)
        self.surface_offset = surface_offset
        self.surface_magnitude = surface_magnitude
        self.target_feature_idx = target_feature_idx
        super().__init__(N, torus_dimensionality, target_concentration, underlying_data_generator, device = device)
    
    def target_function(self, deltas):
        relevant_feature = deltas[...,self.target_feature_idx] # [M, N]
        f_eval = zero_mean_von_mises_log_prob_with_kappa(self.surface_conc, relevant_feature).exp()
        f_eval = (self.surface_magnitude * f_eval) + self.surface_offset
        return f_eval



class ZeroSwapMisattributionCheckerSetSizesEnvelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    """
    target_concentration_dict maps {N: kappa}
    zero_delta_magnitude_dict maps {N: f_N(0)}
    """

    def __init__(self, underlying_envelope: MultipleSetSizesActivitySetDataGeneratorEnvelopeBase, torus_dimensionality: int, target_concentration_dict: dict, zero_delta_magnitude_dict: dict, device = 'cuda'):

        print('No splitting of participants here!')

        data_generators = {
            N: ZeroSwapTargetFunctionDataGenerator(
                N, zero_delta_magnitude_dict[N], torus_dimensionality, 
                target_concentration_dict[N], dg, device = device
            )
            for N, dg in underlying_envelope.data_generators.items()
        }

        super().__init__(underlying_envelope.M_batch, underlying_envelope.feature_names, data_generators, device)


class TargetModulatedSwapMisattributionCheckerSetSizesEnvelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    """
    surface_conc_dict maps {N: conc of surface}
    surface_magnitude_dict maps {N: * surface}
    surface_offset_dict maps {N: +- surface}
    target_concentration_dict maps {N: kappa}
    """

    def __init__(self, underlying_envelope: MultipleSetSizesActivitySetDataGeneratorEnvelopeBase, torus_dimensionality: int, target_feature_idx: int, target_concentration_dict: dict, surface_conc_dict: dict, surface_magnitude_dict: dict, surface_offset_dict: dict, device = 'cuda'):

        print('No splitting of participants here!')

        data_generators = {
            N: TargetModulatedTargetFunctionDataGenerator(
                N, surface_conc_dict[N], surface_offset_dict[N],
                surface_magnitude_dict[N], target_feature_idx, torus_dimensionality,
                target_concentration_dict[N], dg, device = device
            )
            for N, dg in underlying_envelope.data_generators.items()
        }

        super().__init__(underlying_envelope.M_batch, underlying_envelope.feature_names, data_generators, device)




