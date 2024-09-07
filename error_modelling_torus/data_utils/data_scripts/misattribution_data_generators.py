import torch
import numpy as np
from scipy.io import loadmat

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import TargetFunctionDataGeneratorBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase

from purias_utils.util.error_modelling import zero_mean_von_mises_log_prob_with_kappa



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




