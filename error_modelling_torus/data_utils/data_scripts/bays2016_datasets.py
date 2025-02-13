import torch
import numpy as np
from scipy.io import loadmat

from typing import List, Optional

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase

data_path = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/bays2016/big_data_Feb19.mat'


class DatasetsUsedInBays2016SingleSetSize(EstimateDataLoaderBase):

    """
    errors come in as [subjects, trials]
    delta_cued, deltas_estimated come in as [subjects, trials, max setsize]
    target_zetas come in as [subjects, trials, max setsize]
    """

    def __init__(self, N, errors, delta_cued, deltas_estimated, target_zetas, M_batch, M_test, num_repeats, subjects, device) -> None:

        self.N = int(N)
        self.device = device

        if subjects is None:
            subjects = ...

        subject_selected_errors = errors[subjects,:,:N].reshape(-1, N)                  
        subject_selected_delta_cued = delta_cued[subjects,:,:N].reshape(-1, N)
        subject_selected_delta_estimated = deltas_estimated[subjects,:,:N].reshape(-1, N)
        subject_selected_deltas = np.stack([subject_selected_delta_cued.reshape(-1, N), subject_selected_delta_estimated.reshape(-1, N)], -1)
        subject_selected_target_zetas = target_zetas[subjects,:,:N].reshape(-1, N, 1)

        #subject_selected_deltas = subject_selected_deltas[~np.isnan(subject_selected_deltas).any(axis = 1, keepdims=True).repeat(N, 1)]
        #subject_selected_target_zetas = subject_selected_target_zetas[~np.isnan(subject_selected_target_zetas).any(axis = 0)]
        #subject_selected_errors = subject_selected_errors[~np.isnan(subject_selected_errors).any(axis = 0)]

        # Sometimes not all trials are completed
        if (np.isnan(subject_selected_deltas).any() or np.isnan(subject_selected_target_zetas).any() or np.isnan(subject_selected_errors).any()):
            assert (np.isnan(subject_selected_deltas).any(-1).any(-1) == np.isnan(subject_selected_target_zetas).any(-1).any(-1)).all()
            assert (np.isnan(subject_selected_deltas).any(-1).any(-1) == np.isnan(subject_selected_errors).any(-1)).all()

            mask = ~np.isnan(subject_selected_deltas).any(-1).any(-1)
            subject_selected_deltas = subject_selected_deltas[mask]
            subject_selected_target_zetas = subject_selected_target_zetas[mask]
            subject_selected_errors = subject_selected_errors[mask]

        # Some stimuli are not on the invisible circle...
        usable_trials_mask = (subject_selected_deltas[:,1:,0] != 0).all(-1)
        dropped_trials = (~usable_trials_mask).sum()
        if dropped_trials > 0:
            print(f'Dropping {dropped_trials} trials for N = {N}')

        all_deltas = torch.tensor(subject_selected_deltas[usable_trials_mask], device = device)    # [M, N, D (2)]
        all_target_zetas = torch.tensor(subject_selected_target_zetas[usable_trials_mask], device = device)    # [M, N, 1]
        all_errors = torch.tensor(subject_selected_errors[usable_trials_mask], device = device)    # [M, N]

        super().__init__(all_deltas, all_errors, all_target_zetas, M_batch, M_test, num_repeats, device)



class DatasetsUsedInBays2016Envelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    dataset_idx: int
    estimated_feature_name: str

    def __init__(self, *_,  M_batch: int, M_test: int, num_repeats: int, subjects: Optional[List[int]] = None, device = 'cuda'):

        mat = loadmat(data_path)
        dataset_mat = {k: v[0,self.dataset_idx] for k, v in mat.items() if not k.startswith('__')}

        # Errors
        errors = np.concatenate([dataset_mat['error'][...,None], dataset_mat['nt_error']], -1)

        # Deltas[1]
        deltas_estimated = rectify_angles(dataset_mat['nt_dist'])
        print(np.nanmin(np.abs(deltas_estimated)) / np.pi * 180)   # 24 degrees
        deltas_estimated = np.concatenate((np.zeros_like(errors)[...,[0]], deltas_estimated), -1) # [subjects, num setsizes, trials, max setsize]

        # Deltas[0] - always location
        x_loc_target, y_loc_target = dataset_mat['t_pos'][0,0]
        x_loc_distactor, y_loc_distactor = dataset_mat['nt_pos'][0,0]
        x_loc = np.concatenate([x_loc_target[...,None], x_loc_distactor], -1)
        y_loc = np.concatenate([y_loc_target[...,None], y_loc_distactor], -1)
        location_angles = np.arctan2(y_loc, x_loc)
        delta_cued = rectify_angles(location_angles - location_angles[...,[0]])

        # example_location_angles = location_angles[:,2,:-10,:4]
        # example_x_loc = x_loc[:,2,:-10,:4]
        # example_y_loc = y_loc[:,2,:-10,:4]
        # import matplotlib.pyplot as plt
        # plt.scatter(example_x_loc.flatten(), example_y_loc.flatten(), c = example_location_angles.flatten())

        # Actual estimated feature values
        if dataset_mat['target'].size:
            target_zetas = np.concatenate([dataset_mat['target'][...,None], dataset_mat['nontargets']], -1)
        else:
            print('Using deltas as target zetas!')
            target_zetas = deltas_estimated.copy()


        set_sizes: List[int] = dataset_mat['n_items'].flatten().tolist()

        # Split into data_generators based on set size
        assert (subjects is None) or (set(subjects).intersection(set(range(len(deltas_estimated)))) == set(subjects))

        data_generators = {
            N: DatasetsUsedInBays2016SingleSetSize(N, errors[:,i], delta_cued[:,i], deltas_estimated[:,i], target_zetas[:,i], M_batch, M_test, num_repeats, subjects, device)
            for i, N in enumerate(set_sizes)
        }

        feature_names = ['location', self.estimated_feature_name]
        super().__init__(M_batch, feature_names, data_generators, device)


    


class VanDenBerg2012ColourEnvelope(DatasetsUsedInBays2016Envelope):
    D = 2
    dataset_idx = 2
    estimated_feature_name = 'colour'
    

class VanDenBerg2012OrientationEnvelope(DatasetsUsedInBays2016Envelope):
    D = 2
    dataset_idx = 3
    estimated_feature_name = 'orientation'


class Bays2014OrientationEnvelope(DatasetsUsedInBays2016Envelope):
    D = 2
    dataset_idx = 5
    estimated_feature_name = 'orientation'
    

class GorgoraptisOrientationEnvelope(DatasetsUsedInBays2016Envelope):
    D = 3
    dataset_idx = 9
    estimated_feature_name = 'orientation'
    def __init__(self, ):
        raise Exception('Colour and/or location not accounted for!')
    
