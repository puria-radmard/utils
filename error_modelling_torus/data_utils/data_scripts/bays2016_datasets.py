import torch
import numpy as np
from scipy.io import loadmat

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase

data_path = '/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/bays2016/big_data_Feb19.mat'


class DatasetsUsedInBays2016SingleSetSize(EstimateDataLoaderBase):

    """
    errors come in as [subjects, trials]
    delta_cued, deltas_estimated come in as [subjects, trials, max setsize]
    target_zetas come in as [subjects, trials, max setsize]
    """

    def __init__(self, N, errors, delta_cued, deltas_estimated, target_zetas, M_batch, M_test, subjects, device) -> None:
        super().__init__()

        self.N = int(N)
        self.device = device

        if subjects is None:
            subjects = ...

        subject_selected_errors = errors[subjects,:,:N].reshape(-1, N)
        subject_selected_delta_cued = delta_cued[subjects,:,:N].reshape(-1, N)
        subject_selected_delta_estimated = deltas_estimated[subjects,:,:N].reshape(-1, N)
        subject_selected_deltas = np.stack([subject_selected_delta_cued.reshape(-1, N), subject_selected_delta_estimated.reshape(-1, N)], -1)
        subject_selected_target_zetas = target_zetas[subjects,:,:N].reshape(-1, N, 1)

        import pdb; pdb.set_trace(header = 'need to remove all nans!')

        subject_selected_deltas = subject_selected_deltas[~np.isnan(subject_selected_deltas).any(axis = 0)]
        subject_selected_target_zetas = subject_selected_target_zetas[~np.isnan(subject_selected_target_zetas).any(axis = 0)]
        subject_selected_errors = subject_selected_errors[~np.isnan(subject_selected_errors).any(axis = 0)]

        self.all_deltas = torch.tensor(subject_selected_deltas, device = device)    # [M, N, D (2)]
        self.all_target_zetas = torch.tensor(subject_selected_target_zetas, device = device)    # [M, N]
        self.all_errors = torch.tensor(subject_selected_errors, device = device)    # [M, N]

        M_train_each = self.all_deltas.shape[0] - M_test
        print(M_train_each, 'training examples and', M_test, 'testing examples for N =', self.N)
        self.__dict__.update(self.sort_out_M_bullshit(M_batch, M_train_each, M_test))




class DatasetsUsedInBays2016Envelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    dataset_idx: int
    estimated_feature_name: str

    def __init__(self, M_batch, M_test, subjects = None, device = 'cuda'):

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

        # Actual estimated feature values
        target_zetas = np.concatenate([dataset_mat['target'][...,None], dataset_mat['nontargets']], -1)

        set_sizes = dataset_mat['n_items'].flatten().tolist()

        # Split into data_generators based on set size
        assert (subjects is None) or (set(subjects).insersection(set(range(len(deltas_estimated)))) == set(subjects))

        data_generators = {
            N: DatasetsUsedInBays2016SingleSetSize(N, errors[:,i], delta_cued[:,i], deltas_estimated[:,i], target_zetas[:,i], M_batch, M_test, subjects, device)
            for i, N in enumerate(set_sizes)
        }

        feature_names = ['location', self.estimated_feature_name]
        super().__init__(M_batch, feature_names, data_generators, device)


    


class VanDenBerg2012ColourEnvelope(DatasetsUsedInBays2016Envelope):
    dataset_idx = 2
    estimated_feature_name = 'colour'
    

class VanDenBerg2012OrientationEnvelope(DatasetsUsedInBays2016Envelope):
    dataset_idx = 3
    estimated_feature_name = 'orientation'


class Bays2014OrientationEnvelope(DatasetsUsedInBays2016Envelope):
    dataset_idx = 5
    estimated_feature_name = 'orientation'
    

class GorgoraptisOrientationEnvelope(DatasetsUsedInBays2016Envelope):
    dataset_idx = 9
    estimated_feature_name = 'orientation'
    
