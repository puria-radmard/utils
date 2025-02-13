import torch
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase

data_path1 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/mcmaster2022/EXP_1__Data.mat"
data_path2 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/mcmaster2022/EXP_2__Data.mat"
data_path3 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/mcmaster2022/Exp3_data_{loc_type}_locations.npy"

EXPERIMENT_1_SUBTASKS = ['oricue', 'loccue']
EXPERIMENT_1_SUBJECTS = list(range(10))
EXPERIMENT_1_ELONGATIONS = ['cue_AR1', 'cue_AR2', 'cue_AR3']   # low, medium, high

EXPERIMENT_2_SUBTASKS = ['dircue', 'loccue']
EXPERIMENT_2_SUBJECTS = list(range(10))
EXPERIMENT_2_COHERENCES = ['lowC', 'medC', 'highC']   # low, medium, high

EXPERIMENT_3_LOCTYPES = ['rot', 'fix', 'rand']
EXPERIMENT_3_SUBJECTS = list(range(243))



class McMaster2022SingleSetSize(EstimateDataLoaderBase):

    def __init__(self, data_array, N, experiment_number, M_batch, M_test, num_repeats, subtask, subjects, stimulus_strength_feature, target_dim_key, probe_dim_key, distractor_target_dim_key, device, ori_dim_idx=None, output_oris = False):

        subject_structs = data_array[subtask][0]

        across_subjects_deltas = []
        across_subjects_target_zetas = []
        across_subjects_errors = []

        all_subject_idxs = []
        # all_stimulus_strengths = []

        for subject in subjects:
            subject_struct = subject_structs[subject]

            target_item_idx = subject_struct['T_ind'][0,0].reshape(-1) - 1   # [M]
            M = len(target_item_idx)
            target_idx_mask = np.zeros([M, N]).astype(bool)
            target_idx_mask[list(range(M)), target_item_idx] = True
            distractor_idx_mask = ~target_idx_mask
            
            # Relevant info
            subject_responses = subject_struct['Resp'][0,0]                                                 # [M, 1]
            zeta_target_cued = subject_struct['full' + target_dim_key][0,0][target_idx_mask, np.newaxis]  # [M, 1]
            zeta_probed_cued = subject_struct[probe_dim_key][0,0][target_idx_mask, np.newaxis]            # [M, 1]
            zeta_target_distractors = subject_struct['full' + target_dim_key][0,0][distractor_idx_mask].reshape(M, N - 1)
            zeta_probed_distractors = subject_struct[probe_dim_key][0,0][distractor_idx_mask].reshape(M, N - 1)

            # Convert to style we need
            zeta_probed = np.concatenate([zeta_probed_cued, zeta_probed_distractors], 1)
            zeta_target = np.concatenate([zeta_target_cued, zeta_target_distractors], 1)
            zetas = torch.tensor(np.stack([zeta_probed, zeta_target], 2))
            subject_deltas = rectify_angles(zetas - zetas[:, [0], :]).to(device)   # [M, N, D (2)]
            subject_errors = rectify_angles(torch.tensor(subject_responses) - zeta_target).to(device)  # [M, N]
            if output_oris:
                subject_errors = rectify_angles(2 * subject_errors)
            subject_target_zetas = torch.tensor(zeta_target, device = device).unsqueeze(-1)

            # Reduce by selected stimulus_strength_feature
            strengths_mask = torch.tensor(subject_struct[stimulus_strength_feature[0]][0,0])
            for e_key in stimulus_strength_feature[1:]:
                strengths_mask = torch.logical_or(strengths_mask, torch.tensor(subject_struct[e_key][0,0]))
            strengths_mask = strengths_mask[:,0].bool()

            subject_deltas = subject_deltas[strengths_mask]
            subject_errors = subject_errors[strengths_mask]
            subject_target_zetas = subject_target_zetas[strengths_mask]

            # Double check values to be sure...
            assert (zeta_target_cued == subject_struct['T'][0,0]).all()
            assert (zeta_probed_cued == subject_struct[probe_dim_key][0,0][range(M), target_item_idx, np.newaxis]).all()
            if experiment_number == 1:
                assert (zeta_target_distractors == subject_struct[distractor_target_dim_key][0,0]).all()
            else:
                assert (zeta_target == subject_struct[distractor_target_dim_key][0,0]).all() # misnomer
            
            assert torch.isclose(
                rectify_angles(subject_errors[:,[0]]) - torch.tensor(subject_struct['RE'][0,0])[strengths_mask].to(device),
                torch.tensor(0.0).to(subject_errors.dtype)
            ).all()

            # Extend across subjects
            across_subjects_deltas.append(subject_deltas)
            across_subjects_target_zetas.append(subject_target_zetas)
            across_subjects_errors.append(subject_errors)

            # Extend metadata
            all_subject_idxs.extend([subject] * len(subject_deltas))
            # all_stimulus_strengths.extend(new_strengths.tolist())

        all_deltas = torch.concat(across_subjects_deltas, 0)
        all_target_zetas = torch.concat(across_subjects_target_zetas, 0)
        all_errors = torch.concat(across_subjects_errors, 0).squeeze(-1)

        self.subjects = subjects
        self.stimulus_strength_feature = stimulus_strength_feature

        all_metadata = {
            'subject_idx': all_subject_idxs,
            # 'stimulus_strength': all_stimulus_strengths,
        }

        if ori_dim_idx is not None:
            # all_deltas[...,ori_dim_idx] = all_deltas[...,ori_dim_idx] / 2.
            all_deltas[...,ori_dim_idx] = rectify_angles(all_deltas[...,ori_dim_idx] * 2.)

        super().__init__(all_deltas, all_errors, all_target_zetas, all_metadata, M_batch, M_test, num_repeats, device)


class McMaster2022Exp3SingleSetSize(EstimateDataLoaderBase):

    def __init__(self, all_deltas, all_errors, all_target_zetas,  M_batch, M_test, num_repeats, subjects, device, **kwargs):

        self.subjects = subjects
        all_deltas = torch.tensor(all_deltas, dtype=torch.float32)
        all_errors = torch.tensor(all_errors, dtype=torch.float32)
        all_target_zetas = torch.tensor(all_target_zetas, dtype=torch.float32)

        all_deltas[...,1] = rectify_angles(all_deltas[...,1] * 2.)  # orientation
        all_errors = rectify_angles(2 * all_errors)

        if subjects is None:
            super().__init__(all_deltas, all_errors, all_target_zetas, M_batch, M_test, num_repeats, device)
        else:
            raise NotImplementedError('implement')



class McMaster2022ExperimentOneEnvelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    D = 2

    """
    e.g.:
    subtask = 'oricue'
    subjects = None or [0]
    stim_strengths = None or ['cue_AR1', 'cue_AR2']
    """

    def __init__(self, *_, M_batch, M_test, num_repeats: int, subtask, subjects = None, stim_strengths = None, device = 'cuda'):

        assert subtask in EXPERIMENT_1_SUBTASKS
        assert (subjects is None) or (set(subjects).intersection(EXPERIMENT_1_SUBJECTS) == set(subjects))
        assert (stim_strengths is None) or (set(stim_strengths).intersection(EXPERIMENT_1_ELONGATIONS) == set(stim_strengths)), stim_strengths
        if (subjects is None):
            subjects = EXPERIMENT_1_SUBJECTS
        if (stim_strengths is None):
            stim_strengths = EXPERIMENT_1_ELONGATIONS  # Select all

        data_array1: dict = loadmat(data_path1)

        if subtask == 'oricue':
            target_dim_key, probe_dim_key, distractor_target_dim_key = 'L', 'O', 'LwoT'
        else:
            target_dim_key, probe_dim_key, distractor_target_dim_key = 'O', 'L', 'OwoT'

        ori_dim_idx = 0 if subtask == 'oricue' else 1
        output_oris = (subtask == 'loccue')
        data_generators = {6: McMaster2022SingleSetSize(data_array1, 6, 1, M_batch, M_test, num_repeats, subtask, subjects, stim_strengths, target_dim_key, probe_dim_key, distractor_target_dim_key, device, ori_dim_idx, output_oris)}

        if subtask == 'oricue':
            feature_names = ['orientation', 'location']
        else:
            feature_names = ['location', 'orientation']

        super().__init__(M_batch, feature_names, data_generators, device)

        # import matplotlib.pyplot as plt
        # plt.hist(self.data_generators[6].all_errors[:,0].cpu().numpy(), 50)
        # plt.savefig('asdf.png')



class McMaster2022ExperimentTwoEnvelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    D = 2

    """
    e.g.:
    subtask = 'dircue'
    subjects = None
    stim_strengths = None
    """

    def __init__(self, *_, M_batch, M_test, subtask, num_repeats: int, subjects = None, stim_strengths = None, device = 'cuda'):

        assert subtask in EXPERIMENT_2_SUBTASKS
        assert (subjects is None) or (set(subjects).intersection(EXPERIMENT_2_SUBJECTS) == set(subjects))
        assert (stim_strengths is None) or (set(stim_strengths).intersection(EXPERIMENT_2_COHERENCES) == set(stim_strengths)), EXPERIMENT_2_COHERENCES
        if (subjects is None):
            subjects = EXPERIMENT_2_SUBJECTS
        if (stim_strengths is None):
            stim_strengths = EXPERIMENT_2_COHERENCES  # Select all

        data_array2: dict = loadmat(data_path2)

        if subtask == 'dircue':
            target_dim_key, probe_dim_key, distractor_target_dim_key = 'L', 'D', 'L_reshape'
        else:
            # raise NotImplementedError('Waiting for Jessica\'s response')
            target_dim_key, probe_dim_key, distractor_target_dim_key = 'D', 'L', 'fullD'

        data_generators = {4: McMaster2022SingleSetSize(data_array2, 4, 2, M_batch, M_test, num_repeats, subtask, subjects, stim_strengths, target_dim_key, probe_dim_key, distractor_target_dim_key, device)}

        if subtask == 'dircue':
            feature_names = ['direction', 'location']
        else:
            raise Exception('Need to ask Jessica again!')
            feature_names = ['location', 'direction']

        super().__init__(M_batch, feature_names, data_generators, device)



class McMaster2022ExperimentThreeEnvelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    D = 2

    def __init__(self, *_, loc_type, M_batch, M_test, num_repeats: int, subjects = None, device = 'cuda'):

        data_array2: dict = np.load(data_path3.format(loc_type=loc_type), allow_pickle=True).item()

        data_generators = {6: McMaster2022Exp3SingleSetSize(**data_array2, M_batch=M_batch, M_test=M_test, num_repeats=num_repeats, subjects=subjects, device=device)}

        feature_names = ['location', 'orientation']

        super().__init__(M_batch, feature_names, data_generators, device)



if __name__ == '__main__':

    dataset_generator = McMaster2022ExperimentOneEnvelope(32, 200, 10, 'loccue', None, None, device = 'cpu')
    dataset_generator = McMaster2022ExperimentOneEnvelope(32, 200, 10, 'oricue', None, None, device = 'cpu')

#    dataset_generator = McMaster2022ExperimentTwoEnvelope(32, 200, 10, 'loccue', None, None, device = 'cpu')
    dataset_generator = McMaster2022ExperimentTwoEnvelope(32, 200, 10, 'dircue', None, None, device = 'cpu')

