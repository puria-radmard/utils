import numpy as np
import torch
import pandas as pd

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase


data_path1 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/schneegans2017/Data/dataExp1.csv"
data_path2 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/schneegans2017/Data/dataExp2.csv"


EXPERIMENT_2_VALID_TASK_TYPES = ['cueColor_reportLocation', 'cueOrientation_reportLocation', 'cueColor_reportOrientation', 'cueOrientation_reportColor']


class Schneegans2017SingleSetSize(EstimateDataLoaderBase):

    def __init__(self, all_deltas, all_errors, all_target_zetas, all_subject_idxs, M_batch, M_test, num_repeats, subjects, device, **kwargs):

        if subjects is not None:
            raise NotImplementedError('Have not implemented subject filtering for Schneegans2017SingleSetSize yet!')

        self.subjects = subjects
        all_deltas = torch.tensor(all_deltas, dtype=torch.float32)
        all_errors = torch.tensor(all_errors, dtype=torch.float32)
        all_target_zetas = torch.tensor(all_target_zetas, dtype=torch.float32)

        all_metadata = {
            'subject_idx': all_subject_idxs,
            # 'stimulus_strength': all_stimulus_strengths,
        }

        if subjects is None:
            super().__init__(all_deltas, all_errors, all_target_zetas, all_metadata, M_batch, M_test, num_repeats, device)
        else:
            raise NotImplementedError('implement')




class Schneegans2017Experiment2Envelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    """
    Right now we only work with the first half of experiment 2, i.e. not reporting location
    """

    D = 2

    def __init__(self, *_, M_batch, M_test, num_repeats: int, task_type: str, subjects = None, device = 'cuda'):

        assert task_type in EXPERIMENT_2_VALID_TASK_TYPES

        feature_names = [
            'color' if task_type.split('_')[0] == 'cueColor' else 'orientation',
            'location' if task_type.split('_')[1] == 'reportLocation' else 'orientation' if task_type.split('_')[1] == 'reportOrientation' else 'color',
        ]

        df = pd.read_csv(data_path2)
        df = (df[df.reportColor == 1] if feature_names[0] == 'orientation' else df[df.reportColor == 0]).reset_index()
        assert np.isnan(df[f'response{feature_names[0].capitalize()}'].values).all()
        df = (df[df.reportLocationFirst == (1 if feature_names[1] == 'location' else 0)]).reset_index()

        for ori_col in filter(lambda x: 'Orientation' in x, df.columns):
            df[ori_col] = df[ori_col].map(lambda x: rectify_angles(2 * x))

        all_errors = rectify_angles(df[[f'response{feature_names[1].capitalize()}']].values - df[[f'target{feature_names[1].capitalize()}'] + [f'nonTarget{feature_names[1].capitalize()}s_{i}' for i in range(1, 6)]].values)
        all_deltas = np.stack(
            [
                df[[f'target{feature_names[j].capitalize()}'] + [f'nonTarget{feature_names[j].capitalize()}s_{i}' for i in range(1, 6)]].values - df[[f'target{feature_names[j].capitalize()}']].values
                for j in [0,1]
            ],
            -1
        )

        all_target_zetas = df[[f'target{feature_names[1].capitalize()}'] + [f'nonTarget{feature_names[1].capitalize()}s_{i}' for i in range(1, 6)]].values[...,None]

        data_array2 = {
            'all_deltas': rectify_angles(all_deltas),
            'all_errors': all_errors,
            'all_target_zetas': all_target_zetas,
            'all_subject_idxs': (df.subject - 1).tolist()
        }

        data_generators = {6: Schneegans2017SingleSetSize(**data_array2, M_batch=M_batch, M_test=M_test, num_repeats=num_repeats, subjects=subjects, device=device)}

        super().__init__(M_batch, feature_names, data_generators, device)


