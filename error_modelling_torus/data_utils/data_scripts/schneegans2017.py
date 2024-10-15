import numpy as np
import torch
import pandas as pd

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase


data_path1 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/schneegans2017/Data/dataExp1.csv"
data_path2 = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/schneegans2017/Data/dataExp2.csv"


EXPERIMENT_2_VALID_TASK_TYPES = ['cueColor_reportLocation', 'cueOrientation_reportLocation', 'cueColor_reportOrientation', 'cueOrientation_reportColor']


class Schneegans2017SingleSetSize(EstimateDataLoaderBase):

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




class Schneegans2017Experiment2Envelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    """
    Right now we only work with the first half of experiment 2, i.e. not reporting location
    """

    D = 2

    def __init__(self, *_, task_type, M_batch, M_test, num_repeats: int, subjects = None, device = 'cuda'):

        assert task_type in EXPERIMENT_2_VALID_TASK_TYPES

        feature_names = [
            'color' if task_type.split('_')[0] == 'cueColor' else 'orientation',
            'location' if task_type.split('_')[1] == 'reportLocation' else 'orientation' if task_type.split('_')[0] == 'cueColor' else 'color',
        ]

        df = pd.read_csv(data_path2)
        df = (df[df.reportColor == 0] if task_type == 'reportOrientation' else df[df.reportColor == 1]).reset_index()

        if 'reportLocation' in task_type:
            df = (df[df.reportLocationFirst == (1 if feature_names[1] == 'location' else 0)]).reset_index()

        assert np.isnan(df[f'response{feature_names[0].capitalize()}'].values).all()

        import pdb; pdb.set_trace(header = 'double orientations then rectify!')

        all_errors = rectify_angles(df[[f'response{feature_names[1].capitalize()}']].values - df[[f'target{feature_names[1].capitalize()}'] + [f'nonTarget{feature_names[1].capitalize()}s_{i}' for i in range(1, 6)]].values)
        all_deltas = np.stack(
            [
                df[[f'target{feature_names[j].capitalize()}'] + [f'nonTarget{feature_names[j].capitalize()}s_{i}' for i in range(1, 6)]].values - df[[f'target{feature_names[j].capitalize()}']].values
                for j in [0,1]
            ],
            -1
        )

        all_target_zetas = df[[f'target{feature_names[1].capitalize()}'] + [f'nonTarget{feature_names[1].capitalize()}s_{i}' for i in range(1, 6)]].values

        rectify_angles(df[['nonTargetLocations_1', 'nonTargetLocations_2','nonTargetLocations_3', 'nonTargetLocations_4','nonTargetLocations_5',]].sub(df.targetLocation, axis = 0))
        data_array2 = {
            'all_deltas': all_deltas,
            'all_errors': all_errors,
            'all_target_zetas': all_target_zetas,
        }

        data_generators = {6: Schneegans2017SingleSetSize(**data_array2, M_batch=M_batch, M_test=M_test, num_repeats=num_repeats, subjects=subjects, device=device)}

        feature_names = ['location', 'orientation']

        super().__init__(M_batch, feature_names, data_generators, device)


