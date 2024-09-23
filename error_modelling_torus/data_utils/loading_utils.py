import os, sys, json, random, torch, numpy as np

from typing import Type

from purias_utils.error_modelling_torus.data_utils.base import MultipleSetSizesActivitySetDataGeneratorEnvelopeBase
from purias_utils.error_modelling_torus.data_utils.data_scripts.bays2009 import Bays2009MultipleSetSizesEnvelope
from purias_utils.error_modelling_torus.data_utils.data_scripts.mcmaster2022 import McMaster2022ExperimentOneEnvelope, McMaster2022ExperimentTwoEnvelope
from purias_utils.error_modelling_torus.data_utils.data_scripts.bays2016_datasets import VanDenBerg2012ColourEnvelope, VanDenBerg2012OrientationEnvelope, Bays2014OrientationEnvelope, GorgoraptisOrientationEnvelope


from purias_utils.error_modelling_torus.data_utils.data_scripts.misattribution_data_generators import ZeroSwapMisattributionCheckerSetSizesEnvelope
from purias_utils.error_modelling_torus.data_utils.data_scripts.misattribution_data_generators import TargetModulatedSwapMisattributionCheckerSetSizesEnvelope


from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles


dataset_choices = ['bays2009', 'mcmaster2022_e1_oricue', 'mcmaster2022_e2_dircue', 'mcmaster2022_e1_loccue', 'vandenberg2012_color', 'vandenberg2012_orientation', 'bays2014_orientation', 'gorgoraptis2011_orientation']


def dump_training_indices_to_path(dataset_generator: MultipleSetSizesActivitySetDataGeneratorEnvelopeBase, dest_base_path):
    train_indices = {N: dg.train_indices for N, dg in dataset_generator.data_generators.items()}
    with open(os.path.join(dest_base_path, 'train_indices.json'), 'w') as jf:
        json.dump(train_indices, jf)



def load_experimental_data(dataset_name: str, train_indices_seed: int, train_indices_path: str, M_batch: int, M_test_per_set_size: int, num_repeats: int, data_subselection_args, **kwargs) -> MultipleSetSizesActivitySetDataGeneratorEnvelopeBase:

    assert not ((train_indices_seed is not None) and (train_indices_path is not None))

    if train_indices_seed is not None:
        prev_seed = random.randrange(sys.maxsize)
        random.seed(train_indices_seed)

    if dataset_name == 'bays2009':
        dataset_generator = Bays2009MultipleSetSizesEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, participant_id=data_subselection_args.participant_id, stimulus_exposure_id=data_subselection_args.stimulus_exposure_id)    
    elif dataset_name == 'mcmaster2022_e1_oricue':
        dataset_generator = McMaster2022ExperimentOneEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='oricue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths)
    elif dataset_name == 'mcmaster2022_e2_dircue':
        dataset_generator = McMaster2022ExperimentTwoEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='dircue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths)
    elif dataset_name == 'mcmaster2022_e1_loccue':
        dataset_generator = McMaster2022ExperimentOneEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='loccue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths)
    elif dataset_name == 'vandenberg2012_color':
        dataset_generator = VanDenBerg2012ColourEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=None)
    elif dataset_name == 'vandenberg2012_orientation':
        dataset_generator = VanDenBerg2012OrientationEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=None)
    elif dataset_name == 'bays2014_orientation':
        dataset_generator = Bays2014OrientationEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=None)
    elif dataset_name == 'gorgoraptis2011_orientation':
        dataset_generator = GorgoraptisOrientationEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=None)
    else:
        raise ValueError(dataset_name)

    if train_indices_seed is not None:
        random.seed(prev_seed)

    if train_indices_path is not None:
        with open(os.path.join(train_indices_path, f"train_indices.json"), 'r') as jf:
            train_indices = json.load(jf)

        for N, dg in dataset_generator.data_generators.items():
            dg.set_train_indices(train_indices[str(N)])

    return dataset_generator



# load_synthetic_data(dataset_generator, args.synthetic_data_path, args.synthetic_data_code, args.train_examples_discarded_per_set_size)

def load_synthetic_data(dataset_generator: Type[MultipleSetSizesActivitySetDataGeneratorEnvelopeBase], synthetic_data_path, synthetic_data_code):

    data_path = os.path.join(synthetic_data_path, f'synthetic_data_{synthetic_data_code}.npy')
    data = np.load(data_path, allow_pickle=True).item()

    for N, dg in dataset_generator.data_generators.items():
        import pdb; pdb.set_trace(header = 'check shapes are the same here!')

        synthetic_errors = torch.tensor(data['errors'][N]).to(dtype = dg.all_errors.dtype, device = dg.all_errors.device)

        import pdb; pdb.set_trace(header = 'make some args up here perhaps for ensuring this...')
        assert dg.all_errors.shape == synthetic_errors.shape        # [Q, M, N], i.e. includes num_repeats!

        dg.all_errors = synthetic_errors

    true_surfaces = data['function_eval']

    return dataset_generator, true_surfaces

