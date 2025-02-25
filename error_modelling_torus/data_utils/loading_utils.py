import os, sys, json, random, torch, numpy as np

from typing import Type, Union, Optional

from purias_utils.error_modelling_torus.data_utils.base import MultipleSetSizesActivitySetDataGeneratorEnvelopeBase
from purias_utils.error_modelling_torus.data_utils.data_scripts.bays2009 import Bays2009MultipleSetSizesEnvelope
from purias_utils.error_modelling_torus.data_utils.data_scripts.schneegans2017 import Schneegans2017Experiment2Envelope, EXPERIMENT_2_VALID_TASK_TYPES
from purias_utils.error_modelling_torus.data_utils.data_scripts.mcmaster2022 import McMaster2022ExperimentOneEnvelope, McMaster2022ExperimentTwoEnvelope, McMaster2022ExperimentThreeEnvelope
from purias_utils.error_modelling_torus.data_utils.data_scripts.bays2016_datasets import VanDenBerg2012ColourEnvelope, VanDenBerg2012OrientationEnvelope, Bays2014OrientationEnvelope, GorgoraptisOrientationEnvelope


#from purias_utils.error_modelling_torus.data_utils.data_scripts.misattribution_data_generators import ZeroSwapMisattributionCheckerSetSizesEnvelope
#from purias_utils.error_modelling_torus.data_utils.data_scripts.misattribution_data_generators import TargetModulatedSwapMisattributionCheckerSetSizesEnvelope



dataset_choices = ['bays2009', 'mcmaster2022_e1_oricue', 'mcmaster2022_e2_dircue', 'mcmaster2022_e1_loccue', 'mcmaster2022_e2_loccue', 'mcmaster2022_e3', 'vandenberg2012_color', 'vandenberg2012_orientation', 'bays2014_orientation', 'gorgoraptis2011_orientation']
dataset_choices += list(map(lambda x: 'schneegans2017_e2_'+x, EXPERIMENT_2_VALID_TASK_TYPES))


def dump_training_indices_to_path(dataset_generator: MultipleSetSizesActivitySetDataGeneratorEnvelopeBase, dest_base_path):
    train_indices = {N: dg.train_indices.cpu().numpy() for N, dg in dataset_generator.data_generators.items()}
    np.save(os.path.join(dest_base_path, 'train_indices.npy'), train_indices)


def load_experimental_data(dataset_name: str, train_indices_seed: Optional[int], train_indices_path: Optional[str], M_batch: int, M_test_per_set_size: Union[int, float], num_repeats: int, data_subselection_args, device, **kwargs) -> MultipleSetSizesActivitySetDataGeneratorEnvelopeBase:

    assert not ((train_indices_seed is not None) and (train_indices_path is not None))

    if train_indices_seed is not None:
        prev_seed = random.randrange(sys.maxsize)
        random.seed(train_indices_seed)

    if dataset_name == 'bays2009':
        dataset_generator = Bays2009MultipleSetSizesEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, participant_id=data_subselection_args.subjects, stimulus_exposure_id=data_subselection_args.stimulus_exposure_id, device = device)
    elif dataset_name == 'mcmaster2022_e1_oricue':
        dataset_generator = McMaster2022ExperimentOneEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='oricue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths, device = device)
    elif dataset_name == 'mcmaster2022_e2_dircue':
        dataset_generator = McMaster2022ExperimentTwoEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='dircue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths, device = device)
    elif dataset_name == 'mcmaster2022_e1_loccue':
        dataset_generator = McMaster2022ExperimentOneEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='loccue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths, device = device)
    elif dataset_name == 'mcmaster2022_e2_loccue':
        dataset_generator = McMaster2022ExperimentTwoEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subtask='loccue', subjects=data_subselection_args.subjects, stim_strengths=data_subselection_args.stim_strengths, device = device)
    elif dataset_name == 'mcmaster2022_e3':
        dataset_generator = McMaster2022ExperimentThreeEnvelope(loc_type=data_subselection_args.loc_type, M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=data_subselection_args.subjects, device = device)
    elif dataset_name == 'vandenberg2012_color':
        dataset_generator = VanDenBerg2012ColourEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=data_subselection_args.subjects, device = device)
    elif dataset_name == 'vandenberg2012_orientation':
        dataset_generator = VanDenBerg2012OrientationEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=data_subselection_args.subjects, device = device)
    elif dataset_name == 'bays2014_orientation':
        dataset_generator = Bays2014OrientationEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=data_subselection_args.subjects, device = device)
    elif dataset_name == 'gorgoraptis2011_orientation':
        dataset_generator = GorgoraptisOrientationEnvelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=data_subselection_args.subjects, device = device)
    elif dataset_name.startswith('schneegans2017_e2'):
        task_type = dataset_name.split('schneegans2017_e2_')[1]
        dataset_generator = Schneegans2017Experiment2Envelope(M_batch=M_batch, M_test=M_test_per_set_size, num_repeats=num_repeats, subjects=data_subselection_args.subjects, task_type=task_type, device = device)
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset. Valid datasets: {dataset_choices}")

    if train_indices_seed is not None:
        random.seed(prev_seed)

    if train_indices_path is not None:
        train_indices = np.load(os.path.join(train_indices_path, f"train_indices.json"), allow_pickle = True).item()

        for N, dg in dataset_generator.data_generators.items():
            dg.set_train_indices(torch.tensor(train_indices[N]))

    return dataset_generator



# load_synthetic_data(dataset_generator, args.synthetic_data_path, args.synthetic_data_code, args.train_examples_discarded_per_set_size)

def load_synthetic_data(dataset_generator: Type[MultipleSetSizesActivitySetDataGeneratorEnvelopeBase], synthetic_data_path, synthetic_data_code):

    data_path = os.path.join(synthetic_data_path, f'synthetic_data_{synthetic_data_code}.npy')
    data = np.load(data_path, allow_pickle=True).item()

    for N, dg in dataset_generator.data_generators.items():

        synthetic_errors = torch.tensor(data['generated_data']['errors'][N]).to(dtype = dg.all_errors.dtype, device = dg.all_errors.device)

        assert dg.all_errors.shape == synthetic_errors.shape        # [Q, M, N], i.e. num models included here, see for example function_augmentation.py in the other repo!!

        dg.all_errors = synthetic_errors

    true_surfaces = data['generated_data']['function_eval_mean']
    true_surfaces_std = data['generated_data'].get('function_eval_std', {k: None for k in true_surfaces.keys()})

    return dataset_generator, true_surfaces, true_surfaces_std


def load_synthetic_data_hierarchical(dataset_generator: Type[MultipleSetSizesActivitySetDataGeneratorEnvelopeBase], synthetic_data_path, synthetic_data_code):
    """
    Only difference to above is how we are extracting the true mean/std surfaces!
    """
    data_path = os.path.join(synthetic_data_path, f'synthetic_data_{synthetic_data_code}.npy')
    data = np.load(data_path, allow_pickle=True).item()

    for N, dg in dataset_generator.data_generators.items():

        synthetic_errors = torch.tensor(data['generated_data']['errors'][N]).to(dtype = dg.all_errors.dtype, device = dg.all_errors.device)

        assert dg.all_errors.shape == synthetic_errors.shape        # [Q, M, N], i.e. num models included here, see for example function_augmentation.py in the other repo!!

        dg.all_errors = synthetic_errors

    true_primary_surfaces = data['generated_data']['function_eval_primary_mean']
    true_primary_surfaces_std = data['generated_data'].get('function_eval_primary_std', {k: None for k in true_primary_surfaces.keys()})

    true_submodel_surfaces = data['generated_data']['function_eval_submodel_mean']
    true_submodel_surfaces_std = data['generated_data'].get('function_eval_submodel_std', {k: None for k in true_submodel_surfaces.keys()})

    true_total_surfaces = data['generated_data']['function_eval_total_mean']
    true_total_surfaces_std = data['generated_data'].get('function_eval_total_std', {k: None for k in true_total_surfaces.keys()})

    return dataset_generator, true_primary_surfaces, true_primary_surfaces_std, true_submodel_surfaces, true_submodel_surfaces_std, true_total_surfaces, true_total_surfaces_std


 