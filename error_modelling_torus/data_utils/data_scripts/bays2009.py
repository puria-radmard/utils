import torch
import numpy as np
from scipy.io import loadmat

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.data_utils.base import EstimateDataLoaderBase, MultipleSetSizesActivitySetDataGeneratorEnvelopeBase

data_path = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link/experimental_data/bays2009/Exp_data.mat"
data_array: dict = loadmat(data_path)

DELAY_ARRAY = data_array['delay']   #  (7271, 1)                        # subjects 1 to 12
SUBJECT_ARRAY = data_array['subject'] #  (7271, 1)                      # delay 100 to 2000
N_ITEMS_ARRAY = data_array['n_items'] #  (7271, 1)                      # numbers 1 to 6
RESPONSE_ARRAY = data_array['response']    #  (7271, 1)                 # next few are all [-pi, pi]
TARGET_ARRAY = data_array['target']  #  (7271, 1)
NONTARGET_ARRAY = data_array['nontarget']   #  (7271, 5)
TARGET_POS_ARRAY = data_array['target_pos']  #  (7271, 1)
NONTARGET_POS_ARRAY = data_array['nontarget_pos']   #  (7271, 5)
ERROR_ARRAY = data_array['error']   #  (7271, 1)        Response - target, as we would like!

LOCATION_INCREMENT = sorted(np.unique(np.nan_to_num(NONTARGET_POS_ARRAY)))[1]

class Bayes2009SingleSetSize(EstimateDataLoaderBase):

    def __init__(self, N, M_batch, M_test, num_repeats, participant_id, stimulus_exposure_id, device):

        self.device = device

        self.N = int(N)
        trial_N_indexer = (N_ITEMS_ARRAY.squeeze(1) == self.N)
        if participant_id is not None:
            raise Exception(header = 'put in correct checks here please!')
            trial_N_indexer = np.logical_and(trial_N_indexer, SUBJECT_ARRAY.squeeze(1) == participant_id)
        if stimulus_exposure_id is not None:
            raise Exception(header = 'put in correct checks here please!')
            trial_N_indexer = np.logical_and(trial_N_indexer, DELAY_ARRAY.squeeze(1) == stimulus_exposure_id)

        zeta_color = np.concatenate([TARGET_ARRAY[trial_N_indexer], NONTARGET_ARRAY[trial_N_indexer]], 1)[:,:self.N]
        zeta_locat = np.concatenate([TARGET_POS_ARRAY[trial_N_indexer], NONTARGET_POS_ARRAY[trial_N_indexer]], 1)[:,:self.N]
        zeta_locat = ((zeta_locat / LOCATION_INCREMENT) * 2 * torch.pi / 8) - torch.pi


        zetas = np.stack([zeta_locat, zeta_color], 2)
        deltas = rectify_angles(zetas - zetas[:, [0], :])

        all_errors = rectify_angles(RESPONSE_ARRAY[trial_N_indexer] - zeta_color)
        assert (all_errors[:,[0]] == ERROR_ARRAY[trial_N_indexer]).all()

        all_deltas = torch.tensor(deltas).to(device)                    # [M, N, D (2)]
        all_target_zetas = torch.tensor(zeta_color).to(device)          # [M, N, ]
        all_errors = torch.tensor(all_errors).to(device)                # [M, N]

        super().__init__(all_deltas, all_errors, all_target_zetas, M_batch, M_test, num_repeats, device)



class Bays2009MultipleSetSizesEnvelope(MultipleSetSizesActivitySetDataGeneratorEnvelopeBase):

    D = 2

    def __init__(self, *_, M_batch, M_test, num_repeats, participant_id = None, stimulus_exposure_id = None, device = 'cuda'):

        print('No splitting of participants here!')

        data_generators = {
            int(N): Bayes2009SingleSetSize(N, M_batch, M_test, num_repeats, participant_id, stimulus_exposure_id, device)
            for N in np.unique(N_ITEMS_ARRAY).astype(float)
        }

        feature_names = ['location', 'colour']

        super().__init__(M_batch, feature_names, data_generators, device)

# dataset_generator = Bays2009MultipleSetSizesEnvelope(32, 200, device = 'cpu')
# deltas_batch, errors_batch = dataset_generator.new_train_batch()    # [B, N, 2] and [B, N]
# dataset_generator.selection_cdf

