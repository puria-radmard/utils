from __future__ import annotations
import numpy as np
import torch, random
from torch import Tensor as _T

# from multitask_wm.analysis_scnd.utils import load_activity_sets

from purias_utils.multiitem_working_memory.util.circle_utils import rectify_angles
from purias_utils.error_modelling_torus.non_parametric_error_model.generative_model import NonParametricSwapErrorsGenerativeModel

from abc import ABC, abstractmethod


class EstimateDataLoaderBase:

    set_size_to_M_train_each: dict

    def new_train_batch(self, *args, **kwargs):
        raise Exception('EstimateDataLoaderBase.new_train_batch deprecated, use iterate_train_batches instead')
        batch_indices = random.sample(self.train_indices, self.M_batch)
        deltas_batch = self.all_deltas[batch_indices].to(self.device)   # [M_batch, D (2)]
        errors_batch = self.all_errors[batch_indices].to(self.device)
        return deltas_batch, errors_batch

    def iterate_train_batches(self, *_, dimensions, shuffle, total = None, return_indices = False):
        if shuffle:
            for t in range(total):
                if self.M_batch > 0:
                    batch_indices = random.sample(self.train_indices, self.M_batch)
                else:
                    batch_indices = self.train_indices
                deltas_batch = self.all_deltas[batch_indices].to(self.device)
                errors_batch = self.all_errors[batch_indices].to(self.device)
                if return_indices:
                    yield deltas_batch[...,dimensions], errors_batch, batch_indices
                else:
                    yield deltas_batch[...,dimensions], errors_batch
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            for i in range(self.num_train_batches):
                if self.M_batch > 0:
                    indices = self.train_indices[i*self.M_batch: (i+1)*self.M_batch]
                else:
                    indices = self.train_indices[:]
                deltas_batch = self.all_deltas[indices].to(self.device)
                errors_batch = self.all_errors[indices].to(self.device)
                if return_indices:
                    yield deltas_batch[...,dimensions], errors_batch, indices
                else:
                    yield deltas_batch[...,dimensions], errors_batch

    def all_test_batches(self, *_, dimensions, return_indices=False):
        for i in range(self.num_test_batches):
            if self.M_batch > 0:
                indices = self.test_indices[i*self.M_batch: (i+1)*self.M_batch]
            else:
                indices = self.test_indices
            deltas_batch = self.all_deltas[indices].to(self.device)
            if return_indices:
                yield (
                    deltas_batch[...,dimensions],
                    self.all_errors[indices].to(self.device),
                    indices
                )
            else:
                yield (
                    deltas_batch[...,dimensions],
                    self.all_errors[indices].to(self.device)
                )

    @staticmethod
    def sort_out_M_bullshit(M_batch, M_train, M_test):
        if M_batch > 0:
            num_train_batches = (M_train//M_batch) + (1 if M_test%M_batch!=0 else 0)
            num_test_batches = (M_test//M_batch) + (1 if M_test%M_batch!=0 else 0)
        else:
            num_train_batches = 1
            num_test_batches = 1 if M_test > 0 else 0
        all_indices = range(M_train + M_test)
        train_indices = random.sample(all_indices, M_train)
        test_indices = list(set(all_indices) - set(train_indices))
        return {
            'M_train': M_train,
            'M_batch': M_batch,
            'num_train_batches': num_train_batches,
            'M_train': M_train,
            'M_test': M_test,
            'num_test_batches': num_test_batches,
            'all_indices': all_indices,
            'train_indices': train_indices,
            'test_indices': test_indices,
        }

    def steal_M_bullshit_from_another_generator(self, other_generator: EstimateDataLoaderBase):
        self.__dict__.update({
            'M_train': other_generator.M_train,
            'M_batch': other_generator.M_batch,
            'num_train_batches': other_generator.num_train_batches,
            'M_train': other_generator.M_train,
            'M_test': other_generator.M_test,
            'num_test_batches': other_generator.num_test_batches,
            'all_indices': other_generator.all_indices,
            'train_indices': other_generator.train_indices,
            'test_indices': other_generator.test_indices,
        })

    def set_train_indices(self, train_indices: list[int]):
        tsi = list(set(self.all_indices) - set(train_indices))
        self.train_indices = train_indices
        self.test_indices = tsi

        assert len(self.train_indices) == self.M_train
        assert len(self.test_indices) == self.M_test

    def discard_last_n_training_examples(self, n):
        if n > 0:
            self.M_train -= n
            self.train_indices = self.train_indices[:-n]
            if self.M_batch > 0:
                self.num_train_batches = (self.M_train//self.M_batch) + (1 if (self.M_test + n)%self.M_batch!=0 else 0)


class MultipleSetSizesActivitySetDataGeneratorEnvelopeBase(EstimateDataLoaderBase, ABC):
    """
    __init__ needs to be preceded by something that set self.data_generators

    Contains multiple EstimateDataLoaderBase sub-instances for each set size,
    and during training calls upon each one to generate a random set with a 
        PMF scaled by the number of examples for each set size.
    Each instance also has its own test sets, which are all iterated through
    """

    D: int

    def __init__(self, M_batch: int, feature_names: list, data_generators: dict, device: str):

        self.device = device
        self.M_batch = M_batch
        self.feature_names = feature_names
        self.data_generators = data_generators

        self.ordered_Ns, M_trains = [], []
        for N, v in self.data_generators.items():
            self.ordered_Ns.append(N)
            M_trains.append(v.M_train)

        self.selection_pmf = np.array([M_train / sum(M_trains) for M_train in M_trains])
        self.selection_cdf = self.selection_pmf.cumsum()

        self.set_size_to_M_train_each = {N: v.M_train for N, v in self.data_generators.items()}

    def new_train_batch(self, N=None):
        raise Exception('EstimateDataLoaderBase.new_train_batch deprecated, use iterate_train_batches instead')
        if N==None:
            u = random.random()
            N_idx = (u >= self.selection_cdf).sum()
            N = self.ordered_Ns[N_idx]
        return self.data_generators[N].new_train_batch()

    def iterate_train_batches(self, *_, dimensions, shuffle, total = None, return_indices = False, N = None):
        if shuffle:
            for t in range(total):
                if N==None:
                    u = random.random()
                    N_idx = (u >= self.selection_cdf).sum()
                    iter_N = self.ordered_Ns[N_idx]
                else:
                    iter_N = N
                for batch_info in self.data_generators[iter_N].iterate_train_batches(dimensions = dimensions, shuffle = True, total = 1, return_indices = return_indices):
                    yield batch_info
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            if N == None:
                for dg in self.data_generators.values():
                    for ret in dg.iterate_train_batches(shuffle = False, total = None, return_indices = return_indices):
                        yield ret
            else:
                for ret in self.data_generators[N].iterate_train_batches(dimensions = dimensions, shuffle = False, total = None, return_indices = return_indices):
                    yield ret

    def all_test_batches(self, *_, dimensions):
        "Just go through each one, no random selection"
        for dg in self.data_generators.values():
            for ret in dg.all_test_batches(dimensions = dimensions):
                yield ret


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
                estimates = data_dict['samples'].squeeze(0).unsqueeze(-1)   # [I (1 for sure), M] -> [M]
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



################################################################################################
### Moved to archive: OldGibbsEstimateDataLoader, VMFunctionDataGeneratorBase, VMFunctionDataGeneratorOldGibbs, SingleSetSizeSetDataGenerator, DecoderActivitySetMultipleSetSizesDataGeneratorEnvelope, SingleSetSizeVMFunctionDataGeneratorActivitySet, DecoderActivitySetMultipleSetSizesVMFunctionDataGeneratorActivitySet
################################################################################################

