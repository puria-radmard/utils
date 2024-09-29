from __future__ import annotations
import numpy as np
import torch, random
from torch import Tensor as _T

from abc import ABC, abstractmethod

from typing import List, Dict, Type, Optional


class EstimateDataLoaderBase(ABC):

    """
    self.all_deltas of shape [M, N, D] and duplicated to [Q, M, N, D] upon loading
    self.all_errors of shape [Q, M, N], which is already duplicated for real data, but makes it possible to swap out for synthetic data
    self.all_target_zetas of shape [M, N, 1]
    """

    # All set by self.sort_out_M_bullshit
    set_size_to_M_train_each: dict
    M_batch: int
    M_train: int
    M_test: int
    num_test_batches: int
    all_indices: List[int]

    def __init__(self, all_deltas: _T, all_errors: _T, all_target_zetas: _T, M_batch: int, M_test: int, num_repeats: int, device: str) -> None:
        num_examples, self.set_size, self.features = all_deltas.shape
        
        assert tuple(all_errors.shape) == (num_examples, self.set_size)
        assert tuple(all_target_zetas.shape) == (num_examples, self.set_size, 1)

        self.all_deltas = all_deltas
        self.all_target_zetas = all_target_zetas
        self.all_errors = all_errors.unsqueeze(0).repeat(num_repeats, 1, 1)

        M_train_each = all_deltas.shape[0] - M_test
        print(M_train_each, 'training examples and', M_test, 'testing examples for N =', self.set_size)
        self.__dict__.update(self.sort_out_M_bullshit(M_batch, M_train_each, M_test))

        self.num_repeats = num_repeats
        self.device = device

    def new_train_batch(self, *args, **kwargs):
        raise Exception('EstimateDataLoaderBase.new_train_batch deprecated, use iterate_train_batches instead')
        batch_indices = random.sample(self.train_indices, self.M_batch)
        deltas_batch = self.all_deltas[batch_indices].to(self.device)   # [M_batch, D (2)]
        errors_batch = self.all_errors[batch_indices].to(self.device)
        return deltas_batch, errors_batch

    def iterate_train_batches(self, *_, dimensions, shuffle, total: Optional[int] = None):

        if shuffle:
            for t in range(total):
                batch_indices = random.sample(self.train_indices, self.M_batch) if self.M_batch > 0 else self.train_indices
                deltas_batch = self.all_deltas[batch_indices].to(self.device).unsqueeze(0).repeat(self.num_repeats, 1, 1, 1)    # [batch, N, D] -> [Q, batch, N, D]
                errors_batch = self.all_errors[:,batch_indices].to(self.device)                                                 # [batch, N, D]
                yield self.set_size, len(batch_indices), deltas_batch[...,dimensions], errors_batch, batch_indices
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            for i in range(self.num_train_batches):
                batch_indices = self.train_indices[i*self.M_batch: (i+1)*self.M_batch] if self.M_batch > 0 else self.train_indices
                deltas_batch = self.all_deltas[batch_indices].to(self.device).unsqueeze(0).repeat(self.num_repeats, 1, 1, 1)
                errors_batch = self.all_errors[:,batch_indices].to(self.device)
                yield self.set_size, len(batch_indices), deltas_batch[...,dimensions], errors_batch, batch_indices

    def all_test_batches(self, *_, dimensions):
        for i in range(self.num_test_batches):
            indices = self.test_indices[i*self.M_batch: (i+1)*self.M_batch] if self.M_batch > 0 else self.test_indices
            deltas_batch = self.all_deltas[indices].to(self.device).unsqueeze(0).repeat(self.num_repeats, 1, 1, 1)
            yield self.set_size, len(indices), deltas_batch[...,dimensions], self.all_errors[:,indices].to(self.device), indices

    @staticmethod
    def sort_out_M_bullshit(M_batch, M_train, M_test):
        if M_batch > 0:
            num_train_batches = (M_train//M_batch) + (1 if M_test%M_batch!=0 else 0)
            num_test_batches = (M_test//M_batch) + (1 if M_test%M_batch!=0 else 0)
        else:
            num_train_batches = 1
            num_test_batches = 1 if M_test > 0 else 0
        all_indices = list(range(M_train + M_test))
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


class MultipleSetSizesActivitySetDataGeneratorEnvelopeBase(ABC):
    """
    Contains multiple EstimateDataLoaderBase sub-instances for each set size,
    and during training calls upon each one to generate a random set with a 
        PMF scaled by the number of examples for each set size.
    Each instance also has its own test sets, which are all iterated through
    """

    D: int

    def __init__(self, M_batch: int, feature_names: List[str], data_generators: Dict[int, Type[EstimateDataLoaderBase]], device: str):

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
                    iter_N = self.ordered_Ns[(random.random() >= self.selection_cdf).sum()]
                else:
                    iter_N = N
                for batch_info in self.data_generators[iter_N].iterate_train_batches(dimensions = dimensions, shuffle = True, total = 1):
                    yield batch_info
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            if N == None:
                for dg in self.data_generators.values():
                    for ret in dg.iterate_train_batches(shuffle = False, total = None):
                        yield ret
            else:
                for ret in self.data_generators[N].iterate_train_batches(dimensions = dimensions, shuffle = False, total = None):
                    yield ret

    def all_test_batches(self, *_, dimensions):
        "Just go through each one, no random selection"
        for dg in self.data_generators.values():
            for ret in dg.all_test_batches(dimensions = dimensions):
                yield ret



################################################################################################
### Moved to archive: OldGibbsEstimateDataLoader, VMFunctionDataGeneratorBase, VMFunctionDataGeneratorOldGibbs, SingleSetSizeSetDataGenerator, DecoderActivitySetMultipleSetSizesDataGeneratorEnvelope, SingleSetSizeVMFunctionDataGeneratorActivitySet, DecoderActivitySetMultipleSetSizesVMFunctionDataGeneratorActivitySet
################################################################################################

