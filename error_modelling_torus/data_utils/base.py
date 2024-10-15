from __future__ import annotations
import numpy as np
import torch, random
from torch import Tensor as _T

from abc import ABC, abstractmethod

from typing import List, Dict, Optional, Union


class EstimateDataLoaderBase(ABC):

    """
    self.all_deltas of shape [M, N, D] in storage, but stacked to [Q, Mbatch/Mtrain/Mtest, N, D] upon loading
    self.all_errors of shape [Q, M, N], which is already duplicated for real data, but makes it possible to swap out for synthetic data
    self.all_target_zetas of shape [M, N, 1]

    self.train_indices is now generated Q times, and batches are drawn separately from each repeat, rather then selecting it once then
        repeating it
    """

    # All set by self.sort_out_M_bullshit
    set_size_to_M_train_each: dict
    train_indices: _T       # [Q, M_train]
    test_indices: _T        # [Q, M_test]

    def __init__(self, all_deltas: _T, all_errors: _T, all_target_zetas: _T, M_batch: int, M_test: Union[int, float], num_repeats: int, device: str) -> None:
        
        num_examples, self.set_size, self.features = all_deltas.shape
        assert tuple(all_errors.shape) == (num_examples, self.set_size)
        assert tuple(all_target_zetas.shape) == (num_examples, self.set_size, 1)

        # This is always started with real data, so all errors will be the same across repeats
        # if synthetic data is loaded, it will not be the same, but in all cases the underlying data is the same! (and therefore so is all_target_zetas)
        self.all_errors = all_errors.unsqueeze(0).repeat(num_repeats, 1, 1)
        self.all_deltas = all_deltas
        self.all_target_zetas = all_target_zetas

        if M_test < 1:
            M_test = int(all_deltas.shape[0] * M_test)
        else:
            assert isinstance(M_test, int) and M_test >= 0

        M_train_each = all_deltas.shape[0] - M_test
        print(M_train_each, 'training examples and', M_test, 'testing examples for N =', self.set_size)

        self.num_repeats = num_repeats
        self.device = device

        M_train = num_examples - M_test
        all_indices = list(range(num_examples))
        train_indices = [random.sample(all_indices, M_train) for _ in range(num_repeats)]
        test_indices = [list(set(all_indices) - set(ti)) for ti in train_indices]
        self.M_batch = M_batch
        self.train_indices = torch.tensor(train_indices)    # [Q, M_train]
        self.test_indices = torch.tensor(test_indices)      # [Q, M_test]

    def get_batch(self, batch_indices: _T):
        assert len(batch_indices.shape) == 2 and batch_indices.shape[0] == self.num_repeats
        deltas_batch = torch.stack([self.all_deltas[bi] for bi in batch_indices], dim = 0)                      # [Q, batch, N, D] -- not the same for each q!!
        errors_batch = torch.stack([self.all_errors[q,bi] for (q, bi) in enumerate(batch_indices)], dim = 0)    # [Q, batch, N]
        return deltas_batch, errors_batch

    def iterate_train_batches(self, *_, dimensions, shuffle, total: Optional[int] = None):
        num_train_examples = self.M_train
        if shuffle:
            M_batch = self.M_batch if self.M_batch > 0 else None
            for _ in range(total):
                batch_indices = torch.stack([self.train_indices[q][torch.randperm(num_train_examples)[:M_batch]] for q in range(self.num_repeats)])
                deltas_batch, errors_batch = self.get_batch(batch_indices)
                yield self.set_size, batch_indices.shape[1], deltas_batch[...,dimensions], errors_batch, batch_indices
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            for i in range(self.num_train_batches):
                import pdb; pdb.set_trace(header = 'check that this works!!')
                batch_indices = torch.stack([self.train_indices[q][i*self.M_batch:(i+1)*self.M_batch] for q in range(self.num_repeats)])
                deltas_batch, errors_batch = self.get_batch(batch_indices)
                yield self.set_size, batch_indices.shape[1], deltas_batch[...,dimensions], errors_batch, batch_indices

    def all_test_batches(self, *_, dimensions):
        for i in range(self.num_test_batches):
            if self.num_test_batches > 1:
                batch_indices = torch.stack([self.test_indices[q][i*self.M_batch:(i+1)*self.M_batch] for q in range(self.num_repeats)])
            else:
                batch_indices = self.test_indices
            deltas_batch, errors_batch = self.get_batch(batch_indices)
            yield self.set_size, batch_indices.shape[1], deltas_batch[...,dimensions], errors_batch, batch_indices

    @property
    def M_test(self) -> int:
        return self.test_indices.shape[-1]

    @property
    def M_train(self) -> int:
        return self.train_indices.shape[-1]

    @property
    def num_train_batches(self) -> int:
        if self.M_batch > 0:
            M_train = self.M_train
            return (M_train//self.M_batch) + (1 if (self.M_train%self.M_batch)!=0 else 0)
        else:
            return 1

    @property
    def num_test_batches(self) -> int:
        if self.M_batch > 0:
            M_test = self.M_test
            return (M_test//self.M_batch) + (1 if (self.M_test%self.M_batch)!=0 else 0)
        else:
            return 1 if self.M_test > 0 else 0

    def set_train_indices(self, train_indices: _T, test_indices: _T):
        assert test_indices.shape == self.test_indices.shape    # this is never changed

        # We may have called self.discard_last_n_training_examples
        assert tuple(train_indices.shape) == (self.num_repeats, train_indices.shape[1])
        train_indices.shape[1] <= self.train_indices.shape[1]

        import pdb; pdb.set_trace(header = 'assert these add up to a valud full indices set on each Q')

        self.train_indices = train_indices
        self.test_indices = test_indices

    def discard_last_n_training_examples(self, n):
        assert n > 0
        self.train_indices = self.train_indices[:,:-n]

    def separate_to_test_and_train(self, quantites: _T, average_over_data = False):
        assert tuple(quantites.shape) == (self.num_repeats, self.M_test + self.M_train)

        training_quantities = quantites.gather(-1, self.train_indices.to(quantites.device))
        testing_quantities = quantites.gather(-1, self.test_indices.to(quantites.device))

        if average_over_data:
            training_quantities = training_quantities.mean(-1)
            testing_quantities = testing_quantities.mean(-1)

        return training_quantities, testing_quantities


class MultipleSetSizesActivitySetDataGeneratorEnvelopeBase(ABC):
    """
    Contains multiple EstimateDataLoaderBase sub-instances for each set size,
    and during training calls upon each one to generate a random set with a 
        PMF scaled by the number of examples for each set size.
    Each instance also has its own test sets, which are all iterated through
    """

    D: int

    def __init__(self, M_batch: int, feature_names: List[str], data_generators: Dict[int, EstimateDataLoaderBase], device: str):

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

    def iterate_train_batches(self, *_, dimensions, shuffle, total = None, N = None):
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

    def all_test_batches(self, *_, dimensions, N = None):
        "Just go through each one, no random selection"
        if N == None:
            for dg in self.data_generators.values():
                for ret in dg.all_test_batches(dimensions = dimensions):
                    yield ret
        else:
            for ss, dg in self.data_generators.items():
                if ss == N:
                    for ret in dg.all_test_batches(dimensions = dimensions):
                        yield ret
                else:
                    continue



################################################################################################
### Moved to archive: OldGibbsEstimateDataLoader, VMFunctionDataGeneratorBase, VMFunctionDataGeneratorOldGibbs, SingleSetSizeSetDataGenerator, DecoderActivitySetMultipleSetSizesDataGeneratorEnvelope, SingleSetSizeVMFunctionDataGeneratorActivitySet, DecoderActivitySetMultipleSetSizesVMFunctionDataGeneratorActivitySet
################################################################################################

