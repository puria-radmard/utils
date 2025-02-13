from __future__ import annotations
import numpy as np
import torch, random
from torch import Tensor as _T

from abc import ABC
from functools import lru_cache

from typing import List, Dict, Optional, Union, Any, Set, Tuple


class EstimateDataLoaderBase(ABC):

    """
    NOTE: num_repeats are different folds of the data! i.e. train indicies are reselected for each repeat

    self.all_deltas of shape [M, N, D] in storage, but stacked to [Q, Mbatch/Mtrain/Mtest, N, D] upon loading
    self.all_errors of shape [Q, M, N], which is already duplicated for real data, but makes it possible to swap out for synthetic data
    self.all_target_zetas of shape [M, N, 1]

    self.all_metadata maps from the label category to a list of labels, e.g.:
        {'subject': [0, 2, 6, ...], 'condition': [med, low, low, ...]}
    This is inverted to self.all_metadata_inverted

    self.train_indices is now generated Q times, and batches are drawn separately from each repeat, rather then selecting it once then
        repeating it
    """

    # All set by self.sort_out_M_bullshit
    set_size_to_M_train_each: dict
    train_indices: _T       # [Q, M_train]
    test_indices: _T        # [Q, M_test]

    def __init__(self, all_deltas: _T, all_errors: _T, all_target_zetas: _T, all_metadata: Dict[str, List[Any]], M_batch: int, M_test: Union[int, float], num_repeats: int, device: str) -> None:
        
        num_examples, self.set_size, self.features = all_deltas.shape
        assert tuple(all_errors.shape) == (num_examples, self.set_size), all_errors.shape
        assert tuple(all_target_zetas.shape) == (num_examples, self.set_size, 1), all_target_zetas.shape
        assert all(len(v) == num_examples for v in all_metadata.values())

        # This is always started with real data, so all errors will be the same across repeats
        # if synthetic data is loaded, it will not be the same, but in all cases the underlying data is the same! (and therefore so is all_target_zetas)
        self.all_errors = all_errors.unsqueeze(0).repeat(num_repeats, 1, 1)
        self.all_deltas = all_deltas
        self.all_target_zetas = all_target_zetas
        self.all_metadata = all_metadata
        assert None not in all_metadata.keys()
        self.all_metadata_inverted = self.invert_metadata(all_metadata)

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

    @staticmethod
    def invert_metadata(metadata_dict: Dict[str, List[Any]]) -> Dict[str, Dict[Any, Set[int]]]:
        """
        Input: {info_name: [list of length dataset size]}
        Output: {info_name: {unique_values: [lists which add up to length dataset size for each info_name]}}
        """
        output_dict = {k: {} for k in metadata_dict.keys()}
        for info_name, info_dataset in metadata_dict.items():
            unq, unq_inv, unq_cnt = np.unique(info_dataset, return_inverse=True, return_counts=True)
            indices = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
            output_dict[info_name] = {unique_value: sorted(idxs) for unique_value, idxs in zip(unq, indices)}
        return output_dict

    @lru_cache(1024)
    def limit_available_indices(self, metadata_selection_key: Optional[str], metadata_selection_value: Optional[Any], test: bool = False) -> _T:
        """
        Output a list of Q different indices tensors (1D each)
        if M_test > 0, these aren't always the same size!

        Somewhat wasteful to do this everytime so we make an lru cache for it

        We also assume that the counts will be the same (so we can stack)
        For now, this means we can't have a test set at all
        XXX: Make test-train splits necessarily require conservation of proportions of metadata we care about
        """
        available_indices_tensor = self.test_indices if test else self.train_indices            # [Q, M_total]

        if metadata_selection_key is None:
            assert metadata_selection_value is None
            return available_indices_tensor
        
        if self.M_test > 0:
            raise TypeError('Highly do not recommend cross-validation data with the hierarchical model right now - please instead set M_batch = 0 and iterate over possible values manually in training loop')
        
        available_indices_per_repeat = [
            torch.tensor(list(set(ait.tolist()).intersection(self.all_metadata_inverted[metadata_selection_key][metadata_selection_value])))
            for ait in available_indices_tensor
        ]  # [Q of [M_total]]

        return torch.stack(available_indices_per_repeat)    # M_totals are all the same if self.M_test = 0

    @lru_cache(1024)
    def calculate_metadata_proportions(self, metadata_selection_key: str):
        """
        Calculate cdfs over values of a metadata field -> easy selection
        """
        if self.M_test > 0:
            raise TypeError('Highly do not recommend cross-validation data with the hierarchical model right now - please instead set M_batch = 0 and iterate over possible values manually in training loop')
        possible_metadata_values = list(self.all_metadata_inverted[metadata_selection_key].keys())
        possible_metadata_values_counts = list(map(lambda x: self.limit_available_indices(metadata_selection_key, x, False).shape[-1], possible_metadata_values))
        possible_metadata_values_proportions = torch.tensor(possible_metadata_values_counts) / sum(possible_metadata_values_counts)
        possible_metadata_values_proportions = possible_metadata_values_proportions.cumsum(0)
        return possible_metadata_values, possible_metadata_values_proportions

    @staticmethod
    def choose_from_cdf(options: List[Any], cdf: _T) -> Any:
        return options[(random.random() >= cdf).sum()]

    def get_batch(self, batch_indices: _T):
        assert len(batch_indices.shape) == 2 and batch_indices.shape[0] == self.num_repeats
        deltas_batch = torch.stack([self.all_deltas[bi] for bi in batch_indices], dim = 0)                      # [Q, batch, N, D] -- not the same for each q!!
        errors_batch = torch.stack([self.all_errors[q,bi] for (q, bi) in enumerate(batch_indices)], dim = 0)    # [Q, batch, N]
        metadata = {k: torch.tensor([[v[i] for i in batch_indices_rep] for batch_indices_rep in batch_indices]) for k, v in self.all_metadata.items()}
        return deltas_batch, errors_batch, metadata

    def select_metadata_selection_value(self, *_, metadata_selection_key: Optional[str] = None, metadata_selection_value: Optional[Any] = None, test: bool = False):
        if metadata_selection_key == None:
            assert metadata_selection_value == None
            return None
        
        elif metadata_selection_value is not None:
            return metadata_selection_value

        else:
            possible_metadata_values, possible_metadata_values_proportions = self.calculate_metadata_proportions(metadata_selection_key)
            iter_metadata_selection_value = self.choose_from_cdf(possible_metadata_values, possible_metadata_values_proportions)
            return iter_metadata_selection_value

    def iterate_train_batches(self, *_, dimensions, shuffle, total: Optional[int] = None, metadata_selection_key: Optional[str] = None, metadata_selection_value: Optional[Any] = None):
        if shuffle:
            M_batch = self.M_batch if self.M_batch > 0 else None
            for _ in range(total):
                iter_metadata_selection_value = self.select_metadata_selection_value(metadata_selection_key=metadata_selection_key, metadata_selection_value = metadata_selection_value, test = False)
                available_indices = self.limit_available_indices(metadata_selection_key=metadata_selection_key, metadata_selection_value = iter_metadata_selection_value, test = False)
                num_train_examples = available_indices.shape[-1]
                batch_indices = torch.stack([available_indices[q][torch.randperm(num_train_examples)[:M_batch]] for q in range(self.num_repeats)])
                deltas_batch, errors_batch, metadata_batch = self.get_batch(batch_indices)
                yield self.set_size, batch_indices.shape[1], deltas_batch[...,dimensions], errors_batch, metadata_batch, batch_indices
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            available_indices = self.limit_available_indices(metadata_selection_key=metadata_selection_key, metadata_selection_value = metadata_selection_value, test = False)
            for i in range(self.num_train_batches):
                import pdb; pdb.set_trace(header = 'check that this works!!')
                batch_indices = torch.stack([available_indices[q][i*self.M_batch:(i+1)*self.M_batch] for q in range(self.num_repeats)])
                deltas_batch, errors_batch, metadata_batch = self.get_batch(batch_indices)
                yield self.set_size, batch_indices.shape[1], deltas_batch[...,dimensions], errors_batch, metadata_batch, batch_indices

    def all_test_batches(self, *_, dimensions, metadata_selection_key: Optional[str] = None, metadata_selection_value: Optional[Any] = None):
        available_indices = self.limit_available_indices(metadata_selection_key=metadata_selection_key, metadata_selection_value = metadata_selection_value, test = True)
        for i in range(self.num_test_batches):
            if self.num_test_batches > 1:
                batch_indices = torch.stack([available_indices[q][i*self.M_batch:(i+1)*self.M_batch] for q in range(self.num_repeats)])
            else:
                batch_indices = available_indices
            deltas_batch, errors_batch, metadata_batch = self.get_batch(batch_indices)
            yield self.set_size, batch_indices.shape[1], deltas_batch[...,dimensions], errors_batch, metadata_batch, batch_indices

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

        import pdb; pdb.set_trace(header = 'assert these add up to a valid full indices set on each Q')

        self.train_indices = train_indices
        self.test_indices = test_indices
        self.limit_available_indices.cache_clear()
        self.calculate_metadata_proportions.cache_clear()

    def discard_last_n_training_examples(self, n):
        assert n > 0
        self.train_indices = self.train_indices[:,:-n]
        self.limit_available_indices.cache_clear()
        self.calculate_metadata_proportions.cache_clear()
        return self.train_indices.shape
    
    def drop_data_by_metadata_value(self, metadata_selection_key: str, metadata_selection_value: Any):
        if self.train_indices.shape[0] > 1:
            assert self.M_test == 0, "Have not implemented drop_data_by_metadata_value for multiple models when M_test > 0 yet!"
        affected_indices = self.all_metadata_inverted[metadata_selection_key][metadata_selection_value]
        mask = ~torch.stack([ai == self.train_indices for ai in affected_indices]).sum(0).bool()
        self.train_indices = torch.stack([self.train_indices[q,mask[q]] for q in range(self.train_indices.shape[0])])

    def duplicate_underlying_data(self, num_duplicates: int):

        M_tot_prev = self.all_deltas.shape[0]

        self.all_errors = self.all_errors.repeat(1, num_duplicates, 1)
        self.all_target_zetas = self.all_target_zetas.repeat(num_duplicates, 1, 1)
        self.all_deltas = self.all_deltas.repeat(num_duplicates, 1, 1)

        new_metadata = {}
        for k, v in self.all_metadata.items():
            assert isinstance(v, list)
            new_metadata[k] = v * num_duplicates

        new_train_indices = self.train_indices.clone()
        new_test_indices = self.test_indices.clone()

        for nd in range(num_duplicates - 1):
            new_train_indices = torch.concat([new_train_indices, self.train_indices + M_tot_prev * (nd + 1)], 1)    # [Q, M_train*num_duplicates]
            new_test_indices = torch.concat([new_test_indices, self.test_indices + M_tot_prev * (nd + 1)], 1)    # [Q, M_test*num_duplicates]

        self.train_indices = new_train_indices
        self.test_indices = new_test_indices

        self.all_metadata = new_metadata
        self.all_metadata_inverted = self.invert_metadata(new_metadata)

        self.limit_available_indices.cache_clear()
        self.calculate_metadata_proportions.cache_clear()

    def separate_to_test_and_train(self, quantites: _T, average_over_data = False):

        num_quantities = quantites.shape[1]
        assert tuple(quantites.shape) == (self.num_repeats, num_quantities)
        assert num_quantities >= self.M_test + self.M_train, f"Quantities provided to separate_to_test_and_train has shape {quantites.shape}, implying {num_quantities} items. This is less than the number of total datapoints in the dataset ({self.M_test + self.M_train})"

        training_quantities = quantites.gather(-1, self.train_indices.to(quantites.device))
        testing_quantities = quantites.gather(-1, self.test_indices.to(quantites.device))
        
        if num_quantities > self.M_test + self.M_train:
            print("Should implement something to take care of non train, non test quantities in separate_to_test_and_train")

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
        
        for N, dg in self.data_generators.items():
            self.ordered_Ns.append(N)
            M_trains.append(dg.M_train)

        self.selection_pmf = np.array([M_train / sum(M_trains) for M_train in M_trains])
        self.selection_cdf = self.selection_pmf.cumsum()
        self.set_size_to_M_train_each = {N: v.M_train for N, v in self.data_generators.items()}
    
    def drop_data_by_metadata_value(self, metadata_selection_key: str, metadata_selection_value: Any):
        for dg in self.data_generators.values():
            dg.drop_data_by_metadata_value(metadata_selection_key, metadata_selection_value)

    def duplicate_underlying_data(self, num_duplicates: int):
        for dg in self.data_generators.values():
            dg.duplicate_underlying_data(num_duplicates)

    def iterate_train_batches(self, *_, dimensions, shuffle, total = None, N = None, metadata_selection_key=None, metadata_selection_value=None):
        if shuffle:
            for t in range(total):
                iter_N = self.ordered_Ns[(random.random() >= self.selection_cdf).sum()] if N==None else N
                for batch_info in self.data_generators[iter_N].iterate_train_batches(
                    dimensions = dimensions, shuffle = True, total = 1, metadata_selection_key = metadata_selection_key, metadata_selection_value = metadata_selection_value
                ):
                    yield batch_info
        else:
            assert total == None, 'Cannot define total number of training batches if not shuffling training set'
            if N == None:
                for dg in self.data_generators.values():
                    for ret in dg.iterate_train_batches(shuffle = False, total = None, metadata_selection_key = metadata_selection_key, metadata_selection_value = metadata_selection_value):
                        yield ret
            else:
                for ret in self.data_generators[N].iterate_train_batches(
                    dimensions = dimensions, shuffle = False, total = None, metadata_selection_key = metadata_selection_key, metadata_selection_value = metadata_selection_value
                ):
                    yield ret

    def all_test_batches(self, *_, dimensions, N = None):
        "Just go through each one, no random selection"
        if N == None:
            for dg in self.data_generators.values():
                for ret in dg.all_test_batches(dimensions = dimensions):
                    yield ret
        else:
            for ret in self.data_generators[N].all_test_batches(dimensions = dimensions):
                yield ret

################################################################################################
### Moved to archive: OldGibbsEstimateDataLoader, VMFunctionDataGeneratorBase, VMFunctionDataGeneratorOldGibbs, SingleSetSizeSetDataGenerator, DecoderActivitySetMultipleSetSizesDataGeneratorEnvelope, SingleSetSizeVMFunctionDataGeneratorActivitySet, DecoderActivitySetMultipleSetSizesVMFunctionDataGeneratorActivitySet
################################################################################################

