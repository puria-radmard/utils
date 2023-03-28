import torch
import unittest

from purias_utils.rnn.layers.base import WeightLayer, WeightLayerBase, AbsWeightLayer
from purias_utils.rnn.layers.dales import BinaryMaskRecurrent, BinaryMaskForward


class NegativeWeightLayer(WeightLayer):
    @property
    def masked_weight(self):
        return  - self.base_matrix


def generate_standard_test_case(*shape):

    raw_matrix = torch.randn(*shape)

    base_layer = WeightLayerBase(raw_matrix)
    negative_layer = NegativeWeightLayer(base_layer)
    abs_layer = AbsWeightLayer(negative_layer)

    return raw_matrix, base_layer, negative_layer, abs_layer


class TestWeightLayer(unittest.TestCase):

    def test_propagation(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 50)

        self.assertTrue(torch.all(base_layer.raw_matrix == raw_matrix), "Raw matrix not recovered")
        self.assertTrue(torch.all(negative_layer.raw_matrix == raw_matrix), "Raw matrix not recovered")
        self.assertTrue(torch.all(abs_layer.raw_matrix == raw_matrix), "Raw matrix not recovered")

        self.assertTrue(torch.all(base_layer.masked_weight == raw_matrix), "Masked weight not propagated")
        self.assertTrue(torch.all(negative_layer.base_matrix == base_layer.masked_weight), "Masked weight not propagated")
        self.assertTrue(torch.all(abs_layer.base_matrix == negative_layer.masked_weight), "Masked weight not propagated")
    
    def test_setter(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 50)
        raw_matrix_2 = torch.randn(50, 50)

        abs_layer.raw_matrix = raw_matrix_2
        
        self.assertTrue(torch.all(base_layer.raw_matrix == raw_matrix_2), "Setting raw_matrix did not propagate")
        self.assertTrue(torch.all(negative_layer.raw_matrix == raw_matrix_2), "Setting raw_matrix did not propagate")
        self.assertTrue(torch.all(abs_layer.raw_matrix == raw_matrix_2), "Setting raw_matrix did not propagate")


    def test_conversion(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 9)

        self.assertTrue(torch.all(abs_layer.masked_weight == negative_layer.masked_weight.abs()), "Abs layer just doesn't work!")
        self.assertTrue(torch.all(negative_layer.masked_weight == - raw_matrix), "Negative layer just doesn't work!")

 
    def test_feedforward_mitosis(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 9)

        split = abs_layer.feedforward_mitosis(4)

        self.assertTrue(tuple(base_layer.shape) == (50, 9), "Mitosis rewrites in place")
        self.assertTrue(tuple(split._base_matrix.base_matrix.shape) == (51, 9), "Mitosis does not propagate back")
        self.assertTrue(tuple(split._base_matrix.base_matrix.shape) == (51, 9), "Mitosis does not propagate back")
        self.assertTrue(torch.all(split.masked_weight[4,:] == split.masked_weight[5,:]), "Mitosis does not propagate back")


    def test_recurrent_mitosis(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 50)

        split = abs_layer.recurrent_mitosis(4)
        self.assertTrue(torch.all(split.masked_weight[4,:] == split.masked_weight[5,:]), "Mitosis does not propagate back")
        self.assertTrue(torch.all(split.masked_weight[:,4] == split.masked_weight[:,5]), "Mitosis does not propagate back")


    def test_dales_recurrent(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 50)
        dales_layer = BinaryMaskRecurrent(abs_layer, exc_indexes=list(range(30)))

        mask = torch.ones_like(raw_matrix)
        mask[:,30:] = -1.0
        test_weight = raw_matrix.abs() * mask

        self.assertTrue(torch.all(test_weight == dales_layer.masked_weight), "Dales doesn't work")


    def test_dales_indexes(self):
        raw_matrix, base_layer, negative_layer, abs_layer = generate_standard_test_case(50, 50)
        dales_layer = BinaryMaskRecurrent(abs_layer, exc_indexes=list(range(30)))

        dales_layer = dales_layer.recurrent_mitosis(5)
        self.assertTrue(dales_layer.exc_indexes == set(range(31)))

        dales_layer = dales_layer.recurrent_mitosis(38)
        self.assertTrue(dales_layer.inh_indexes == set(range(31, 52)))


    def test_index_excemption(self):
        W_rec = BinaryMaskForward(
            torch.arange(100).float().unsqueeze(1).repeat(1, 9),
            exc_indexes=range(80), 
            inh_mask=-1, 
            exempt_indices=[0, 1, 2]
        )
        output = W_rec(torch.eye(9))

        self.assertTrue(torch.all(W_rec.masked_weight[:,:3].mean(-1) == torch.arange(100)))
        self.assertTrue(torch.all(W_rec.masked_weight[:80,3:].mean(-1) == torch.arange(80)))
        self.assertTrue(torch.all(W_rec.masked_weight[80:,3:].mean(-1) == -torch.arange(80,100)))



if __name__ == '__main__':

    unittest.main()

