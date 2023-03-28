import torch
from torch import Tensor as T

class StimulusBasisFunctionSetBase:
    """
    This translates to the full set of m(s) in the original DDPC paper
    At the moment, this only makes sense via the vector subclass, so focus on that
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def multiplicative_stimulus(self, weights: T):
        """
        Combine many bases to generate this multivalued function
        """
        raise NotImplementedError

    def full_basis(self):
        """
        Present all bases together
        """
        raise NotImplementedError


class VectorisedStimulusBasisFunctionSet(StimulusBasisFunctionSetBase):
    """
    This is the case where each m(s) is a vector, such as in the first example of the original DDPC paper

    basis_vectors is of shape [B, N], where:
        B is the number of bins in the discretised domain, i.e. 51 bins around a circle
        N is the number of basis functions, i.e. 2 in the original paper
    """

    def __init__(self, basis_vectors: T) -> None:
        
        self.basis_vectors = basis_vectors
        self.B, self.N = basis_vectors.shape

        super(VectorisedStimulusBasisFunctionSet, self).__init__(dimension = 1)

    def multiplicative_stimulus(self, weights: T):
        "weights of shape [batch, B]. output of shape [batch, B]"
        assert len(weights.shape) == 2
        assert weights.shape[-1] == self.N
        return weights @ self.full_basis().T

    def full_basis(self):
        return self.basis_vectors
