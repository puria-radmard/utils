from __future__ import annotations

from geometry.topology import *
from geometry.utils import repeat_points_dimension, immerse_coordinates


class Embedding:
    """
    Defines an embedding from a manifold into an ambient space, assumed Euclidean.

    For an n dimensional ambient, requires an m < n dimensional manifold.

    Each dimension is acted on by a 1 dimensional map
    """

    def __init__(self, phi: Callable[[Points], T], domain: Manifold, codomain_dim: int):

        self.phi = phi
        self.domain = domain
        self.codomain_dim = codomain_dim

        assert self.domain.dim <= codomain_dim

    def embed_points(self, ps: Points):
        assert ps.dim == self.domain.dim
        return check_and_map(ps, self.domain, self.phi)

    def compose_multiply(self, other: Embedding, multiplication_dim: int, new_domain: Manifold = None) -> Embedding:
        """
            Multiply own embedding with some dimensions of other embedding,
                acting overall on the cartesian product of the two manifolds,
                and embedding into the product Euclidean space

            TODO: This is very rough - see torus.py
                There is a better way to do this, both mathematically and in implementation
        """
        if new_domain is None:
            new_domain = self.domain.cartesian_product(other.domain)
        assert new_domain.dim == (self.domain.dim + other.domain.dim)
        assert 0 <= multiplication_dim < other.domain.dim
        
        def new_phi(ps: Points):
        
            other_embedding = other.embed_points(ps.get_dim(0, other.domain.dim))
            own_embedding = self.embed_points(ps.get_dim(other.domain.dim, other.domain.dim + self.domain.dim))

            # Repeat along dimensions to be multiplied by own embedding
            other_multiplied_embedding = repeat_points_dimension(
                other_embedding, repeat_dim = multiplication_dim, repeat_times = self.codomain_dim
            )

            # Make dimensions not being multipled just one
            self_multipled_embedding = immerse_coordinates(
                own_embedding, final_dim = other.codomain_dim + self.codomain_dim - 1, 
                used_dimensions = list(range(multiplication_dim, multiplication_dim + self.codomain_dim)),
                ambient_coords = 1.0
            )
            
            return Points(other_multiplied_embedding.coords * self_multipled_embedding.coords)

        return Embedding(
            phi = new_phi, domain = new_domain, codomain_dim = self.codomain_dim + other.codomain_dim
        )
            

class IdentityEmbedding(Embedding):
    """
        A quick fix to some tensor shape issues.
        
        TODO: Write docstring and standardise self.embed_basis_curves as a separate Embedding method
            Also, make evaluate_unweighted_basis_coordinates a new class rather than of the tangent fields class
    """
    def __init__(self, domain: Manifold):
        super().__init__(phi=self._phi, domain=domain, codomain_dim=domain.dim)

    def _phi(self, ps: Points):
        output = torch.zeros(len(ps), self.codomain_dim)
        if isinstance(ps, BatchedPoints):
            for i in range(self.codomain_dim):
                output[:, i] = ps.coords[:, :, i].sum(-1)
        else:
            for i in range(self.codomain_dim):
                output[:, i] = ps.coords[:, i]
        return Points(output)

