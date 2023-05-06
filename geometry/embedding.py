from __future__ import annotations

from purias_utils.geometry.topology import *
from purias_utils.geometry.utils import repeat_points_dimension, immerse_coordinates

from sklearn.decomposition import PCA

import numpy as np

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

        self._embedded_grid_points = None     # Don't generate until needed
        self._pca = None                      # Also don't generate until needed

    def embed_points(self, ps: Points):
        assert ps.dim == self.domain.dim
        return check_and_map(ps, self.domain, self.phi)
    
    @property
    def embedded_grid_points(self):
        if self._embedded_grid_points is None:
            self._embedded_grid_points = self.embed_points(self.domain.grid_points)
        return self._embedded_grid_points

    def supplement_pca(self, new_data):
        new_data =torch.tensor(new_data)
        full_data = torch.cat([self.embedded_grid_points.coords, new_data])
        self._pca = PCA(n_components=3).fit(full_data)

    def pca(self, obj):
        "Fit PCA to embedded points of manifold (if not done already), then transform obj"

        if isinstance(obj, Points):
            obj = obj.coords.detach().numpy()
        elif isinstance(obj, T):
            obj = obj.detach().numpy()

        # First no PCA case
        if self.codomain_dim == 3:
            assert obj.shape[-1] == 3
            return obj

        # Second no PCA case, add a new dimension
        elif self.codomain_dim == 2:
            assert obj.shape[-1] == 2
            extra_zeros = np.zeros([obj.shape[0], 1])
            return np.hstack([obj, extra_zeros])

        # Actual PCA case
        elif self._pca is None:
            self._pca = PCA(n_components=3).fit(self.embedded_grid_points.coords.detach().numpy())

        return self._pca.transform(obj)


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

    def project_randomly(self, target_dim: int):
        return RandomOrthogonalEmbedding(
            phi = self.phi,
            domain = self.domain,
            initial_embedding_codomain_dim = self.codomain_dim,
            target_embedding_codomain_dim = target_dim
        )

            

class IdentityEmbedding(Embedding):
    """
        A quick fix to some tensor shape issues.
        
        TODO: Write docstring and standardise self.embed_basis_curves as a separate Embedding method
            Also, make evaluate_unweighted_basis_coordinates a new class rather than of the tangent fields class
    """
    def __init__(self, domain: Manifold):
        raise TypeError # Not actually a continuous embedding!
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


class RandomOrthogonalEmbedding(Embedding):
    """
        This embeds a manifold in a given space, then multiplies all embeddings by
        a random orthogonal matrix into some target dimensionality Euclidean space

        phi embeds in d1 dimensions
        domain is the original manifold as before
        initial_embedding_codomain_dim is d1
        target_embedding_codomain_dim is d, the final (target) dimensionality 
            of the Euclidean space into which we embed 
    """

    def __init__(self, phi: Callable[[Points], T], domain: Manifold, initial_embedding_codomain_dim: int, target_embedding_codomain_dim: int):
        super().__init__(self.phi_then_project, domain, codomain_dim=target_embedding_codomain_dim)

        assert target_embedding_codomain_dim >= initial_embedding_codomain_dim

        self.initial_phi = phi
        self.initial_embedding_codomain_dim = initial_embedding_codomain_dim
        self.extra_dims = target_embedding_codomain_dim - initial_embedding_codomain_dim

        _random_orth = np.random.randn(target_embedding_codomain_dim, initial_embedding_codomain_dim)
        self.random_orth = torch.tensor(np.linalg.qr(_random_orth)[0]).float()

    def phi_then_project(self, ps: Points):
        initial_embedding = self.initial_phi(ps).coords
        return Points((self.random_orth @ initial_embedding.T).T)


