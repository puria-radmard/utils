from __future__ import annotations
import itertools


import torch
from torch import Tensor as T
from torch import tensor

import copy
from typing import Union, Callable, Type, List


def check_and_map(
    points: Union[Points, BatchedPoints],
    domain: RightRectangle,
    map: Callable[[Points], Points],
    unwrap=False,
) -> Points:
    """
    TODO: make this a decorator properly!!
    """
    assert points.dim == domain.dim, \
        f"Function {map} on {domain} expected points of dimension {domain.dim}, not of {points.dim}"
    if isinstance(points, BatchedPoints):
        contained = domain.contains(points)
    else:
        contained = domain.contains(points).flatten()
    mapping = map(points)
    mapping.coords[~contained] = torch.nan
    return mapping.coords if unwrap else mapping




class Points:
    def __init__(self, coords: Union[T, Points]) -> None:
        if isinstance(coords, Points):
            point_coords: T = coords.coords
        else:
            point_coords = coords
        assert len(point_coords.shape) == 2
        self.dim = point_coords.shape[-1]
        self.coords = point_coords
        self.num_points = point_coords.shape[0]

    def slice_dims(self, i: int, i_end: Union[None, int] = None):
        return self.coords[:,[i]] if i_end is None else self.coords[:,i:i_end]

    def get_dim(self, i: int, i_end: Union[None, int] = None):
        # TODO: should really use a slice?
        return Points(self.slice_dims(i, i_end))

    def __getitem__(self, i):
        return Points(self.coords[[i]])

    def __len__(self):
        return len(self.coords)


class BatchedPoints(Points):
    def __init__(self, coords: Union[T, BatchedPoints]) -> None:
        if isinstance(coords, Points):
            point_coords: T = coords.coords.unsqueeze(1)
        else:
            point_coords = coords
        assert len(point_coords.shape) == 3
        self.dim = point_coords.shape[-1]
        self.coords = point_coords
        self.num_points = point_coords.shape[1]
        self.num_batches = point_coords.shape[0]

    def slice_dims(self, i: int, i_end: Union[None, int] = None):
        return self.coords[:,:,[i]] if i_end is None else self.coords[:,:,i:i_end]

    def get_dim(self, i: int, i_end: Union[None, int] = None):
        # TODO: should really use a slice?
        return BatchedPoints(self.slice_dims(i, i_end))

    def __getitem__(self, i):
        """
        Retain the batch structure, but get the ith point out of each batch
        """
        return Points(self.coords[i])



class RightRectangle:
    """
    Cartesian product of <dim> intervals
    """

    def __init__(self, left: T, right: T):
        assert (len(right.shape) == 1) and (len(left.shape) == 1)
        assert all(left < right) and (len(left) == len(right))
        self.left = left
        self.right = right
        self.dim = right.shape[0]

    def __len__(self):
        return self.dim

    def length(self, i):
        return self.right[i] - self.left[i]

    @property
    def volume(self):
        return (self.left - self.right).prod().item()

    def contains(self, obj: Union[Points, BatchedPoints, RightRectangle]):
        assert obj.dim == self.dim
        if isinstance(obj, (BatchedPoints)):  # Array of points it does contain
            ret = torch.logical_and(
                (self.left <= obj.coords), (obj.coords <= self.right)
            )
            return ret.all(-1).all(-1)
        elif isinstance(obj, (Points)):  # Array of points it does contain
            ret = torch.logical_and(
                (self.left <= obj.coords), (obj.coords <= self.right)
            )
            return ret.all(-1)
        elif isinstance(obj, RightRectangle):  # Covers set
            return all(self.left <= obj.left) and all(obj.right <= self.right)

    def generate_grid(self, count: int):
        linspaces = [
            torch.linspace(lower, upper, count).tolist()
            for lower, upper in zip(self.left, self.right)
        ]

        return Points(tensor(list(itertools.product(*linspaces))))

    def cartesian_product(self, other: RightRectangle):
        return RightRectangle(
            left=torch.cat([self.left, other.left]),
            right=torch.cat([self.right, other.right]),
        )


class UnitRightRectangle(RightRectangle):
    def __init__(self, dim):
        super().__init__(left=torch.zeros(dim), right=torch.ones(dim))


class Interval(RightRectangle):
    def __init__(self, left, right):
        super().__init__(left=tensor([left]), right=tensor([right]))


class Map:
    def __init__(self, f: Callable[[Points], T], domain_dim: int):
        self._f = f
        self.domain_dim = domain_dim

    def f(self, ps: Union[Points, BatchedPoints]):
        assert ps.dim == self.domain_dim
        return Points(self._f(ps))

    def cartesian_product(self, other: Map):
        def new_f(ps: Union[Points, BatchedPoints]):
            own_ps = ps.get_dim(0, self.domain_dim)
            other_ps = ps.get_dim(self.domain_dim, self.domain_dim + other.domain_dim)
            return torch.cat([self.f(own_ps).coords, other.f(other_ps).coords], dim=-1)
        return Map(f=new_f, domain_dim=self.domain_dim + other.domain_dim)


class TopoplogicalHomeomorphy(Map):
    def __init__(
        self,
        f: Callable[[Points], T],
        f_inverse: Callable[[Points], T],
        domain_dim: int,
        codomain_dim: int
    ):
        super(TopoplogicalHomeomorphy, self).__init__(f, domain_dim)
        self._f_inv = f_inverse
        self.codomain_dim = codomain_dim

    def f_inv(self, ps: Union[Points, BatchedPoints]):
        return type(ps)(self._f_inv(ps))

    def cartesian_product(self, other: TopoplogicalHomeomorphy):
        forward_map = super(TopoplogicalHomeomorphy, self).cartesian_product(other)
        def new_f_inv(ps: Union[Points, BatchedPoints]):
            own_ps = ps.get_dim(0, self.domain_dim)
            other_ps = ps.get_dim(self.domain_dim, self.domain_dim + other.domain_dim)
            output = torch.cat([self.f_inv(own_ps).coords, other.f_inv(other_ps).coords], dim=-1)
            return output
        return TopoplogicalHomeomorphy(
            f=forward_map._f, 
            f_inverse=new_f_inv, 
            domain_dim=self.domain_dim + other.domain_dim, 
            codomain_dim=self.codomain_dim + other.codomain_dim
        )


class Chart(TopoplogicalHomeomorphy):
    """
    Simply first checks that we're in the domain, which is assumed to be a right rectangle
    Also assumes that the codomain (i.e. a subset of R^d) is a right rectangle!
        This is fine to assume given the standard topology I believe

    Defines the homeomorphy between the chart domain and Euclidean space
    """

    def __init__(
        self,
        f: Callable[[Points], T],
        f_inverse: Callable[[Points], T],
        domain: RightRectangle,
        codomain: RightRectangle,
    ):
        super().__init__(f, f_inverse, domain.dim, codomain.dim)
        self.domain = domain
        self.codomain = codomain
        assert domain.dim == codomain.dim

    def f(self, ps: Union[Points, BatchedPoints]):
        """Returns map if in domain, else a NaN."""
        return check_and_map(ps, self.domain, super(Chart, self).f)

    def f_inv(self, ps: Union[Points, BatchedPoints]):
        """Again, just a check"""
        return check_and_map(ps, self.codomain, super(Chart, self).f_inv)

    def cartesian_product(self, other: Chart):
        new_homeography = super(Chart, self).cartesian_product(other)
        return Chart(
            f=new_homeography._f, f_inverse=new_homeography._f_inv, 
            domain = self.domain.cartesian_product(other.domain),
            codomain = self.codomain.cartesian_product(other.codomain)
        )



class Atlas:
    def __init__(self, charts: List[Chart]):
        self.charts = charts

    def map(self, ps: Union[Points, BatchedPoints]):
        """
        Map via ALL charts.
        NB: charts for which this point is out of bounds will just map to nan
        """
        return [chart.f(ps) for chart in self.charts]

    def unmap(self, ps: Points, i: int):
        """
        Unmapping, i.e. x^-1, only makes sense when the chart is chosen
        """
        return self.charts[i].f_inv(ps)

    def cartesian_product_charts(self, other: Atlas):
        new_charts = []
        for chart in self.charts:
            for other_chart in other.charts:
                new_charts.append(chart.cartesian_product(other_chart))
        return Atlas(charts=new_charts)


class Manifold(RightRectangle, Atlas):
    """
    Assumes the underlying set is a right rectangle
    Notice that Manifold class does not inherit from a "Topology" class - we assume
        the standard topology throughout, meaning we only need to equip the resulting topological
        space with an Atlas
    """

    def __init__(self, set_left: T, set_right: T, charts: List[Chart]):

        RightRectangle.__init__(
            self, left=set_left, right=set_right
        )  # the set M (already equipped with O_std)
        Atlas.__init__(self, charts=charts)  # the altas A

        for chart in charts:
            assert self.contains(chart.domain) and (chart.domain.dim == self.dim)

    @classmethod
    def from_set_and_atlas(cls, underlying_set: RightRectangle, atlas: Atlas):
        return cls(
            set_left=underlying_set.left,
            set_right=underlying_set.right,
            charts=atlas.charts,
        )

    def cartesian_product(self, other: Manifold):
        new_underlying_set = RightRectangle.cartesian_product(self, other)
        new_atlas = self.cartesian_product_charts(other)
        return Manifold.from_set_and_atlas(new_underlying_set, new_atlas)
