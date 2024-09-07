"""
Individual features, conjuncted into stimuli, that go into the stimulus board
"""

from __future__ import annotations

import random
import numpy as np
from math import pi
from purias_utils.multiitem_working_memory.util.circle_utils import rot_to_rgb
from torch import Tensor as _T
from typing import Any, Tuple



def random_location(lower_x, upper_x, lower_y=None, upper_y=None):
    lower_y = lower_y or lower_x
    upper_y = upper_y or upper_x
    x = random.uniform(lower_x, upper_x)
    y = random.uniform(lower_y, upper_y)
    return (x, y)

def random_orientation(lower = 0, upper = 2 * pi):
    return random.uniform(lower, upper)


class FeatureBase:
    value: Any
    name: str
    def __init__(self, value) -> None: ...
    def alter_image(self, image: _T) -> _T: ...
    def change(self, *args, **kwargs) -> FeatureBase: ...
    def set(self, *args, **kwargs) -> FeatureBase: ...


class Location(FeatureBase):
    value: Tuple[float, float]
    name = 'location'
    def __init__(self, value: Tuple[float, float]) -> None:
        self.value = value
    def center_distances(self, other: Location):
        return (
            abs(self.value[0] - other.value[0]),
            abs(self.value[1] - other.value[1]),
        )
    def set(self, x, y):
        return self.__class__(value = (x,y))


class CircularFeatureBase(FeatureBase):
    value: float
    def __init__(self, value: float) -> None:
        assert (np.isnan(value)) or 0<=value<=2*pi, value   # Using nan to denote cue here!
        self.value = value
    def change(self, radians):
        new_value = (self.value + radians) % (2 * pi)
        return self.__class__(value = new_value)
    def set(self, radians):
        return self.__class__(value = radians)


class Orientation(CircularFeatureBase):
    name = 'orientation'
    def alter_image(self, image):
        raise NotImplementedError


class Colour(CircularFeatureBase):
    name = 'colour'
    def to_rgb(self, s=1, v=1, scale_255 = False):
        return rot_to_rgb(rot = self.value, s=s, v=v, scale_255 = scale_255)

    def alter_image(self, image):
        r,g,b = self.to_rgb()
        image[:,:,0] = r
        image[:,:,1] = g
        image[:,:,2] = b
        return image
        