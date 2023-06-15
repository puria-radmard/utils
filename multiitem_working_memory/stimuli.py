"""
Individual features, conjuncted into stimuli, that go into the stimulus board
"""

from __future__ import annotations

import random
import numpy as np
from math import pi, floor
from torch import Tensor as T
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
    def alter_image(self, image: T) -> T: ...
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

class CiruclarFeatureBase(FeatureBase):
    value: float
    def __init__(self, value: float) -> None:
        assert (np.isnan(value)) or 0<=value<=2*pi   # Using nan to denote cue here!
        self.value = value
    def change(self, radians):
        new_value = (self.value + radians) % (2 * pi)
        return self.__class__(value = new_value)

class Orientation(CiruclarFeatureBase):
    name = 'orientation'
    def alter_image(self, image):
        raise NotImplementedError

class Colour(CiruclarFeatureBase):
    name = 'colour'
    def to_rgb(self, s=1, v=1, scale_255 = False):
        h = self.value / pi * 180
        s = float(s)
        v = float(v)
        h60 = h / 60.0
        try:
            h60f = floor(h60)
        except:
            assert np.isnan(h60)
            return (0,0,0)  # show up as black!
        hi = int(h60f) % 6
        f = h60 - h60f
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = 0, 0, 0
        if hi == 0: r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        elif hi == 5: r, g, b = v, p, q
        if scale_255:
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return r, g, b
    def alter_image(self, image):
        r,g,b = self.to_rgb()
        image[:,:,0] = r
        image[:,:,1] = g
        image[:,:,2] = b
        return image
        