from math import pi

from purias_utils.geometry.topology import Points, BatchedPoints


def generate_offset_chart_s1(offset):
    "Standard chart for S1. Just do minus offset for the inverse chart"
    def inner(points: Points):
        return points.coords.clone() - offset
    return inner


def generate_offset_cylinder_chart(offset):
    "Standard chart for 2-cylinder"
    def inner(points: Points):
        # TODO: batched points?
        output = points.coords.clone()
        output[:, 0] = (output[:, 0] - offset) / pi
        return output
    return inner


def generate_offset_cylinder_inverse_chart(offset):
    "Standard inverse chart for 2-cylinder"
    def inner(points):
        output = points.coords.clone()
        # TODO: make a 'get coordinates' type of function for both Points and BatchedPoints
        if isinstance(points, BatchedPoints):
            output[:, :, 0] = (output[:, :, 0] * pi) + offset
        else:
            output[:, 0] = (output[:, 0] * pi) + offset
        return output
    return inner


def generate_offset_s2_chart(offset):
    "Standard chart for 2-sphere"
    def inner(points: Points):
        # TODO: batched points?
        output = points.coords.clone()
        output[:, 0] = (output[:, 0]) / pi
        output[:, 1] = (output[:, 1] - offset) / pi
        return output
    return inner


def generate_offset_s2_inverse_chart(offset):
    "Standard inverse chart for 2-sphere"
    def inner(points):
        output = points.coords.clone()
        # TODO: make a 'get coordinates' type of function for both Points and BatchedPoints
        if isinstance(points, BatchedPoints):
            output[:, :, 0] = (output[:, :, 0] * pi)
            output[:, :, 1] = (output[:, :, 1] * pi) + offset
        else:
            output[:, 0] = (output[:, 0] * pi)
            output[:, 1] = (output[:, 1] * pi) + offset
        return output
    return inner
