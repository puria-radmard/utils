from purias_utils.geometry.topology import Points, BatchedPoints
import torch
from math import pi

def generate_offset_s1_embedding(radius: float, offset: float):
    "Embed S1 in R^2 with radius and offset along x dimension"
    def inner(ps: Points):
        output = torch.zeros(len(ps), 2)
        # TODO: equip embeddings with some kind of batch mode to deal with this - effectively creating e_i^* instead!
        if isinstance(ps, BatchedPoints):
            output[:, 0] = offset + radius * torch.cos(ps.coords[:, :, 0]).sum(-1)  # XXX
            output[:, 1] = radius * torch.sin(ps.coords[:, :, 0]).sum(-1)  # XXX
        else:
            output[:, 0] = offset + radius * torch.cos(ps.coords[:, 0])
            output[:, 1] = radius * torch.sin(ps.coords[:, 0])
        return Points(output)
    return inner


def generate_offset_cylinder_embedding(radius: float, offset: float):
    "Embed cylinder in R^2 with radius and offset along x dimension"
    def inner(ps: Points):
        output = torch.zeros(len(ps), 3)
        # TODO: equip embeddings with some kind of batch mode to deal with this - effectively creating e_i^* instead!
        if isinstance(ps, BatchedPoints):
            output[:, 0] = offset + radius * torch.cos(ps.coords[:, :, 0]).sum(-1)  # XXX
            output[:, 1] = radius * torch.sin(ps.coords[:, :, 0]).sum(-1)  # XXX
            output[:, 2] = (ps.coords[:, :, 1] - 0.5).sum(-1)
        else:
            output[:, 0] = offset + radius * torch.cos(ps.coords[:, 0])
            output[:, 1] = radius * torch.sin(ps.coords[:, 0])
            output[:, 2] = ps.coords[:, 1] - 0.5
        return Points(output)
    return inner


def generate_offset_s2_embedding(radius: float, offset: float):
    "Embed S2 in R^3 with radius and offset along x dimension"
    def inner(ps: Points):
        output = torch.zeros(len(ps), 3)
        ## A SPHERE NOW
        if isinstance(ps, BatchedPoints):
            output[:, 0] = radius * (ps.coords[:, :, 0].cos() * (ps.coords[:, :, 1]).sin()).sum(-1) - offset  # XXX
            output[:, 1] = radius * (ps.coords[:, :, 0].sin() * (ps.coords[:, :, 1]).sin()).sum(-1)  # XXX
            output[:, 2] = radius * (ps.coords[:, :, 1]).cos().sum(-1)             # XXX
        else:
            output[:, 0] = radius * (ps.coords[:, 0].cos() * (ps.coords[:, 1]).sin()) - offset
            output[:, 1] = radius * (ps.coords[:, 0].sin() * (ps.coords[:, 1]).sin())
            output[:, 2] = radius * (ps.coords[:, 1]).cos()
        return Points(output)
    return inner
