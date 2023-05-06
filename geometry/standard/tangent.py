from purias_utils.geometry.topology import Points
import torch

def generate_around_the_s1_tangent(mult):
    def around_the_s1_tangent(ps: Points):
        output = torch.zeros_like(ps.coords)
        output[:, 0] = mult
        return Points(output)
    return around_the_s1_tangent
