import torch
from typing import List, Callable, Union
from purias_utils.geometry.topology import Points, BatchedPoints
from torch.autograd.functional import jacobian as J

from tqdm import tqdm


def compose_functions(ordered_functions):
    def inner(x):
        output = Points(x)
        for func in ordered_functions[::-1]:
            output = func(output)
        return output
    return inner


def differentiate(func: Callable[[Points], Points], eval_points: Union[Points, BatchedPoints]):
    # TODO: Fix this! Use this: https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771

    if isinstance(eval_points, BatchedPoints):
        relevant_batch_jacobian = []
        for point in tqdm(eval_points):
            unwrapped_func = lambda x: func(x).coords
            full_batch_jacobian = J(unwrapped_func, point.coords, create_graph=True).squeeze()
            relevant_batch_jacobian.append(full_batch_jacobian)
        return torch.stack(relevant_batch_jacobian, 0)

    else:
        relevant_jacobian = []
        unwrapped_func = lambda x: func(x).coords
        full_jacobian = J(unwrapped_func, eval_points.coords, create_graph=True).squeeze()
        relevant_jacobian = torch.diagonal(torch.diagonal(full_jacobian, dim1=0, dim2=1))
        return relevant_jacobian


def repeat_points_dimension(ps: Union[Points, BatchedPoints], repeat_dim: int, repeat_times: int):
    output_coords = ps.coords.clone()
    split_up_coords = output_coords.split(1, dim=-1)
    new_coords = torch.cat(
        list(split_up_coords[:repeat_dim]) +
        repeat_times*[split_up_coords[repeat_dim]] + 
        list(split_up_coords[repeat_dim + 1:]),
        dim=-1
    )
    return type(ps)(new_coords)


def immerse_coordinates(ps: Union[Points, BatchedPoints], final_dim: int, used_dimensions: List[int] = None, ambient_coords = 0.):
    """
        Add zeros (or whatever else) to coordinates
    """

    if used_dimensions is None:
        used_dimensions = list(range(ps.dim))
    assert final_dim >= ps.dim
    assert len(used_dimensions) == ps.dim

    output_coords = ps.coords.clone().split(1, dim=-1)
    final_coords = []
    counter = 0
    for fd in range(final_dim):
        if fd in used_dimensions:
            final_coords.append(output_coords[counter])
            counter += 1
        else:
            final_coords.append(ambient_coords * torch.ones_like(output_coords[0]))
    final_coords = torch.cat(final_coords, -1)
    return type(ps)(final_coords)
