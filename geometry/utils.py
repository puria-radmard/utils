import torch
from typing import List, Callable, Union
from purias_utils.geometry.topology import Points, BatchedPoints

from torch.autograd import grad
from torch import Tensor as T

from tqdm import tqdm


def compose_functions(ordered_functions):
    def inner(x):
        output = x
        for func in ordered_functions[::-1]:
            output = func(output)
        return output
    return inner


def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)   


def differentiate(evaluations: Callable[[Points], Points], eval_points: Union[Points, BatchedPoints, T]):
    # TODO: make this way more efficient, and able to chain with drop off at each order

    # TODO: fix this!!

    evaluations = evaluations.coords

    if isinstance(eval_points, BatchedPoints) or (eval_points.shape == 3):
        relevant_batch_jacobian = []
        for evaluation, point in tqdm(zip(evaluations, eval_points)):
            full_batch_jacobian = jacobian(evaluation, eval_points.coords, create_graph=True).squeeze()
            relevant_batch_jacobian.append(full_batch_jacobian)
        return torch.stack(relevant_batch_jacobian, 0)

    elif isinstance(eval_points, Points) or (eval_points.shape == 2):
        relevant_jacobian = []
        full_jacobian = jacobian(evaluations, eval_points.coords, create_graph=True).squeeze()
        relevant_jacobian = torch.diagonal(torch.diagonal(full_jacobian, dim1=0, dim2=1))
        return relevant_jacobian


def so_differentiate(func: Callable[[Points], Points], eval_points: Union[Points, BatchedPoints]):
    # Hessian
    inner = lambda x: differentiate(func, x)
    return differentiate(inner, eval_points)




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
    import pdb; pdb.set_trace()
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
