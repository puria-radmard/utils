from typing import Callable, Union

import torch
from torch import Tensor as _T
from torch.autograd import grad
from torch.autograd.functional import jacobian

from tqdm import tqdm

from purias_utils.geometry.topology import Points


class FunctionDifferentiator:
    """
        Use this to get the nth order differential of a function
        
        _cached_results holds the order differential
            e.g. _cached_results[0] holds the original evaluation, _cached_results[1] holds the Jacobian, etc.
    """

    def __init__(self, function: Callable[[Points], Points], points: Points, d_out: int, sub_batch_size: int = 64):
        
        self.function = function
        
        self.eval_points = points
        assert points.requires_grad
        assert len(self.eval_points.coords.shape) == 2

        self.batch_size = self.eval_points.coords.shape[0]
        self.d_in = self.eval_points.coords.shape[1]
        self.d_out = d_out
        self.sub_batch_size = sub_batch_size

        num_batches = self.batch_size // sub_batch_size + 1
        batches = []
        for i in range(num_batches):
            batch = self.eval_points[i*sub_batch_size:(i+1)*sub_batch_size]
            if len(batch) == 0:
                break
            batches.append(batch.coords)

        self.batches = batches

        self._cached_results = {
            0: []
        }

    def check_shape(self, tensor: _T):
        assert tuple(tensor.shape[:2]) == (self.batch_size, self.d_out)
        for dim in tensor.shape[2:]:
            assert dim == self.d_in

    def calculate_nth_order_diff_function(self, n: int):
        for batch_number in tqdm(range(len(self.batches))):
            self._nth_order_diff_function_inner(self.batches[batch_number], batch_number, n)
        for m in range(n):
            result = torch.cat(self._cached_results[m], 0)
            self.check_shape(result)

    def _nth_order_diff_function_inner(self, batch: _T, batch_number: int, n: int):
        "Following example here: https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/3"
        """
            Should be of shape [batch, d_out, (d_in n times)] for example:
                d^0f/dx^0 = f(x) is of shape [batch, d_out]
                d^1f/dx^1 is of shape [batch, d_out, d_in] (Jacobian)
                d^2f/dx^2 is of shape [batch, d_out, d_in, d_in] (Hessian)
                etc.
            
            The output is the next order up, so a d_in dimension is inserted at the end
        """
        
        eval_points = batch #self.batches[batch_number]

        if n < 0:
            raise ValueError

        elif n == 0:
            result = self.function(Points(eval_points)).coords
            assert len(self._cached_results[0]) == batch_number
            self._cached_results[0].append(result)
            return result

        if (n in self._cached_results) and len(self._cached_results[n]) > batch_number:
            return self._cached_results[n][batch_number]

        else:
            if (n in self._cached_results):
                assert len(self._cached_results[n]) == batch_number # Should be done sequentially

            else:
                self._cached_results[n] = []

            diffed = jacobian(lambda x: self._nth_order_diff_function_inner(x, batch_number, n - 1).sum(0), eval_points, create_graph=True)

            for _ in range(n):
                diffed = torch.stack(list(diffed), -2)

            self._cached_results[n].append(diffed)

            return diffed





    def batch_differentiate(self, df_prevn_x: _T):
        """
            Input is the evaluation of d^{n-1}f/dx^{n-1}
            
            This should be of shape [batch, d_out, (d_in n-1 times)] for example:
                d^0f/dx^0 = f(x) is of shape [batch, d_out]
                d^1f/dx^1 is of shape [batch, d_out, d_in] (Jacobian)
                d^2f/dx^2 is of shape [batch, d_out, d_in, d_in] (Hessian)
                etc.
            
            The output is the next order up, so a d_in dimension is inserted at the end
        """
        self.check_shape(df_prevn_x)

        # There cannot be across batch terms, so we can do this safely
        # nansum because out of domain points eval to NaN
        eval = df_prevn_x.nansum(dim=0)
        
        # Being canvas to add everything to it
        # canvas = torch.zeros(*df_prevn_x.shape, self.d_in, requires_grad=True)
        all_element_grads = []

        for el in tqdm(eval.reshape(-1)):
            element_grad = grad(el, self.eval_points, retain_graph=True)
            all_element_grads.append(element_grad[0])

        full_moment = torch.cat(all_element_grads, -1)
        full_moment = full_moment.reshape(*df_prevn_x.shape, self.d_in)

        self.check_shape(full_moment)
        return full_moment

    def nth_order_grad(self, n: int):
        assert n >= 0
        if n in self._cached_results:
            return self._cached_results[n]

        # d^{n-1}f/dx^{n-1}, evaluated
        order_down = self.nth_order_grad(n - 1)

        # Differentiate once more
        self._cached_results[n] = self.batch_differentiate(order_down)

        return self._cached_results[n]
        
