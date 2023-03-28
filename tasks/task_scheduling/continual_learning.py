"""
Continual learning as implemented by Yang et al. 2019
"""

from typing import List, Union

import torch
from tasks.task_scheduling.base import TaskSchedulerABC
from util.python_arithmetic import (
    add_list_of_tensors,
    divide_list_of_tensors,
    multiply_list_of_tensors,
    subtract_list_of_tensors,
)


class ZenkeContinualLearningTaskScheduler(TaskSchedulerABC):
    """
    See Yang at al. 2019 and Zenke et al. 2017 for notation
    """

    def __init__(
        self,
        num_tasks,
        c: float,
        xi: float,
        task_counts: Union[int, List[int]],
        **kwargs
    ):
        if isinstance(task_counts, int):
            task_counts = [task_counts for _ in range(num_tasks)]

        self.tilde_theta: List[torch.Tensor] = []
        self.previous_step_theta: List[torch.Tensor] = []
        self.small_omega: List[torch.Tensor] = []
        self.big_omega: List[torch.Tensor] = []

        self.xi = xi

        super(ZenkeContinualLearningTaskScheduler, self).__init__(
            num_tasks, c=c, task_counts=task_counts
        )

    def generate_task_order(self, num_tasks, task_counts) -> List[int]:
        """
        Same as block case, which each block representing a different mu
        """
        task_order = []
        for i in range(num_tasks):
            task_order.extend([i for _ in range(task_counts[i])])
        return task_order

    def start_task(self, params: List[torch.Tensor]) -> None:
        """
        Call this when the task is about to change, and we have not yet made a step for the next task

        We are about to start a new task, so we must:
            - Generate a new list of big deltas (total parameter changes over the task just finished)
            - Update our big omega, using big deltas and small omegas to be used in the next task
            - Restart our calculation of little omega for the new task, before first addition
            - Replace tilde theta (params at the end of the previous task)
        """
        new_big_deltas = subtract_list_of_tensors(
            [param.detach().clone() for param in params], self.tilde_theta
        )
        new_big_deltas_squared = multiply_list_of_tensors(
            new_big_deltas, new_big_deltas
        )
        contribution_to_big_omega_denominator = [
            nbds + self.xi for nbds in new_big_deltas_squared
        ]
        contribution_to_big_omega = divide_list_of_tensors(
            self.small_omega, contribution_to_big_omega_denominator
        )

        if len(self.big_omega):
            self.big_omega = add_list_of_tensors(
                self.big_omega, contribution_to_big_omega
            )
        else:
            self.big_omega = contribution_to_big_omega

        self.small_omega = [torch.zeros_like(param) for param in params]
        self.tilde_theta = [param.detach().clone() for param in params]

    def generate_current_loss(self, params: List[torch.Tensor]):
        """
        To keep things clean, generate the final quadratic loss in this function
        
        importance_weighted will automatically be [] if self.tilde_theta not set yet,
            i.e. if we're on the first task
        This still has to go through manual calculation, to ensure gradients work
        """
        tilde_theta = (
            self.tilde_theta
            if len(self.tilde_theta)
            else [p.detach().clone() for p in params]
        )
        difference_from_tilde = subtract_list_of_tensors(params, tilde_theta)

        quadratic_term = multiply_list_of_tensors(
            difference_from_tilde, difference_from_tilde
        )

        # Quick fix just to repliacte sizes, doesn't make any difference difference
        big_omega = self.big_omega if len(self.big_omega) else quadratic_term
        importance_weighted = multiply_list_of_tensors(quadratic_term, big_omega)

        return self.c * sum([iw.sum() for iw in importance_weighted])

    def imbue_task_scheduler_gradients(self, params: List[torch.Tensor]) -> None:
        """
        Register a training loss for a particular class, and possibly return
            a regularisation loss.
        Depending on the continual learning method, the return depends on
            self.state

        Implement loss itself, explained in Yang et al.
        Also determine if this is the last epoch of this task, hence we should update $\\tilde\\theta$
            and clear our $\omega^\mu$ cache
        """
        # First step of task - not yet optimised using this new task
        # This also covers self.i == 0 case, as self.i - 1 = -1 wraps to end of self.task_order
        if self.task_order[self.i - 1] != self.task_order[self.i]:
            self.start_task(params)

        # Furthermore, if this is the first task, set the previous step theta here, to allow maths to flow through
        if self.i == 0:
            self.previous_step_theta = [param.data.clone() for param in params]

        # Gradients at time t
        g = [p.grad for p in params]

        # Delta theta at time t
        deltas = subtract_list_of_tensors(params, self.previous_step_theta)

        # g(theta) * delta theta at time t
        g_times_delta_running_total_contribution = multiply_list_of_tensors(g, deltas)

        # Sum of g(theta) * delta theta from previous task
        self.small_omega = add_list_of_tensors(
            self.small_omega, g_times_delta_running_total_contribution
        )

        self.previous_step_theta = [
            param.data.clone() for param in params
        ]  # Last of all

        loss = self.generate_current_loss(params)

        # Hacky part of this implementation - manually augmenting the gradients
        CL_gradients = torch.autograd.grad(loss, params, retain_graph=True)
        for CL_gradient, param in zip(CL_gradients, params):
            param.grad = CL_gradient.data + param.grad.data

    def __iter__(self):
        for i in range(len(self.task_order)):
            yield self.step(i)
