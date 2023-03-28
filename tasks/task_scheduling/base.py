from typing import List
from torch import nn
import torch


class TaskSchedulerABC(nn.Module):
    """
    ABC for all task schedulers, showing generator implementation.
    Only use kwargs throughout, to filter through at different abstraction levels.

    c is the weight parameter shared amongst all classes (set to zero for simple case)
    """

    def __init__(self, num_tasks, c: float, **kwargs):
        self.num_tasks = num_tasks
        self.task_order: List[int] = self.generate_task_order(num_tasks, **kwargs)
        self.i: int = -1
        self.c: float = c
        super(TaskSchedulerABC, self).__init__()

    def generate_task_order(self, num_tasks, **kwargs) -> List[int]:
        raise NotImplementedError

    def step(self, i) -> int:
        """
        Produces task index provided to a GymMultiTaskMultiEnv in the loop.
        For continual learning cases, determines whether we should apply a
            regularisation loss at this round.
        """
        task_index = self.task_order[i]  # Return this
        self.i = i  # Update this
        return task_index

    def __iter__(self):
        for i in range(len(self.task_order)):
            yield self.step(i)
