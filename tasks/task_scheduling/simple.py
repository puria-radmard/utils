"""
Simple task schedulers that don't do anything but provide a task order
"""

import random, torch
from typing import List, Union
from tasks.task_scheduling.base import TaskSchedulerABC


class SimpleTaskSchedulerABC(TaskSchedulerABC):
    """
    ABC for task schedulers which only provide the task upcoming,
        and not any regularisation

    'Traditional Learning' in Yang et al. 2019
    """

    def __init__(self, num_tasks: int, task_counts: Union[int, List[int]], **kwargs):
        """
        Task counts is number of times we see each task total.
            Either a list of ints of length num_tasks, or one int that is
            duplicated for each task
        State doesn't matter, leave it at 0
        """
        if isinstance(task_counts, int):
            task_counts = [task_counts for _ in range(num_tasks)]
        super(SimpleTaskSchedulerABC, self).__init__(
            num_tasks, c=0.0, task_counts=task_counts
        )

    def generate_task_order(self, num_tasks, task_counts) -> List[int]:
        """
        Only thing specialised between subtypes
        """
        raise NotImplementedError

    def imbue_task_scheduler_gradients(self, params: List[torch.Tensor]):
        """
        No regularisation applied
        """

    def __iter__(self):
        for i in range(len(self.task_order)):
            yield self.step(i)


class BlockSimpleTaskScheduler(SimpleTaskSchedulerABC):
    """
    Each task is presented in order, for its full counts
    """

    def generate_task_order(self, num_tasks, task_counts) -> List[int]:
        task_order = []
        for i in range(num_tasks):
            task_order.extend([i for _ in range(task_counts[i])])
        return task_order


class RandomlyInterleavedSimpleTaskScheduler(SimpleTaskSchedulerABC):
    """
    BlockSimpleTaskScheduler but shuffled at start
    """

    def generate_task_order(self, num_tasks, task_counts) -> List[int]:
        task_order = []
        for i in range(num_tasks):
            task_order.extend([i for _ in range(task_counts[i])])
        random.shuffle(task_order)
        return task_order
