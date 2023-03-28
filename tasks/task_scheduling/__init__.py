"""
Task scehduler informs what task is coming up, and handles relevant regularisation,
such as continual learning.
"""

from tasks.task_scheduling.simple import *
from tasks.task_scheduling.continual_learning import *

__all__ = [
    "BlockSimpleTaskScheduler",
    "RandomlyInterleavedSimpleTaskScheduler",
    "ZenkeContinualLearningTaskScheduler",
]
