"""rsl_rl distillation algorithm and runner configuration options."""

from __future__ import annotations

from typing import Literal

from eden.options.learning.rsl_rl.rl import (
    RslRlBaseRunnerOptions,
    RslRlModelOptions,
    RslRlModelWithDistributionOptions,
)
from eden.options.options import ConfigurableOptions

############################
# Algorithm configurations #
############################


class RslRlDistillationAlgorithmOptions(ConfigurableOptions):
    """Configuration for the distillation algorithm.

    Parameters
    ----------
    class_name: str
        The algorithm class name.
    num_learning_epochs: int
        The number of updates performed with each sample.
    learning_rate: float
        The learning rate for the student policy.
    gradient_length: int
        The number of environment steps the gradient flows back.
    max_grad_norm: float | None
        The maximum norm the gradient is clipped to.
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"]
        The optimizer to use.
    loss_type: Literal["mse", "huber"]
        The loss type.
    """

    class_name: str = "Distillation"
    num_learning_epochs: int = 1
    learning_rate: float = 1.0e-3
    gradient_length: int = 1
    max_grad_norm: float | None = None
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    loss_type: Literal["mse", "huber"] = "mse"


#########################
# Runner configurations #
#########################


class RslRlDistillationRunnerOptions(RslRlBaseRunnerOptions):
    """Configuration of the runner for distillation algorithms.

    Parameters
    ----------
    class_name: str
        The runner class name.
    student: RslRlModelWithDistributionOptions
        The student model configuration with distribution to sample from.
    teacher: RslRlModelOptions
        The teacher model configuration.
    algorithm: RslRlDistillationAlgorithmOptions
        The algorithm configuration.
    """

    class_name: str = "DistillationRunner"
    student: RslRlModelWithDistributionOptions
    teacher: RslRlModelOptions
    algorithm: RslRlDistillationAlgorithmOptions


# Aliases for backward compatibility
RslRlStudentModelOptions = RslRlModelWithDistributionOptions
