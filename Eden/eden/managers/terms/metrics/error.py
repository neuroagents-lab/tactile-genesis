"""Velocity-tracking error metric terms."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.command_manager import CommandTerm
from eden.managers.metric_manager import METRIC_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@METRIC_TERM_REGISTRY.register(
    is_cumulative=True,
    direction=MetricDirection.LIB,
    metric_mode=MetricMode.RESET,
)
def error_vel_xy(
    env: RLEnvBase,
    *,
    command_name: str = "velocity",
):
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.norm(command.vel_command_b[:, :2] - command.robot.get_vel()[:, :2], dim=-1)


@METRIC_TERM_REGISTRY.register(
    is_cumulative=True,
    direction=MetricDirection.LIB,
    metric_mode=MetricMode.RESET,
)
def error_vel_yaw(
    env: RLEnvBase,
    *,
    command_name: str = "velocity",
):
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.abs(command.vel_command_b[:, 2] - command.robot.get_ang()[:, 2])
