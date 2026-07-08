from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from eden.managers import COMMAND_TERM_REGISTRY, REWARD_TERM_REGISTRY, RewardTerm
from eden.utils.isaac_math import quat_error_magnitude

from shared_terms import TargetRotationCommand

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


@REWARD_TERM_REGISTRY.register()
class OrientationProgressReward(RewardTerm):
    """Reward per-second reduction in orientation error to the active goal."""

    entity_name: str | None = None
    command_name: str = "goal_rot"
    clip: tuple[float, float] = (-10.0, 10.0)
    negative_scale: float = 1.0

    def build(self) -> None:
        super().build()
        self._cmd = self._env.command_manager.get_term(self.command_name)
        resolved_entity_name = (
            self.entity_name if self.entity_name is not None else getattr(self._cmd, "entity_name", "obj")
        )
        self._obj = self._env.entities[resolved_entity_name]

        self._prev_goal_quat = self._cmd.quat.detach().clone()
        self._prev_error = self._current_error().detach().clone()

    def _current_error(self) -> torch.Tensor:
        return quat_error_magnitude(self._obj.get_quat(), self._cmd.quat)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_goal_quat[envs_idx] = self._cmd.quat[envs_idx].detach().clone()
        self._prev_error[envs_idx] = self._current_error()[envs_idx].detach().clone()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        current_goal = self._cmd.quat
        current_error = self._current_error()
        goal_changed = torch.any(torch.abs(current_goal - self._prev_goal_quat) > 1e-6, dim=-1)

        dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        reward = (self._prev_error - current_error) / dt
        reward[reward < 0.0] *= self.negative_scale
        reward = torch.clamp(reward, min=self.clip[0], max=self.clip[1])
        reward = torch.where(goal_changed, torch.zeros_like(reward), reward)

        self._prev_goal_quat.copy_(current_goal.detach())
        self._prev_error.copy_(current_error.detach())
        return reward


@COMMAND_TERM_REGISTRY.register()
class TimeoutTrackingTargetRotationCommand(TargetRotationCommand):
    """``TargetRotationCommand`` that flags envs whose goal was just reset by *timeout*.

    Captures the timeout mask before ``super().compute(dt)`` runs the resample
    (which refills ``time_left``), so ``_last_timeout_reset`` is True for one
    step on the env(s) whose resampling timer expired. Success-driven goal
    resets fire from ``_update_command`` after the timeout resample and do not
    set this flag.
    """

    def build(self) -> None:
        super().build()
        self._last_timeout_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, dt: float) -> None:
        self._last_timeout_reset = (self.time_left - dt) <= 0.0
        super().compute(dt)


@REWARD_TERM_REGISTRY.register()
def target_timeout_reset_penalty(env: "EnvBase", *, command_name: str = "goal_rot") -> torch.Tensor:
    """1.0 on the step the orientation goal was reset by the resampling timeout."""
    cmd = env.command_manager.get_term(command_name)
    return cmd._last_timeout_reset.to(torch.float32)
