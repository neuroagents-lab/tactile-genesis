"""Custom terms for the in_palm_rotate task.

Holds the reward / observation / command terms for the partial-hand in-palm
rotation task, plus the ``SetRandomActiveDofsPos`` event used to randomize only
the active (non-frozen) DOFs at reset.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import genesis as gs
import genesis.utils.geom as gu
import torch
from eden.envs.base import EnvBase
from eden.managers import (
    EVENT_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    ObservationTerm,
)
from eden.managers.command_manager import COMMAND_TERM_REGISTRY
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.managers.terms.events.domain_rand import SetRandomDofsPos
from eden.options import ObservationTermOptions
from eden.utils.geom import quat_to_rot6d
from eden.utils.isaac_math import quat_error_magnitude
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform

from shared_terms import AxisRotationProgressReward, BaseRotationCommand, _tc_quat_to_rotvec

if TYPE_CHECKING:
    from eden.options.managers.commands import CommandTermOptions


# ================== REWARDS ==================


@REWARD_TERM_REGISTRY.register()
class GatedAxisRotationProgressReward(AxisRotationProgressReward):
    """Signed spin rate about the task axis with a hard perpendicular-error gate.

    Progress is zeroed when the goal-relative orientation error component orthogonal to the rotation axis
    exceeds the command's ``allowed_off_axis_error``.
    """

    penalty_scale: float = 1.0
    clip: float = 1.0

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        obj_quat, _, inc = self._step_terms()
        if self._cached_obj is not None and torch.allclose(self._cached_obj, obj_quat, rtol=0.0, atol=1e-9):
            return self._last_reward

        dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        rate = inc / dt

        cmd = self._env.command_manager.get_term(self.command_name)
        ent = self.entity_name if self.entity_name is not None else getattr(cmd, "entity_name", "obj")
        goal_quat = cmd.quat
        obj_quat_goal = self._env.entities[ent].get_quat()
        q_rel = gu.transform_quat_by_quat(gu.inv_quat(goal_quat), obj_quat_goal)
        axis = self._axis_obj
        xi = _tc_quat_to_rotvec(q_rel)
        xi_ax = (xi * axis).sum(dim=-1, keepdim=True)
        xi_perp = xi - xi_ax * axis
        perp_error = torch.linalg.norm(xi_perp, dim=-1)
        gate = (perp_error <= cmd.allowed_off_axis_error).to(dtype=rate.dtype)

        reward = rate * gate
        reward[reward < 0.0] *= self.penalty_scale
        reward = torch.clamp(reward, max=self.clip)
        self._buf[:, 0] = self._buf[:, 0] + inc.to(dtype=self._buf.dtype)
        self._last_inc.copy_(inc.detach().to(dtype=self._last_inc.dtype))
        self._last_reward.copy_(reward.detach().to(dtype=self._last_reward.dtype))
        self._prev_quat.copy_(obj_quat.detach())
        self._cached_obj = obj_quat.detach().clone()
        return reward


@REWARD_TERM_REGISTRY.register()
def is_dropped_penalty(
    env: EnvBase,
    *,
    exclude_termination_term: str = "reached_max_consecutive_successes",
) -> torch.Tensor:
    """Like ``is_terminated_penalty``, but no penalty when the only matching termination is ``exclude_termination_term``."""
    tm = env.termination_manager
    excluded = tm.get_term(exclude_termination_term)
    return (tm.terminated & ~excluded).float()


@REWARD_TERM_REGISTRY.register()
def off_axis_orientation_penalty(
    env: EnvBase,
    *,
    command_name: str = "goal_rot",
    entity_name: str | None = None,
) -> torch.Tensor:
    """Penalty based on goal-relative orientation error orthogonal to axis."""
    cmd = env.command_manager.get_term(command_name)
    ent = entity_name if entity_name is not None else getattr(cmd, "entity_name", "obj")
    obj_quat = env.entities[ent].get_quat()
    goal_quat = cmd.quat

    axis_world = getattr(cmd, "rotation_axis_world", None)
    if axis_world is None:
        raise ValueError(f"off_axis_orientation_penalty: command {command_name!r} has no 'rotation_axis_world'.")

    axis_t = torch.as_tensor(axis_world, device=env.device, dtype=obj_quat.dtype).view(1, 3)
    axis_t = axis_t / (torch.linalg.norm(axis_t, dim=-1, keepdim=True) + 1e-9)
    axis_obj = gu.transform_by_quat(axis_t.expand(env.num_envs, -1), gu.inv_quat(obj_quat))
    axis_obj = axis_obj / (torch.linalg.norm(axis_obj, dim=-1, keepdim=True) + 1e-9)

    q_rel = gu.transform_quat_by_quat(gu.inv_quat(goal_quat), obj_quat)
    xi = _tc_quat_to_rotvec(q_rel)
    xi_ax = (xi * axis_obj).sum(dim=-1, keepdim=True)
    xi_perp = xi - xi_ax * axis_obj
    return torch.linalg.norm(xi_perp, dim=-1)


# ================== OBSERVATIONS ==================


@OBSERVATION_TERM_REGISTRY.register()
class ObjectSizeObs(ObservationTerm):
    """Static object dimensions as ``[height_z, width_xy]`` in meters."""

    entity_name: str = "obj"

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self._buf = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)

    def build(self) -> None:
        super().build()
        obj = self._env.entities[self.entity_name]
        aabb = obj.get_AABB()
        if aabb.ndim == 2:
            extents = (aabb[1] - aabb[0]).unsqueeze(0).expand(self.num_envs, -1)
        else:
            extents = aabb[:, 1, :] - aabb[:, 0, :]
        height = extents[:, 2:3]
        width = torch.amax(extents[:, :2], dim=-1, keepdim=True)
        self._buf = torch.cat((height, width), dim=-1).to(device=self.device, dtype=torch.float32)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        return self._buf


# ================== TERMINATIONS ==================


@TERMINATION_TERM_REGISTRY.register()
def reached_max_consecutive_successes(
    env: EnvBase,
    *,
    command_name: str = "goal_rot",
    max_consecutive_successes: float = 1000.0,
) -> torch.Tensor:
    """Episode success cap using the command term's cumulative ``consecutive_success`` stat."""
    streak = env.command_manager.get_term(command_name).stats["consecutive_success"]
    return streak >= max_consecutive_successes


# ================== COMMANDS ==================


@COMMAND_TERM_REGISTRY.register()
class SteppingRotationCommand(BaseRotationCommand):
    """
    Shifting orientation goal for continuous in-hand rotation.

    On time/env resample, the goal is ``delta_quat * default_root_quat`` (one world-axis
    step from the object’s configured default pose, not its live pose).

    On success, the goal advances by ``delta_quat * goal_quat`` — a further world-axis
    step from the previous command orientation.
    """

    step_rad: float = math.pi / 2
    rotation_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    allowed_off_axis_error: float = math.inf

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        self._default_root_quat: torch.Tensor | None = None

        dtype = self.quat.dtype
        self.stats["on_axis_error"] = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.stats["off_axis_error"] = torch.zeros(self.num_envs, device=self.device, dtype=dtype)

    def build(self) -> None:
        super().build()
        _, self._default_root_quat = self.object.get_default_root_pose()
        # Match ConstantOrientationCommand: set initial goal here so command is never
        # the ctor identity quaternion before the first env reset.
        self._resample_command(slice(None))
        self._update_orientation_stats(advance_streak=False)

    def __str__(self) -> str:
        msg = "SteppingRotationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.resampling_time_range}\n"
        msg += f"\tStep (rad): {self.step_rad}\n"
        msg += f"\tRotation axis (world): {self.rotation_axis_world}"
        return msg

    def _apply_world_axis_step(self, envs_idx: slice | torch.Tensor, base_quat_all: torch.Tensor) -> None:
        """Set ``self.quat[envs_idx]`` to ``delta_quat * base`` with ``base`` taken from ``base_quat_all``."""
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]
        if n_envs == 0:
            return

        base_quat = base_quat_all[envs_idx]
        axis = torch.tensor(self.rotation_axis_world, device=self.device, dtype=base_quat.dtype)
        axis = axis / (torch.norm(axis) + 1e-9)
        axis_w = axis.unsqueeze(0).expand(n_envs, -1)
        angles = torch.full((n_envs,), self.step_rad, device=self.device, dtype=base_quat.dtype)
        step_quat = gu.axis_angle_to_quat(angles, axis_w)
        self.quat[envs_idx] = gu.transform_quat_by_quat(base_quat, step_quat)
        self.rotation[envs_idx] = quat_to_rot6d(self.quat[envs_idx])
        self._sync_goal_vis_orientation(envs_idx)

    def _resample_command(self, envs_idx: slice | torch.Tensor):
        """Set goal to default root orientation rotated by ``step_rad`` about the world axis."""
        assert self.object is not None and self._default_root_quat is not None
        self._apply_world_axis_step(envs_idx, self._default_root_quat)

    def _update_orientation_stats(self, *, advance_streak: bool = True) -> torch.Tensor:
        """Success requires on-axis error < ``orientation_success_threshold`` AND off-axis error < ``allowed_off_axis_error``."""
        assert self.object is not None
        obj_quat = self.object.get_quat()
        self.stats["orientation_error"][:] = quat_error_magnitude(obj_quat, self.quat)

        axis_w = torch.as_tensor(self.rotation_axis_world, device=self.device, dtype=obj_quat.dtype).view(1, 3)
        axis_w = axis_w / (torch.linalg.norm(axis_w, dim=-1, keepdim=True) + 1e-9)
        axis_obj = gu.transform_by_quat(axis_w.expand(self.num_envs, -1), gu.inv_quat(obj_quat))
        axis_obj = axis_obj / (torch.linalg.norm(axis_obj, dim=-1, keepdim=True) + 1e-9)

        q_rel = gu.transform_quat_by_quat(gu.inv_quat(self.quat), obj_quat)
        xi = _tc_quat_to_rotvec(q_rel)
        xi_ax_signed = (xi * axis_obj).sum(dim=-1, keepdim=True)
        xi_perp = xi - xi_ax_signed * axis_obj

        on_axis_err = xi_ax_signed.squeeze(-1).abs()
        off_axis_err = torch.linalg.norm(xi_perp, dim=-1)
        self.stats["on_axis_error"][:] = on_axis_err
        self.stats["off_axis_error"][:] = off_axis_err

        success = (on_axis_err < self.orientation_success_threshold) & (off_axis_err < self.allowed_off_axis_error)
        self._last_success = success
        if advance_streak:
            self.stats["consecutive_success"] += success.float()
        return success

    def _update_command(self) -> None:
        """Advance the goal by another world-fixed step when on-axis error is below threshold and off-axis error is within bounds."""
        success = self._update_orientation_stats()
        if self.update_goal_on_success:
            goal_reset_ids = success.nonzero(as_tuple=False).squeeze(-1)
            if goal_reset_ids.numel() > 0:
                self._apply_world_axis_step(goal_reset_ids, self.quat)


# ================== EVENTS ==================


@EVENT_TERM_REGISTRY.register()
class SetRandomActiveDofsPos(SetRandomDofsPos):
    """``SetRandomDofsPos`` restricted to a named subset of DOFs.

    DOFs not listed in ``dofs_name`` are left untouched, so the frozen fingers
    keep their canonical default pose instead of being randomized at reset.
    When ``dofs_name`` is empty this behaves exactly like ``SetRandomDofsPos``.
    """

    dofs_name: tuple[str, ...] = ()

    def build(self) -> None:
        super().build()
        if not self.dofs_name:
            self._dofs_idx_local = None
            return
        _, idx = self.entity.find_named_dofs_idx_local(
            list(self.dofs_name),
            name_scope=self.entity.dofs_name,
            preserve_order=True,
        )
        self._dofs_idx_local = torch.as_tensor(idx, dtype=gs.tc_int, device=self._env.device).contiguous()
        if self.apply_as_ratio:
            # Base build cached per-DOF limits as (1, n_dofs); restrict to the subset.
            self._dof_lower = self._dof_lower[:, self._dofs_idx_local]
            self._dof_range = self._dof_range[:, self._dofs_idx_local]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if self._dofs_idx_local is None:
            super().compute(envs_idx)
            return
        if envs_idx is None:
            envs_idx = slice(None)
        sampled = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (self._env.num_envs, self._dofs_idx_local.numel()),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            dof_lower = self._dof_lower if self._dof_lower.shape[0] == 1 else self._dof_lower[envs_idx]
            dof_range = self._dof_range if self._dof_range.shape[0] == 1 else self._dof_range[envs_idx]
            dofs_pos = dof_lower + sampled * dof_range
        else:
            dofs_pos = sampled
        self.entity.set_dofs_pos(dofs_pos, dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx)
