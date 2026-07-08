"""Oracle pose command terms (uniform T3 / SE3 targets)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
from genesis.utils.geom import xyz_to_quat

from eden.managers import CommandTerm, COMMAND_TERM_REGISTRY
from eden.utils.isaac_math import quat_apply
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class UniformT3Command(CommandTerm):
    """
    Command that samples T(3) translation command.

    Parameters
    ----------
    x_range: tuple[float, float]
        Range of the x-coordinate of the target position.
    y_range: tuple[float, float]
        Range of the y-coordinate of the target position.
    z_range: tuple[float, float]
        Range of the z-coordinate of the target position.
    debug_vis: bool
        When True (and not training), draw a debug sphere at the goal each step.
    marker_frame_entity: str
        Entity whose frame the command is expressed in. When set, the marker is
        placed at ``entity.pos + R(entity.quat) @ command`` (the command is in that
        entity's local frame, e.g. the robot base); empty -> command is world-frame.
    marker_entity_name: str
        Name of a scene entity (e.g. a non-colliding ``SphereOptions``) to move to
        the goal each step. Preferred over ``draw_debug_spheres`` because debug
        draws are viewer-only and do NOT appear in recorded camera videos.
    marker_radius: float
        Fallback debug-sphere radius [m] (used only when ``marker_entity_name`` is
        unset).
    marker_color: tuple[float, float, float, float]
        Fallback debug-sphere RGBA.
    """

    x_range: tuple[float, float] = (0.3, 0.5)
    y_range: tuple[float, float] = (-0.2, 0.2)
    z_range: tuple[float, float] = (0.2, 0.4)
    marker_frame_entity: str = ""
    marker_entity_name: str = ""
    marker_radius: float = 0.03
    marker_color: tuple[float, float, float, float] = (0.1, 0.9, 0.1, 0.85)

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        self.oracle_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.oracle_pos_b

    def _resample_command(self, envs_idx: slice | torch.Tensor):
        if envs_idx is None:
            envs_idx = slice(None)

        lower = torch.tensor(
            [self.x_range[0], self.y_range[0], self.z_range[0]],
            device=self.device,
            dtype=torch.float32,
        )
        upper = torch.tensor(
            [self.x_range[1], self.y_range[1], self.z_range[1]],
            device=self.device,
            dtype=torch.float32,
        )
        sampled = sample_uniform(lower, upper, (self._env.num_envs, 3), device=self.device)[envs_idx]
        self.oracle_pos_b[envs_idx] = sampled

    def _update_command(self):
        # No per-step update needed; command stays constant until resample.
        pass

    def draw_vis(self) -> None:
        # Goal marker (eval/inference only; gated on debug_vis by the manager).
        pos = self.oracle_pos_b
        if self.marker_frame_entity:
            ent = self._env.entities[self.marker_frame_entity]
            pos = ent.get_pos() + quat_apply(ent.get_quat(), pos)
        marker = (
            self._env.entities[self.marker_entity_name]
            if self.marker_entity_name and self.marker_entity_name in self._env.entities
            else None
        )
        if marker is not None:
            # Move a real (non-colliding) entity to the goal — renders in cameras,
            # unlike draw_debug_* which only shows in the interactive viewer.
            marker.set_pos(pos.detach())
        else:
            self._env.clear_debug_objects()
            self._env.draw_debug_spheres(pos.detach().cpu().numpy(), radius=self.marker_radius, color=self.marker_color)


@COMMAND_TERM_REGISTRY.register()
class UniformSE3Command(CommandTerm):
    """
    Command that samples SE(3) translation and rotation command.

    Parameters
    ----------
    x_range: tuple[float, float]
        Range of the x-coordinate of the target position.
    y_range: tuple[float, float]
        Range of the y-coordinate of the target position.
    z_range: tuple[float, float]
        Range of the z-coordinate of the target position.
    roll_range: tuple[float, float]
        Range of the roll angle of the target rotation (in radians).
    pitch_range: tuple[float, float]
        Range of the pitch angle of the target rotation (in radians).
    yaw_range: tuple[float, float]
        Range of the yaw angle of the target rotation (in radians).

    Note
    ----
    This term provides a equivalent command to `UniformPoseCommand` in IsaacLab.
    """

    x_range: tuple[float, float] = (0.3, 0.5)
    y_range: tuple[float, float] = (-0.2, 0.2)
    z_range: tuple[float, float] = (0.2, 0.4)
    roll_range: tuple[float, float] = (-np.pi, np.pi)
    pitch_range: tuple[float, float] = (-np.pi, np.pi)
    yaw_range: tuple[float, float] = (-np.pi, np.pi)

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.oracle_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.oracle_pose_b[:, 3] = 1.0  # ensure the quaternion has real part as positive

    @property
    def command(self) -> torch.Tensor:
        return self.oracle_pose_b

    def _resample_command(self, envs_idx: slice | torch.Tensor):
        if envs_idx is None:
            envs_idx = slice(None)
        # NOTE: sample new position targets
        lower = torch.tensor(
            [self.x_range[0], self.y_range[0], self.z_range[0]],
            device=self.device,
            dtype=torch.float32,
        )
        upper = torch.tensor(
            [self.x_range[1], self.y_range[1], self.z_range[1]],
            device=self.device,
            dtype=torch.float32,
        )
        sampled = sample_uniform(lower, upper, (self._env.num_envs, 3), device=self.device)[envs_idx]
        self.oracle_pose_b[envs_idx, :3] = sampled
        # NOTE: sample new orientation targets
        lower = torch.tensor(
            [self.roll_range[0], self.pitch_range[0], self.yaw_range[0]],
            device=self.device,
            dtype=torch.float32,
        )
        upper = torch.tensor(
            [self.roll_range[1], self.pitch_range[1], self.yaw_range[1]],
            device=self.device,
            dtype=torch.float32,
        )
        sampled = sample_uniform(lower, upper, (self._env.num_envs, 3), device=self.device)[envs_idx]
        self.oracle_pose_b[envs_idx, 3:] = xyz_to_quat(sampled, rpy=True)

    def _update_command(self):
        pass
