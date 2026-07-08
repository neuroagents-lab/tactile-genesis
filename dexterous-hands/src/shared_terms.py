from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import eden as en
import genesis as gs
import genesis.utils.geom as gu
import torch
from eden.constants import MetricDirection, MetricMode, ReferenceSource
from eden.managers import (
    ACTION_TERM_REGISTRY,
    COMMAND_TERM_REGISTRY,
    EVENT_TERM_REGISTRY,
    METRIC_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    TERMINATION_TERM_REGISTRY,
    ActionTerm,
    CommandTerm,
    EventTerm,
    MetricTerm,
    ObservationTerm,
    RewardTerm,
)
from eden.managers.modifiers.actions.actuators import (
    Compose,
    ConstantTorqueKick,
    Deadband,
    GearBacklash,
    MotorStrength,
    TorqueOffset,
)
from eden.options import (
    ActionTermOptions,
    CommandTermOptions,
    EventTermOptions,
    MetricTermOptions,
    ObservationTermOptions,
)
from eden.utils.geom import quat_to_rot6d
from eden.utils.isaac_math import axis_angle_from_quat, quat_conjugate, quat_error_magnitude, quat_mul
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform
from eden.utils.string import resolve_matching_names

from tactile_sensors import (
    TACTILE_SENSORS,
    TemporalReductionMode,
    postprocess_generic,
    spec_for_sensor_name,
)
from utils import resolve_entity_link

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


if TYPE_CHECKING:
    from eden.entities import Entity, RigidEntity
    from eden.envs.base import EnvBase

SIM_DT = 1.0 / 200.0
DECIMATION = 5
PI = 3.14159

FINGER_NAME_ALIASES = {
    "thumb": ("thumb", "th"),
    "index": ("index", "ff"),
    "middle": ("middle", "mid", "mf"),
    "ring": ("ring", "rf"),
    "pinky": ("pinky", "little", "lf"),
}


def _env_cache_masked_set(
    env: object,
    attr: str,
    env_ids: torch.Tensor,
    value: torch.Tensor | float | int,
) -> None:
    """Write ``value`` into ``getattr(env, attr)[env_ids]`` without inplace-updating inference tensors."""
    buf = getattr(env, attr, None)
    if buf is None:
        return
    if getattr(buf, "is_inference", lambda: False)():
        buf = buf.clone()
        setattr(env, attr, buf)
    if isinstance(value, torch.Tensor):
        buf[env_ids] = value.to(device=buf.device, dtype=buf.dtype)
    else:
        buf[env_ids] = value


def _name_matches_alias(name: str, alias: str) -> bool:
    """Match long aliases by substring and short aliases by token boundary."""
    normalized = name.lower()
    token = alias.lower()

    if len(token) >= 3:
        return token in normalized

    if normalized.startswith(token):
        return True

    for delimiter in ("_", "-", ".", "/"):
        if f"{delimiter}{token}" in normalized:
            return True

    return False


def filter_hand_dof_names(
    dof_names: list[str] | tuple[str, ...],
    *,
    excluded_fingers: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return DOF names excluding the provided semantic fingers."""
    excluded_aliases = []
    for finger in excluded_fingers:
        aliases = FINGER_NAME_ALIASES.get(finger)
        if aliases is None:
            raise ValueError(f"Unsupported finger: {finger}")
        excluded_aliases.extend(aliases)

    if not excluded_aliases:
        return tuple(dof_names)

    return tuple(name for name in dof_names if not any(_name_matches_alias(name, alias) for alias in excluded_aliases))


def frozen_link_names(
    link_names: list[str] | tuple[str, ...],
    dof_names: list[str] | tuple[str, ...],
    frozen_dofs: list[str] | tuple[str, ...],
) -> set[str]:
    """Return the subset of ``link_names`` belonging to fully-frozen fingers.

    A finger is considered frozen when it has at least one DOF and every one of
    its DOFs appears in ``frozen_dofs`` (as produced by a partial frozen hand
    controller). Such links are rigidly held relative to the palm, so tactile
    sensors mounted on them carry no learnable signal.
    """
    if not frozen_dofs:
        return set()
    frozen_set = set(frozen_dofs)
    frozen_aliases: list[str] = []
    for aliases in FINGER_NAME_ALIASES.values():
        finger_dofs = [name for name in dof_names if any(_name_matches_alias(name, alias) for alias in aliases)]
        if finger_dofs and all(name in frozen_set for name in finger_dofs):
            frozen_aliases.extend(aliases)
    if not frozen_aliases:
        return set()
    return {link for link in link_names if any(_name_matches_alias(link, alias) for alias in frozen_aliases)}


# ================== ACTIONS ==================


@ACTION_TERM_REGISTRY.register()
class PartialFrozenExplicitPDController(en.actions.ExplicitPDController):
    """Delta explicit PD on active DOFs while frozen DOFs hold their reset pose."""

    frozen_dofs: tuple[str, ...] = ()

    def build(self) -> None:
        super().build()
        if not self.frozen_dofs:
            self._frozen_dofs_idx_local = torch.empty(0, dtype=gs.tc_int, device=self.device)
            self._frozen_hold_pos = torch.empty(self.num_envs, 0, device=self.device, dtype=self._kp.dtype)
            return
        _, frozen_idx = self.entity.find_named_dofs_idx_local(
            list(self.frozen_dofs),
            name_scope=self.entity.dofs_name,
            preserve_order=True,
        )
        self._frozen_dofs_idx_local = torch.as_tensor(frozen_idx, dtype=gs.tc_int, device=self.device).contiguous()
        n_frozen = len(frozen_idx)
        self._frozen_hold_pos = torch.zeros(self.num_envs, n_frozen, device=self.device, dtype=self._kp.dtype)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if self._frozen_hold_pos.shape[-1] == 0:
            return
        pos = self.entity.get_dofs_pos(dofs_idx_local=self._frozen_dofs_idx_local)
        if envs_idx is None:
            envs_idx = slice(None)
        self._frozen_hold_pos[envs_idx] = pos[envs_idx]

    def apply_actions(self) -> None:
        super().apply_actions()
        self.entity.control_dofs_pos(self._frozen_hold_pos, dofs_idx_local=self._frozen_dofs_idx_local)


ACTION_MODIFIERS = Compose.configure(
    modifiers=[
        ConstantTorqueKick.configure(),
        MotorStrength.configure(),
        TorqueOffset.configure(),
        GearBacklash.configure(),
        Deadband.configure(),
    ]
)

HAND_CONTROLLER = en.actions.ExplicitPDController.configure(
    entity_name="robot",
    dofs_name=["*"],
    reference_source=ReferenceSource.DEFAULT,
    scale_ratio=0.5,
    modifier=ACTION_MODIFIERS,
)

PARTIAL_HAND_CONTROLLER = PartialFrozenExplicitPDController.configure(
    entity_name="robot",
    # dofs_name=["*"], # set by task mod
    reference_source=ReferenceSource.DEFAULT,
    scale_ratio=0.5,
    modifier=ACTION_MODIFIERS,
)

CONTROLLER_RANDOMIZATIONS: dict[str, EventTerm] = dict(
    dofs_randomize_bias=en.events.RandomizeStartupDofsPosBias.configure(
        mode=en.EventMode.RESET,
        action_terms_name=["dofs_pos_controller"],
        entity_name="robot",
        dofs_pos_range=(-0.01, 0.01),
    ),
    pd_gains=en.events.RandomizeKpKdGains.configure(
        mode=en.EventMode.RESET,
        entity_name="robot",
        kp_range=(0.95, 1.05),
        kd_range=(0.95, 1.05),
    ),
    motor_strength=en.events.RandomizeMotorStrength.configure(
        mode=en.EventMode.RESET,
        entity_name="robot",
        action_term_name="dofs_pos_controller",
        motor_strength_range=(0.95, 1.05),
    ),
    deadband=en.events.RandomizeDeadbandEpsilon.configure(
        mode=en.EventMode.RESET,
        entity_name="robot",
        action_term_name="dofs_pos_controller",
        deadband_epsilon_range=(0.0, 0.005),
    ),
    kick=en.events.RandomizeConstantTorqueKick.configure(
        mode=en.EventMode.RESET,
        entity_name="robot",
        action_term_name="dofs_pos_controller",
        torque_kick_range=(0.9, 1.1),
        apply_as_ratio=True,
    ),
    backlash=en.events.RandomizeGearBacklash.configure(
        mode=en.EventMode.RESET,
        entity_name="robot",
        action_term_name="dofs_pos_controller",
        backlash_range=(0.0, 0.01),
    ),
    torque_noise=en.events.RandomizeTorqueNoise.configure(
        mode=en.EventMode.INTERVAL,
        interval_range_s=(SIM_DT, SIM_DT),
        action_term_name="dofs_pos_controller",
        rfi_scale=0.1,
    ),
)


@ACTION_TERM_REGISTRY.register()
class RootPoseController(ActionTerm):
    """
    Action term that controls the root (base) 6 DOF pose via control_dofs_pos(dofs_idx_local=slice(0, 6)).

    The controlled entity should be a floating-base robot (is_fixed_base=False).

    Raw actions are 6D: [pos_x, pos_y, pos_z, euler_x, euler_y, euler_z], typically in [-1, 1].
    Each dimension is linearly mapped from [-1, 1] to the configured range (center ± scale * half_extent).
    Output is 6D: position (m) for DOFs 0–2, euler angles (rad) for DOFs 3–5.

    Parameters
    ----------
    entity_name : str
        The name of the entity to control.
    scale : float
        Scale applied to the raw action before mapping to ranges. Default 1.0.
    pos_x_range, pos_y_range, pos_z_range : tuple[float, float]
        (min, max) in meters for the root position (DOFs 0–2).
    euler_x_range, euler_y_range, euler_z_range : tuple[float, float]
        (min, max) in degrees for the root orientation (DOFs 3–5), interpreted
        relative to ``entity.default_root_quat`` and stored in radians.
    """

    entity_name: str = ""

    scale: float = 1.0
    pos_x_range: tuple[float, float] = (-0.5, 0.5)
    pos_y_range: tuple[float, float] = (-0.5, 0.5)
    pos_z_range: tuple[float, float] = (0.0, 1.0)
    euler_x_range: tuple[float, float] = (-180.0, 180.0)
    euler_y_range: tuple[float, float] = (-180.0, 180.0)
    euler_z_range: tuple[float, float] = (-180.0, 180.0)
    kp: float | tuple[float, float, float, float, float, float] = 80.0
    kd: float | tuple[float, float, float, float, float, float] = 12.0

    if TYPE_CHECKING:
        entity: RigidEntity

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._pos_center: torch.Tensor | None = None
        self._pos_half_extent: torch.Tensor | None = None
        self._pos_min: torch.Tensor | None = None
        self._pos_max: torch.Tensor | None = None
        self._euler_center_deg: torch.Tensor | None = None
        self._euler_half_extent_deg: torch.Tensor | None = None
        self._euler_min_deg: torch.Tensor | None = None
        self._euler_max_deg: torch.Tensor | None = None
        self._processed_action: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return 6

    def build(self) -> None:
        super().build()
        # Override base allocation: we use 6D action, not DOFs.
        self._n_dofs = 6
        self._raw_action = torch.zeros(self.num_envs, 6, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, 6, device=self.device)

        # Precompute center and half-extent for position (meters)
        self._pos_center = torch.tensor(
            [
                0.5 * (self.pos_x_range[0] + self.pos_x_range[1]),
                0.5 * (self.pos_y_range[0] + self.pos_y_range[1]),
                0.5 * (self.pos_z_range[0] + self.pos_z_range[1]),
            ],
            dtype=gs.tc_float,
            device=self.device,
        )
        self._pos_half_extent = torch.tensor(
            [
                0.5 * (self.pos_x_range[1] - self.pos_x_range[0]),
                0.5 * (self.pos_y_range[1] - self.pos_y_range[0]),
                0.5 * (self.pos_z_range[1] - self.pos_z_range[0]),
            ],
            dtype=gs.tc_float,
            device=self.device,
        )
        self._pos_min = torch.tensor(
            [self.pos_x_range[0], self.pos_y_range[0], self.pos_z_range[0]],
            dtype=gs.tc_float,
            device=self.device,
        )
        self._pos_max = torch.tensor(
            [self.pos_x_range[1], self.pos_y_range[1], self.pos_z_range[1]],
            dtype=gs.tc_float,
            device=self.device,
        )
        default_root_quat = torch.as_tensor(self.entity.default_root_quat, dtype=gs.tc_float, device=self.device)
        if default_root_quat.ndim > 1:
            default_root_quat = default_root_quat[0]
        default_euler_deg = torch.rad2deg(gu.quat_to_xyz(default_root_quat))

        # Euler in degrees
        self._euler_center_deg = (
            torch.tensor(
                [
                    0.5 * (self.euler_x_range[0] + self.euler_x_range[1]),
                    0.5 * (self.euler_y_range[0] + self.euler_y_range[1]),
                    0.5 * (self.euler_z_range[0] + self.euler_z_range[1]),
                ],
                dtype=gs.tc_float,
                device=self.device,
            )
            + default_euler_deg
        )
        self._euler_half_extent_deg = torch.tensor(
            [
                0.5 * (self.euler_x_range[1] - self.euler_x_range[0]),
                0.5 * (self.euler_y_range[1] - self.euler_y_range[0]),
                0.5 * (self.euler_z_range[1] - self.euler_z_range[0]),
            ],
            dtype=gs.tc_float,
            device=self.device,
        )
        self._euler_min_deg = (
            torch.tensor(
                [self.euler_x_range[0], self.euler_y_range[0], self.euler_z_range[0]],
                dtype=gs.tc_float,
                device=self.device,
            )
            + default_euler_deg
        )
        self._euler_max_deg = (
            torch.tensor(
                [self.euler_x_range[1], self.euler_y_range[1], self.euler_z_range[1]],
                dtype=gs.tc_float,
                device=self.device,
            )
            + default_euler_deg
        )
        self._processed_action = torch.zeros(self.num_envs, 6, device=self.device)
        self.entity.set_dofs_kp(self.kp, dofs_idx_local=slice(0, 6), envs_idx=slice(None))
        self.entity.set_dofs_kd(self.kd, dofs_idx_local=slice(0, 6), envs_idx=slice(None))

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        # pos (m): scale raw to range then clip to pos_*_range
        raw_pos = torch.clamp(self._raw_action[:, :3], -1.0, 1.0)
        self._processed_action[:, :3] = torch.clamp(
            self._pos_center + self.scale * raw_pos * self._pos_half_extent,
            self._pos_min,
            self._pos_max,
        )
        # euler (deg): scale raw to range, clip to euler_*_range, then convert to rad
        raw_euler = torch.clamp(self._raw_action[:, 3:6], -1.0, 1.0)
        euler_deg = torch.clamp(
            self._euler_center_deg + self.scale * raw_euler * self._euler_half_extent_deg,
            self._euler_min_deg,
            self._euler_max_deg,
        )
        self._processed_action[:, 3:6] = torch.deg2rad(euler_deg)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=slice(0, 6))


# ================== COMMANDS ==================


@COMMAND_TERM_REGISTRY.register()
class RotationAxisCommand(CommandTerm):
    """Command for target rotation axis.

    Generates a random axis around which the object should rotate.
    In HORA, this is typically the Z-axis (up).

    Parameters
    ----------
    axis_mode : str
        Mode for axis generation: "z_only", "random", or "xyz". Default: "z_only"
    """

    axis_mode: str = "z_only"

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        self.axis = torch.zeros(self.num_envs, 3, device=self.device)
        self.stats["rotation_axis"] = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The target rotation axis command. Shape is (num_envs, 3)."""
        return self.axis

    def _resample_command(self, envs_idx: torch.Tensor | slice):
        """Generate new rotation axis for specified environments."""
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return

        if self.axis_mode == "z_only":
            self.axis[envs_idx] = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32).repeat(
                n_envs, 1
            )
        elif self.axis_mode == "random":
            random_vecs = torch.randn(n_envs, 3, device=self.device)
            self.axis[envs_idx] = random_vecs / torch.norm(random_vecs, dim=1, keepdim=True)
        elif self.axis_mode == "xyz":
            axis_choices = torch.randint(0, 3, (n_envs,), device=self.device)
            axes = torch.zeros(n_envs, 3, device=self.device)
            axes[torch.arange(n_envs), axis_choices] = 1.0
            directions = torch.randint(0, 2, (n_envs,), device=self.device) * 2 - 1
            self.axis[envs_idx] = axes * directions.unsqueeze(1).float()

        self.stats["rotation_axis"][envs_idx] = self.axis[envs_idx]

    def _update_command(self):
        """Update command (no tracking needed for fixed axis)."""
        ...

    def _update_metrics(self):
        """Update command metrics."""
        ...


class BaseRotationCommand(CommandTerm):
    """
    Shared pieces for orientation goal commands: tracked quaternion, optional
    ``vis`` entity sync, and orientation error / success streak metrics.
    """

    entity_name: str = "obj"
    goal_entity_name: str | None = None
    update_goal_on_success: bool = True
    orientation_success_threshold: float = 0.1

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        self.object: Entity | None = None
        self.vis_object: Entity | None = None

        dtype = torch.float32
        self.quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=dtype)
        self.quat[:, 0] = 1.0

        self.rotation = torch.zeros(self.num_envs, 6, device=self.device, dtype=dtype)

        self.stats["orientation_error"] = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.stats["consecutive_success"] = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._last_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def build(self) -> None:
        super().build()
        self.object = self._env.entities[self.entity_name]
        if self.goal_entity_name is not None:
            if self.goal_entity_name in self._env.entities:
                self.vis_object = self._env.entities[self.goal_entity_name]
            else:
                print(f"Warning: goal_entity_name '{self.goal_entity_name}' not found in env entities.")
        else:
            self.vis_object = None

    @property
    def command(self) -> torch.Tensor:
        return self.rotation

    def _sync_goal_vis_orientation(
        self,
        envs_idx: slice | torch.Tensor,
    ) -> None:
        if self.vis_object is None:
            return
        self.vis_object.set_quat(self.quat[envs_idx], envs_idx=envs_idx)

    def _update_orientation_stats(self, *, advance_streak: bool = True) -> torch.Tensor:
        """Fill orientation metrics; returns per-env success mask."""
        assert self.object is not None
        obj_quat = self.object.get_quat()
        angle_diff = quat_error_magnitude(obj_quat, self.quat)
        self.stats["orientation_error"][:] = angle_diff
        success = angle_diff < self.orientation_success_threshold
        self._last_success = success
        if advance_streak:
            self.stats["consecutive_success"] += success.float()
        return success

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, float]:
        """Resample like :meth:`CommandTerm.reset`, then refresh ``orientation_error``.

        Parent reset clears stats to zero before the new goal is applied; reward is
        computed before the first post-reset :meth:`CommandTerm.compute`, so we
        recompute angle error here without advancing ``consecutive_success`` (the
        next :meth:`_update_command` pass handles the streak).
        """
        extras = super().reset(envs_idx=envs_idx)
        self._update_orientation_stats(advance_streak=False)
        return extras


@COMMAND_TERM_REGISTRY.register()
class ConstantOrientationCommand(BaseRotationCommand):
    """Target orientation: intrinsic XYZ Euler offset (degrees) composed with the object's live root quaternion."""

    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    update_goal_on_success: bool = False
    resampling_time_range: tuple[float, float] = (1e9, 1e9)  # Never resample

    def build(self) -> None:
        super().build()
        assert self.object is not None
        self._apply_goal_from_object_quat(slice(None))

    def _apply_goal_from_object_quat(self, envs_idx: slice | torch.Tensor) -> None:
        assert self.object is not None
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]
        if n_envs == 0:
            return
        base_quat = self.object.get_quat()[envs_idx]
        euler_t = torch.tensor(self.euler, device=self.device, dtype=base_quat.dtype).view(1, 3).expand(n_envs, 3)
        offset = gu.xyz_to_quat(euler_t, rpy=True, degrees=True)
        q = gu.transform_quat_by_quat(offset, base_quat)
        self.quat[envs_idx] = q
        self.rotation[envs_idx] = quat_to_rot6d(q)
        self._sync_goal_vis_orientation(envs_idx)

    def _resample_command(self, envs_idx: slice | torch.Tensor) -> None:
        self._apply_goal_from_object_quat(envs_idx)

    def _update_command(self) -> None:
        self._update_orientation_stats()


@COMMAND_TERM_REGISTRY.register()
class TargetRotationCommand(BaseRotationCommand):
    """
    Command generator for target quaternion orientation.

    Generates random quaternion targets by sampling random orientations
    around the X/Y/Z axes.

    Parameters
    ----------
    x_range : tuple[float, float], optional
        Range for rotation around x-axis in radians.
    y_range : tuple[float, float], optional
        Range for rotation around y-axis in radians.
    z_range : tuple[float, float], optional
        Range for rotation around z-axis in radians.
    """

    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    z_range: tuple[float, float] | None = None
    sample_relative: bool = False

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        # Tensors for sampling
        self._identity_quat = torch.as_tensor(((1.0, 0.0, 0.0, 0.0),), device=self.device, dtype=torch.float32)
        self._ranges_and_unit_vecs = []
        for i, sample_range in enumerate([self.x_range, self.y_range, self.z_range]):
            if sample_range is not None:
                unit_vec = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                unit_vec[:, i] = 1.0
                self._ranges_and_unit_vecs.append((sample_range, unit_vec))

    def __str__(self) -> str:
        msg = "TargetRotationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.resampling_time_range}\n"
        msg += f"\tX rotation range: {self.x_range}\n"
        msg += f"\tY rotation range: {self.y_range}"
        msg += f"\tZ rotation range: {self.z_range}"
        return msg

    def _resample_command(self, envs_idx: slice | torch.Tensor):
        """Resample the quaternion command for given environments.

        With ``sample_relative=True`` the axis-angle perturbations are composed onto
        the current goal quaternion instead of identity, so each resample is a small
        step from where the goal previously was (curriculum-friendly).
        """
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]
        if n_envs == 0:
            return

        if self.sample_relative:
            quat = self.quat[envs_idx].clone()
        else:
            quat = self._identity_quat.expand(n_envs, 4)
        for sample_range, unit_vec in self._ranges_and_unit_vecs:
            rand_angles = torch.empty(n_envs, device=self.device).uniform_(*sample_range)
            axis = unit_vec.expand(n_envs, 3)
            quat = gu.transform_quat_by_quat(quat, gu.axis_angle_to_quat(rand_angles, axis))
        self.quat[envs_idx] = quat
        self.rotation[envs_idx] = quat_to_rot6d(quat)
        self._sync_goal_vis_orientation(envs_idx)

    def _update_command(self) -> None:
        """Update the command based on goal success."""
        success = self._update_orientation_stats()
        if self.update_goal_on_success:
            goal_reset_ids = success.nonzero(as_tuple=False).squeeze(-1)
            self._resample(goal_reset_ids)


# ================== OBSERVATIONS ==================


@OBSERVATION_TERM_REGISTRY.register()
def base_rot6d(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Entity root orientation as 6D rotation (first two columns of the rotation matrix)."""
    return quat_to_rot6d(env.entities[entity_name].get_quat())


@OBSERVATION_TERM_REGISTRY.register()
def goal_rot6d_diff(
    env: EnvBase,
    *,
    command_name: str = "goal_quat",
    entity_name: str = "obj",
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame, as 6D rotation."""
    command = env.command_manager.get_term(command_name)
    obj = env.entities[entity_name]
    quat_diff = gu.transform_quat_by_quat(gu.inv_quat(obj.get_quat()), command.quat)
    return quat_to_rot6d(quat_diff)


@OBSERVATION_TERM_REGISTRY.register()
class OrientationErrorObs(ObservationTerm):
    """Angular error (rad) between the command goal quaternion and the tracked entity.

    Caches the last ``quat_error_magnitude`` and reuses it until ``command.quat`` or the object quaternion
    change, so multiple reward terms can read one computation per
    post-physics state (including after env resets resample the goal).
    """

    command_name: str = "goal_rot"
    entity_name: str | None = None

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self._cached_cmd: torch.Tensor | None = None
        self._cached_obj: torch.Tensor | None = None
        self._buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        self._cached_cmd = None
        self._cached_obj = None

    def compute(self, *args, **kwargs) -> torch.Tensor:
        env = self._env
        if env.command_manager is None:
            return self._buf
        command = env.command_manager.get_term(self.command_name)
        ent = self.entity_name if self.entity_name is not None else getattr(command, "entity_name", "obj")
        obj_quat = env.entities[ent].get_quat()
        if self._cached_cmd is not None and torch.allclose(self._cached_cmd, command.quat, rtol=0.0, atol=1e-7):
            if torch.allclose(self._cached_obj, obj_quat, rtol=0.0, atol=1e-7):
                return self._buf
        self._buf[:, 0] = quat_error_magnitude(obj_quat, command.quat)
        self._cached_cmd = command.quat.detach().clone()
        self._cached_obj = obj_quat.detach().clone()
        return self._buf


def _orientation_error_for_rewards(
    env: EnvBase,
    *,
    command_name: str,
    obs_term_name: str | None,
    entity_name: str | None,
) -> torch.Tensor:
    """Scalar angular error per env; either from ``OrientationErrorObs`` or a direct quaternion measure."""
    if obs_term_name is not None:
        raw = env.observation_manager.get_term(obs_term_name).compute()
        if raw.ndim == 2 and raw.shape[-1] == 1:
            return raw.squeeze(-1)
        return raw.reshape(env.num_envs)
    command = env.command_manager.get_term(command_name)
    ent = entity_name if entity_name is not None else getattr(command, "entity_name", "obj")
    return quat_error_magnitude(env.entities[ent].get_quat(), command.quat)


@OBSERVATION_TERM_REGISTRY.register()
class TactileSensorRead(ObservationTerm):
    """Read tactile observations from deploy extras when available, else sim sensors."""

    sensor_names: list[str] | None = None
    # How the within-step substep history (``history_length=DECIMATION``) is reduced
    # before the obs reaches the encoder. ``TactileSensorsMod`` writes this from the
    # ``--temporal_reduction`` CLI arg; every postprocess function supports every mode.
    temporal_reduction: TemporalReductionMode = "none"

    def build(self) -> None:
        all_sensor_names = list(self._env.sensors.keys())
        if self.sensor_names is None:
            matched_sensor_names = all_sensor_names
        else:
            _, matched_sensor_names = resolve_matching_names(
                self.sensor_names,
                all_sensor_names,
                preserve_order=True,
            )
        # Keep the matched names alongside the sensor handles so the deploy path
        # can map each sim sensor back to its attached fingertip link (and from
        # there to the real-hand finger id).
        self._sensor_names = list(matched_sensor_names)
        self._sensors = [(self._env.sensors[name], spec_for_sensor_name(name)) for name in matched_sensor_names]
        # Cache the per-sensor output feature count the sim path produces. The
        # deploy path's ``xhand1_func`` outputs a single per-finger value (1 for
        # ``bool``, 3 for ``agg_force``); we tile each per-finger real reading
        # up to ``_sim_per_sensor_size`` so the deploy obs matches the sim obs
        # the policy was trained on -- regardless of probe count, history
        # length, or temporal_reduction mode.
        self._sim_per_sensor_size: list[int] = []
        for sensor, spec in self._sensors:
            postprocess = spec.postprocess if spec is not None else postprocess_generic
            sim_feat = postprocess(
                sensor.read(),
                sensor=sensor,
                num_envs=self.num_envs,
                device=self.device,
                temporal_reduction=self.temporal_reduction,
            )
            self._sim_per_sensor_size.append(int(sim_feat.shape[-1]))

    def _read_sim_sensors(self) -> torch.Tensor:
        sensor_readings: list[torch.Tensor] = []
        for sensor, spec in self._sensors:
            postprocess = spec.postprocess if spec is not None else postprocess_generic
            sensor_readings.append(
                postprocess(
                    sensor.read(),
                    sensor=sensor,
                    num_envs=self.num_envs,
                    device=self.device,
                    temporal_reduction=self.temporal_reduction,
                )
            )
        return torch.cat(sensor_readings, dim=-1)

    def _compute_deploy_tactile(self) -> torch.Tensor | None:
        try:
            robot_entity = self._env.entities["robot"]
        except Exception:
            return None
        extra = getattr(robot_entity, "extra", None)
        if not isinstance(extra, dict):
            return None

        tactile_sensor_type = extra.get("tactile_sensor_type")
        spec = TACTILE_SENSORS.get(tactile_sensor_type) if tactile_sensor_type else None
        if spec is None or spec.xhand1_func is None:
            return None

        fingertip_sensors = extra.get("fingertip_sensors")
        if not isinstance(fingertip_sensors, dict):
            raise RuntimeError("Deploy tactile observation expects RobotState.extra['fingertip_sensors'].")

        tactile_bool_threshold = float(extra.get("tactile_bool_threshold", 0.0))
        if tactile_sensor_type in ("bool", "agg_bool") and tactile_bool_threshold < 0.0:
            raise RuntimeError(
                f"Deploy tactile {tactile_sensor_type!r} mode requires a positive tactile_bool_threshold."
            )

        # Drive iteration off the *sim* tactile sensors, not the real-hand fingertip_sensors
        # dict, so the deploy obs has exactly one entry per sim sensor in the same order
        # (and same total feature count) that ``_read_sim_sensors`` would produce at build
        # time. The real hand always reports all five fingertips; if the sim config only
        # sensores a subset (e.g. a partial-hand controller freezing fingers), reading every
        # real finger would emit more features than the pre-allocated obs buffer.
        from tactile_compare import FINGER_IDS  # late import: src/ is a flat package

        # ``fingertip_links`` lives on the robot *options*' metadata (a ClassVar
        # on e.g. ``SharpaHand`` / ``XHand1``), not on the runtime entity, so
        # reach back through ``env.scene_options`` to get it.
        robot_options = getattr(getattr(self._env, "scene_options", None), "robot", None)
        robot_meta = getattr(robot_options, "metadata", None)
        fingertip_links = list(getattr(robot_meta, "fingertip_links", None) or ())
        # Canonical position i in the robot's fingertip_links corresponds to FINGER_IDS[i]
        # on the xhand1 (thumb, index, mid, ring, pinky); see tactile_compare.FINGER_NAMES.
        link_to_finger_id = {link: fid for link, fid in zip(fingertip_links, FINGER_IDS)}
        prefix = f"tactile_{tactile_sensor_type}_"

        sensor_readings: list[torch.Tensor] = []
        for sensor_name, (_sensor, sensor_spec), sim_size in zip(
            self._sensor_names, self._sensors, self._sim_per_sensor_size, strict=True
        ):
            if sensor_spec is None or sensor_spec.xhand1_func is None:
                continue
            link = sensor_name.removeprefix(prefix)
            finger_id = link_to_finger_id.get(link)
            if finger_id is None:
                raise RuntimeError(
                    f"Sim tactile sensor {sensor_name!r} has no matching real-hand finger; "
                    f"link {link!r} is not in robot.metadata.fingertip_links={fingertip_links!r}."
                )
            reading = fingertip_sensors.get(finger_id)
            if not isinstance(reading, dict):
                raise RuntimeError(
                    f"Real hand did not report fingertip_sensors[{finger_id}] for sim sensor {sensor_name!r}."
                )
            deploy_reading = sensor_spec.xhand1_func(
                reading, device=self.device, threshold=tactile_bool_threshold
            )
            # Tile the per-finger real reading up to the sim's per-sensor feature count.
            # The real hand can only resolve one fingertip-aggregate value per finger
            # (``calc_pressure`` for agg_force, the threshold bit for bool), so every
            # taxel/substep slot on that finger gets the same value. ``repeat_interleave``
            # along the last axis lays the deploy features in [v0_repeat, v1_repeat, ...]
            # order, matching how the sim postprocess interleaves the history axis into
            # each innermost feature.
            deploy_size = int(deploy_reading.shape[-1])
            if sim_size % deploy_size != 0:
                raise RuntimeError(
                    f"Cannot tile deploy reading for {sensor_name!r}: sim per-sensor size "
                    f"{sim_size} is not a multiple of deploy per-finger size {deploy_size}."
                )
            factor = sim_size // deploy_size
            if factor > 1:
                deploy_reading = deploy_reading.repeat_interleave(factor, dim=-1)
            sensor_readings.append(deploy_reading)

        if not sensor_readings:
            return None

        tactile_obs = torch.cat(sensor_readings, dim=-1)
        if tactile_obs.shape[0] != self.num_envs:
            tactile_obs = tactile_obs.expand(self.num_envs, -1)
        return tactile_obs

    def compute(self, *args, **kwargs) -> torch.Tensor:
        data = self._compute_deploy_tactile()
        self._cached = data if data is not None else self._read_sim_sensors()
        return self._cached


@OBSERVATION_TERM_REGISTRY.register()
class CachedObs(ObservationTerm):
    default_value: float = 0.0

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self.dtype = torch.float32
        self._cached = torch.full((self.num_envs, 1), self.default_value, device=self.device, dtype=self.dtype)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        return self._cached


# ================== REWARDS ==================


def _tc_quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Batched torch rotation vector matching ``genesis.utils.geom.quat_to_rotvec`` (scalar-first quat)."""
    q_w = quat[..., :1]
    q_vec = quat[..., 1:4]
    s2 = torch.linalg.norm(q_vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s2, torch.abs(q_w))
    inv_sinc = angle / torch.clamp(s2, min=gs.EPS)
    sgn = torch.where(q_w < 0.0, -1.0, 1.0).to(dtype=quat.dtype)
    return (sgn * inv_sinc) * q_vec


@REWARD_TERM_REGISTRY.register()
class AxisRotationProgressReward(RewardTerm):
    """Signed spin rate (rad/s) about a fixed object-frame axis.

    The reward tracks the object quaternion internally, accumulates the total
    signed rotation about the configured object axis, and exposes the last per-step
    signed increment for subclasses.
    """

    entity_name: str | None = None
    command_name: str = "goal_rot"
    axis_world: tuple[float, float, float] | None = None

    def build(self) -> None:
        super().build()
        cmd = self._env.command_manager.get_term(self.command_name)
        resolved_entity_name = self.entity_name if self.entity_name is not None else getattr(cmd, "entity_name", "obj")
        self._obj = self._env.entities[resolved_entity_name]

        obj_quat = self._obj.get_quat()
        dtype = obj_quat.dtype
        axis = self.axis_world if self.axis_world is not None else getattr(cmd, "rotation_axis_world", None)
        if axis is None:
            raise ValueError(
                f"{type(self).__name__}: command {self.command_name!r} has no 'rotation_axis_world'; "
                "pass axis_world=... in options."
            )
        axis_t = torch.as_tensor(axis, device=self.device, dtype=dtype).view(1, 3)
        self._axis_world = axis_t / (torch.linalg.norm(axis_t, dim=-1, keepdim=True) + 1e-9)
        self._axis_obj = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)

        self._prev_quat = obj_quat.detach().clone()
        self._buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=dtype)
        self._last_inc = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._last_reward = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._cached_obj: torch.Tensor | None = None
        self._refresh_axis_obj(slice(None), obj_quat)

    def _refresh_axis_obj(self, envs_idx: slice | torch.Tensor, obj_quat: torch.Tensor) -> None:
        axis_world = self._axis_world.expand(self.num_envs, -1)
        axis_obj = gu.transform_by_quat(axis_world[envs_idx], gu.inv_quat(obj_quat[envs_idx]))
        axis_obj = axis_obj / (torch.linalg.norm(axis_obj, dim=-1, keepdim=True) + 1e-9)
        self._axis_obj[envs_idx] = axis_obj

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        self._cached_obj = None
        if envs_idx is None:
            envs_idx = slice(None)
        cur = self._obj.get_quat()
        self._prev_quat[envs_idx] = cur[envs_idx].detach().clone()
        self._refresh_axis_obj(envs_idx, cur)
        self._buf[envs_idx] = 0.0
        self._last_inc[envs_idx] = 0.0
        self._last_reward[envs_idx] = 0.0

    def _step_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj_quat = self._obj.get_quat()
        if self._cached_obj is not None and torch.allclose(self._cached_obj, obj_quat, rtol=0.0, atol=1e-7):
            return obj_quat, torch.zeros(self.num_envs, 3, device=self.device, dtype=obj_quat.dtype), self._last_inc

        quat_diff = gu.transform_quat_by_quat(gu.inv_quat(self._prev_quat), obj_quat)
        rotvec = _tc_quat_to_rotvec(quat_diff)
        axis = self._axis_obj
        inc = (rotvec * axis).sum(dim=-1)
        return obj_quat, rotvec, inc

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        obj_quat, _, inc = self._step_terms()
        if self._cached_obj is not None and torch.allclose(self._cached_obj, obj_quat, rtol=0.0, atol=1e-7):
            return self._last_reward

        dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        reward = inc / dt
        self._buf[:, 0] = self._buf[:, 0] + inc.to(dtype=self._buf.dtype)
        self._last_inc.copy_(inc.detach().to(dtype=self._last_inc.dtype))
        self._last_reward.copy_(reward.detach().to(dtype=self._last_reward.dtype))
        self._prev_quat.copy_(obj_quat.detach())
        self._cached_obj = obj_quat.detach().clone()
        return reward

    @property
    def last_step_signed_axis_rad(self) -> torch.Tensor:
        """Signed rotation increment about the axis for the last fresh :meth:`compute` (rad)."""
        return self._last_inc

    @property
    def total_signed_axis_rad(self) -> torch.Tensor:
        """Cumulative signed rotation about the fixed axis (rad)."""
        return self._buf[:, 0]


@REWARD_TERM_REGISTRY.register()
def surface_distance_reward(
    env: EnvBase, *, obs_name: str = "surface_distance", nearest_k: int = -1, sigma: float = 0.1
) -> torch.Tensor:
    """
    Reward based on surface distance sensor readings.
    ``obs_name`` should be a SensorRead term that reads SurfaceDistanceProbe sensors.

    Parameters
    ----------
    nearest_k : int
        Use only the ``nearest_k`` smallest per-channel distances (closest sensors). If <= 0,
        use all channels (same as summing every fingertip/probe).
    sigma : float
        Standard deviation of the Gaussian kernel.
    """
    sensor_obs_term = env.observation_manager.get_term(obs_name)
    # Fresh read: ``_cached`` is only updated during ``observation_manager.compute``, which runs after rewards.
    readings = sensor_obs_term.compute()
    dist_sq = readings**2
    n = dist_sq.shape[-1]
    k = n if nearest_k <= 0 else min(nearest_k, n)
    if k >= n:
        nearest_dist_sq = dist_sq
    else:
        nearest_dist_sq, _ = torch.topk(dist_sq, k=k, largest=False, dim=-1)
    return torch.exp(-nearest_dist_sq / sigma**2).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
class ForceMagnitudePenalty(RewardTerm):
    """
    Reward based on force sensor magnitude.
    """

    sensor_name: str = "force"
    threshold: float = 0.5
    clip: float = 10.0

    def build(self) -> None:
        self._sensor = self._env.sensors[self.sensor_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        readings = self._sensor.read(envs_idx)
        magnitudes = torch.norm(readings, dim=-1) - self.threshold
        return torch.clamp(magnitudes, min=0.0, max=self.clip)


@REWARD_TERM_REGISTRY.register()
def orientation_success_bonus(
    env: EnvBase,
    *,
    command_name: str = "goal_rot",
) -> torch.Tensor:
    """Bonus reward for reaching the goal orientation, using the command's own success criterion."""
    return env.command_manager.get_term(command_name)._last_success.float()


@REWARD_TERM_REGISTRY.register()
def rotation_reward(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    command_name: str = "rotation_axis",
    local_axis: tuple[float, float, float] | None = None,
    angvel_clip_min: float = -0.5,
    angvel_clip_max: float = 0.5,
) -> torch.Tensor:
    """Reward for rotating object around target axis.

    Based on HORA implementation: measures angular velocity component
    along the target rotation axis.

    Parameters
    ----------
    entity_name : str
        Name of the object entity. Default: "obj"
    command_name : str
        Name of the rotation axis command. Default: "rotation_axis"
    local_axis : tuple[float, float, float] | None
        If set, an axis fixed in the entity's local frame; it is rotated by the entity's
        current quaternion each step so the reward tracks an object-attached axis (e.g.
        a screwdriver's shaft). Overrides ``command_name`` when provided. Default: None
        (use the world-frame command axis).
    angvel_clip_min : float
        Minimum angular velocity for clipping. Default: -0.5
    angvel_clip_max : float
        Maximum angular velocity for clipping. Default: 0.5
    """
    obj = env.entities[entity_name]

    # Get current and previous object rotation
    obj_quat = obj.get_quat()

    # Initialize previous quat buffer if needed
    if not hasattr(env, "_prev_obj_quat"):
        env._prev_obj_quat = obj_quat.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute angular difference (quat_mul(curr, conjugate(prev)))
    quat_diff = quat_mul(obj_quat, quat_conjugate(env._prev_obj_quat))

    # Angular velocity = axis_angle / dt
    axis_angle = axis_angle_from_quat(quat_diff)
    dt = env.env_options.sim_dt * env.env_options.decimation
    ang_vel = axis_angle / dt

    if local_axis is not None:
        local = torch.tensor(local_axis, device=env.device, dtype=obj_quat.dtype).expand(env.num_envs, 3)
        target_axis = gu.transform_by_quat(local, obj_quat)
    else:
        target_axis = env.command_manager.get_command(command_name)

    # Compute dot product: positive when rotating in correct direction
    vec_dot = (ang_vel * target_axis).sum(dim=-1)

    # Clip to configured range (HORA: [-0.5, 0.5])
    rotation_reward = torch.clip(vec_dot, min=angvel_clip_min, max=angvel_clip_max)

    # Store for next iteration
    env._prev_obj_quat = obj_quat.clone()

    return rotation_reward


@REWARD_TERM_REGISTRY.register()
def termination_penalty(
    env: EnvBase,
    *,
    termination_names: tuple[str, ...],
) -> torch.Tensor:
    """One-shot penalty (1.0 in envs terminated by any of ``termination_names``, else 0.0).

    Reads cached per-term done tensors from ``env.termination_manager`` — termination is
    computed before reward each step (``EnvBase`` step order), so no work is repeated.
    """
    tm = env.termination_manager
    fired = tm.get_term(termination_names[0]).clone()
    for name in termination_names[1:]:
        fired |= tm.get_term(name)
    return fired.float()


@REWARD_TERM_REGISTRY.register()
def track_orientation_inv_l2(
    env: EnvBase,
    *,
    command_name: str = "goal_quat",
    obs_term_name: str | None = None,
    entity_name: str | None = None,
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the object orientation using inverse orientation error.

    Pass ``obs_term_name`` (e.g. ``\"goal_angular_dist\"``) to reuse :class:`OrientationErrorObs` so rewards
    share one ``quat_error_magnitude`` with observations. If ``None``, the error is computed from quaternions.
    """
    angle_diff = _orientation_error_for_rewards(
        env, command_name=command_name, obs_term_name=obs_term_name, entity_name=entity_name
    )
    return 1.0 / (angle_diff + rot_eps)


@REWARD_TERM_REGISTRY.register()
def track_orientation_gaussian(
    env: EnvBase,
    *,
    command_name: str = "goal_quat",
    obs_term_name: str | None = None,
    entity_name: str | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Reward for tracking the object orientation with a Gaussian on the angle error.

    Returns ``exp(-err^2 / sigma^2)``, monotonically decreasing with error and
    bounded in ``(0, 1]``. Unlike the inverse-L2 form, this gives a non-degenerate
    gradient across the full error range, which is important when the policy starts
    far from the goal.
    """
    angle_diff = _orientation_error_for_rewards(
        env, command_name=command_name, obs_term_name=obs_term_name, entity_name=entity_name
    )
    return torch.exp(-(angle_diff**2) / (sigma**2))


@REWARD_TERM_REGISTRY.register()
def object_dist_penalty(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    target_pos: tuple[float, float, float] | None = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Distance penalty: penalizes object moving away from hand.

    With ``margin > 0``, distances at or below ``margin`` produce zero penalty (deadzone);
    beyond that, the penalty grows linearly as ``dist - margin``.
    """
    obj = env.entities[entity_name]
    obj_pos = obj.get_pos()

    hand_pos = torch.tensor(target_pos or obj.default_root_pos, device=obj_pos.device, dtype=obj_pos.dtype).unsqueeze(0)
    dist = torch.norm(obj_pos - hand_pos, dim=-1)
    return torch.clamp(dist - margin, min=0.0)


@REWARD_TERM_REGISTRY.register()
def pose_diff_penalty(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Penalty for deviating from initial hand pose."""
    robot = env.entities[entity_name]

    if not hasattr(env, "_init_hand_pose"):
        env._init_hand_pose = robot.get_dofs_pos(robot.dofs_idx_local).clone()

    current_pose = robot.get_dofs_pos(robot.dofs_idx_local)
    pose_diff = current_pose - env._init_hand_pose

    return (pose_diff**2).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def work_penalty(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Penalty for mechanical work done (power squared), normalized by DOF count.

    Dividing by the number of robot DOFs keeps the term roughly comparable across
    hands with different DOF counts (e.g. xhand1 vs. shadow vs. allegro).
    """
    robot = env.entities[entity_name]

    control_forces = robot.get_dofs_control_force(robot.dofs_idx_local)
    dof_vel = robot.get_dofs_vel(robot.dofs_idx_local)

    work = (control_forces * dof_vel).sum(dim=-1)
    return work**2 / control_forces.shape[-1]


@REWARD_TERM_REGISTRY.register()
def ee_work_penalty(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Penalty for mechanical work done by the ee_controller (floating-base DOFs 0:6)."""
    robot = env.entities[entity_name]
    ee_slice = slice(0, 6)

    control_forces = robot.get_dofs_control_force(ee_slice)
    dof_vel = robot.get_dofs_vel(ee_slice)

    work = (control_forces * dof_vel).sum(dim=-1)
    return work**2 / control_forces.shape[-1]


# ================== METRICS ==================


@METRIC_TERM_REGISTRY.register(
    direction=MetricDirection.HIB,
    metric_mode=MetricMode.INTERVAL,
)
def episode_reward_metric(env: EnvBase, *, reward_names: list[str], weights: list[float]) -> torch.Tensor:
    total_reward = torch.zeros(env.num_envs, device=env.device)
    for reward_name, metric_weight in zip(reward_names, weights):
        term_idx = env.reward_manager._term_names.index(reward_name)
        term_weight = env.reward_manager._weights[term_idx]
        episode_sum = env.reward_manager._episode_sums[:, term_idx].float()
        total_reward += episode_sum / env.max_episode_length_s / term_weight * metric_weight
    return total_reward


@METRIC_TERM_REGISTRY.register(
    direction=MetricDirection.HIB,
    metric_mode=MetricMode.RESET,
)
class UpdateCurriculumWeights(MetricTerm):
    """Linearly interpolates selected reward weights using ``env.common_step_counter``.

    Uses :class:`MetricMode.RESET` so this side-effect term does not participate in
    interval success aggregation with other metrics (e.g. ``objective``).
    """

    curriculum: dict[str, float] | None = None
    curriculum_step_start: int = 0
    curriculum_step_end: int = 10000

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env, options)
        self._rew_ends = torch.tensor([v for v in self.curriculum.values()], device=self.device)
        self._rew_starts = torch.zeros_like(self._rew_ends)

    def build(self):
        super().build()
        for i, term_name in enumerate(self.curriculum.keys()):
            self._rew_starts[i] = self._env.reward_manager.get_term(term_name).weight

    def compute(self) -> torch.Tensor:
        span = max(self.curriculum_step_end - self.curriculum_step_start, 1)
        steps = self._env.common_step_counter
        t = min(1.0, max(0.0, (steps - self.curriculum_step_start) / span))
        rew_weights = torch.lerp(self._rew_starts, self._rew_ends, t)
        for i, term_name in enumerate(self.curriculum.keys()):
            self._env.reward_manager.get_term(term_name).weight = float(rew_weights[i].item())

        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)


# ================== EVENTS ==================


@EVENT_TERM_REGISTRY.register()
class SetSampledBottomAlignedPos(EventTerm):
    """Place the entity at a sampled xy with bottom-aligned z using precomputed AABB offsets."""

    entity_name: str = "obj"
    pos_z: float = 0.5
    range_x: tuple[float, float] = (-0.02, 0.02)
    range_y: tuple[float, float] = (-0.02, 0.02)
    range_yaw: tuple[float, float] = (-PI, PI)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self._entity: Entity | None = None
        self.offset_z: torch.Tensor | None = None
        self.quat: torch.Tensor | None = None

    def build(self) -> None:
        super().build()
        self._entity = self._env.entities[self.entity_name]
        _, self.quat = self._entity.get_default_root_pose()
        self._entity.set_quat(self.quat)
        aabb = self._entity.get_AABB()
        half_height_z = (aabb[:, 1, 2] - aabb[:, 0, 2]) / 2.0
        self.offset_z = half_height_z + self.pos_z

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]
        if n_envs == 0:
            return

        pos_z = self.offset_z[envs_idx]
        pos_x = sample_uniform(*self.range_x, size=n_envs, device=self.device)
        pos_y = sample_uniform(*self.range_y, size=n_envs, device=self.device)
        pos = torch.stack([pos_x, pos_y, pos_z], dim=-1)
        self._entity.set_pos(pos, envs_idx=envs_idx)

        yaw = sample_uniform(*self.range_yaw, size=n_envs, device=self.device)
        yaw_eulers = torch.zeros((n_envs, 3), device=self.device)
        yaw_eulers[:, 2] = yaw
        yaw_quat = gu.xyz_to_quat(yaw_eulers, rpy=True, degrees=False)
        self._entity.set_quat(gu.transform_quat_by_quat(self.quat[envs_idx], yaw_quat), envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class LoadGraspPose(EventTerm):
    entity_name: str = "robot"
    obj_name: str = "obj"
    action_term_name: str = "dofs_pos_controller"
    # Map from ROBOT_REGISTRY hand name (e.g. "allegro", "shadow") to the
    # precomputed-grasp file for this task. The entry matching the robot
    # actually loaded into the env is selected at build time.
    grasps_paths: dict[str, str] = {}

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)

        self.robot: Entity | None = None
        self.obj: Entity | None = None
        self.action_term: ActionTerm | None = None
        self._robot_dof_name_to_grasp_idx: dict[str, int] = {}

    def build(self) -> None:
        super().build()

        # Store references to entities
        self.robot = self._env.entities[self.entity_name]
        self.obj = self._env.entities[self.obj_name]
        action_manager = getattr(self._env, "action_manager", None)
        self.action_term = action_manager.get_term(self.action_term_name) if action_manager is not None else None
        self._robot_dof_name_to_grasp_idx = {name: i for i, name in enumerate(self.robot.dofs_name)}

        # Cache the sampled grasp target on the env because several downstream
        # screwdriver terms read it after reset: screwdriver frozen pinkie/ring-grasp control
        # using ResetRobotFromCachedGrasp.
        self._env._sampled_grasp_indices = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._env._sampled_grasp_joint_pos = torch.zeros(
            (self.num_envs, len(self.robot.dofs_name)), device=self.device, dtype=gs.tc_float
        )
        self._env._sampled_grasp_robot_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self._env._sampled_grasp_robot_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self._env._sampled_grasp_obj_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self._env._sampled_grasp_obj_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self._env._sampled_grasp_has_robot_root_pose = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._env._sampled_grasp_has_obj_pose = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Pick the grasp file for the hand currently loaded in the env.
        from registry import ROBOT_REGISTRY

        robot_options_cls = type(self.robot.options)
        hand_name = next((n for n, c in ROBOT_REGISTRY if c is robot_options_cls), None)
        self.grasps_path = self.grasps_paths.get(hand_name, "") if hand_name is not None else ""
        if self.grasps_path and Path(self.grasps_path).exists():
            grasp_data = torch.load(self.grasps_path, map_location=self.device)
            self.grasp_joint_pos = grasp_data["joint_pos"]  # (N, num_dofs)
            self.grasp_robot_pos = grasp_data.get("robot_pos")  # (N, 3) optional
            self.grasp_robot_quat = grasp_data.get("robot_quat")  # (N, 4) optional
            self.grasp_obj_pos = grasp_data.get("obj_pos")  # (N, 3) optional
            self.grasp_obj_quat = grasp_data.get("obj_quat")  # (N, 4) optional
            self.num_grasps = len(self.grasp_joint_pos)
            assert self.grasp_joint_pos.shape[1] == len(self.robot.dofs_name)
            self.has_robot_root_pose = self.grasp_robot_pos is not None and self.grasp_robot_quat is not None
            assert self.has_robot_root_pose or (self.grasp_robot_pos is None and self.grasp_robot_quat is None), (
                "Grasp file must provide both robot_pos and robot_quat, or neither."
            )
            if self.has_robot_root_pose:
                assert self.num_grasps == len(self.grasp_robot_pos) == len(self.grasp_robot_quat)
                assert self.grasp_robot_pos.shape[1] == 3
                assert self.grasp_robot_quat.shape[1] == 4
            self.has_obj_pose = self.grasp_obj_pos is not None and self.grasp_obj_quat is not None
            assert self.has_obj_pose or (self.grasp_obj_pos is None and self.grasp_obj_quat is None), (
                "Grasp file must provide both obj_pos and obj_quat, or neither."
            )
            if self.has_obj_pose:
                assert self.num_grasps == len(self.grasp_obj_pos) == len(self.grasp_obj_quat)
                assert self.grasp_obj_pos.shape[1] == 3
                assert self.grasp_obj_quat.shape[1] == 4

            grasp_target_link_name = getattr(self.obj.options.metadata, "grasp_target_link", None)
            if self.has_obj_pose:
                self.obj_pose_frame = grasp_data.get("obj_pose_frame")
                if self.obj_pose_frame is None:
                    # Backward compatibility for older files that saved grasp-target link pose as obj_pos.
                    self.obj_pose_frame = "root"
                    if grasp_target_link_name is not None:
                        root_pos = self.obj.get_pos()[0]
                        target_pos = resolve_entity_link(self.obj, grasp_target_link_name).get_pos()[0]
                        mean_saved_pos = self.grasp_obj_pos.float().mean(dim=0)
                        if torch.norm(mean_saved_pos - target_pos) + 1e-6 < torch.norm(mean_saved_pos - root_pos):
                            self.obj_pose_frame = "grasp_target_link"
                            print("Detected legacy grasp file with grasp-target object pose. Converting to root pose.")

                assert self.obj_pose_frame in {"root", "grasp_target_link"}, (
                    f"Unsupported obj_pose_frame in grasp file: {self.obj_pose_frame}"
                )
                self.obj_pose_needs_conversion = self.obj_pose_frame == "grasp_target_link"
                if self.obj_pose_needs_conversion:
                    assert grasp_target_link_name is not None, (
                        "obj_pose_frame='grasp_target_link' requires grasp_target_link."
                    )

                    root_pos = self.obj.get_pos()[:1]
                    root_quat = self.obj.get_quat()[:1]
                    target = resolve_entity_link(self.obj, grasp_target_link_name)
                    target_pos = target.get_pos()[:1]
                    target_quat = target.get_quat()[:1]

                    root_quat_inv = gu.inv_quat(root_quat)
                    self._target_local_pos = gu.transform_by_quat(target_pos - root_pos, root_quat_inv)
                    self._target_local_quat = gu.transform_quat_by_quat(target_quat, root_quat_inv)
                    self._target_local_quat_inv = gu.inv_quat(self._target_local_quat)
            else:
                self.obj_pose_frame = None
                self.obj_pose_needs_conversion = False

            # Check if object is heterogeneous (different mesh per env)
            self.obj_is_heterogeneous = "geom_idx" in grasp_data
            if self.obj_is_heterogeneous:
                self.grasp_geom_idx = grasp_data["geom_idx"]  # (N,)
                assert len(self.grasp_geom_idx) == self.num_grasps

                # Build env-to-geom mapping
                self.env_to_geom_idx = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)
                for geom in self.obj.links[0].geoms:
                    active_envs = geom.active_envs_idx
                    self.env_to_geom_idx[active_envs] = geom.idx

                # Build grasp indices per geom for efficient sampling
                env_unique_geom_indices = torch.unique(self.env_to_geom_idx)
                grasp_unique_geom_indices = torch.unique(self.grasp_geom_idx)
                # When num_envs is smaller than the number of heterogeneous object
                # variants, only a prefix subset of the variants gets an active env.
                # That is fine — we just map the env geoms onto the first N grasp
                # geoms positionally and ignore the variants no env uses. Erroring
                # only makes sense when the env has MORE geoms than the grasp file
                # covers, since then some env objects have no grasps at all.
                assert len(env_unique_geom_indices) <= len(grasp_unique_geom_indices), (
                    "More env object geoms than grasp geoms; grasp file does not cover all object variants."
                )

                env_sorted = torch.sort(env_unique_geom_indices).values
                grasp_sorted = torch.sort(grasp_unique_geom_indices).values
                # zip truncates to the shorter (env) list when num_envs < num variants.
                self.env_to_grasp_geom_idx = {
                    int(env_geom.item()): int(grasp_geom.item())
                    for env_geom, grasp_geom in zip(env_sorted, grasp_sorted)
                }

                self.grasps_per_geom = {}
                for grasp_geom_idx in grasp_sorted:
                    mask = self.grasp_geom_idx == grasp_geom_idx.item()
                    self.grasps_per_geom[grasp_geom_idx.item()] = torch.where(mask)[0]

            print(f"Loaded {self.num_grasps} precomputed grasps from {self.grasps_path}")
        else:
            print(f"[LoadGraspPose] Warning: grasp file not found: '{self.grasps_path}'. Event will be a no-op.")
            self.num_grasps = 0
            self.has_robot_root_pose = False
            self.has_obj_pose = False
            self.obj_is_heterogeneous = False
            self.obj_pose_needs_conversion = False

    def _env_ids_from_index(self, envs_idx: slice | torch.Tensor) -> torch.Tensor:
        if isinstance(envs_idx, slice):
            return torch.arange(self.num_envs, device=self.device)[envs_idx]
        return envs_idx

    def _resolve_primary_obj_dof_index(self) -> int | None:
        if self.obj is None:
            return None

        dof_names = list(getattr(getattr(self.obj, "options", None), "dofs_name", ()) or ())
        runtime_dof_names = list(getattr(self.obj, "dofs_name", ()) or ())
        for name in runtime_dof_names:
            if name not in dof_names:
                dof_names.append(name)

        preferred_names = ("handle_to_shaft", "cylindrical_1_2", "nut", "screw")
        for preferred_name in preferred_names:
            try:
                return dof_names.index(preferred_name)
            except ValueError:
                continue

        if len(dof_names) == 1:
            return 0
        return None

    def _reset_object_dof_state(self, envs_idx: slice | torch.Tensor, n_envs: int) -> None:
        if self.obj is None:
            return

        dofs_idx_local = getattr(self.obj, "dofs_idx_local", None)
        if dofs_idx_local is None or len(dofs_idx_local) == 0:
            return

        env_ids = self._env_ids_from_index(envs_idx)
        default_dofs_pos = getattr(self.obj, "default_dofs_pos", None)
        if isinstance(default_dofs_pos, torch.Tensor) and default_dofs_pos.numel() > 0:
            if default_dofs_pos.ndim == 1:
                dofs_pos = default_dofs_pos.unsqueeze(0).expand(n_envs, -1)
            elif default_dofs_pos.shape[0] == self.num_envs:
                dofs_pos = default_dofs_pos[env_ids]
            elif default_dofs_pos.shape[0] == 1:
                dofs_pos = default_dofs_pos.expand(n_envs, -1)
            else:
                dofs_pos = default_dofs_pos[:n_envs]
            dofs_pos = dofs_pos.to(device=self.device, dtype=gs.tc_float)
        else:
            dofs_pos = torch.zeros((n_envs, len(dofs_idx_local)), device=self.device, dtype=gs.tc_float)

        # Some tasks read articulated object joint state directly into observations
        # (for example screwdriver `screw_joint_pos` / `screw_joint_vel`). When a
        # grasp reset only restores the object root pose, those DOF tensors can stay
        # stale across resets and leak NaNs into the next rollout step.
        dofs_vel = torch.zeros_like(dofs_pos)
        self.obj.set_dofs_pos(dofs_pos, dofs_idx_local, envs_idx)
        self.obj.set_dofs_vel(dofs_vel, dofs_idx_local, envs_idx, skip_forward=True)

        primary_dof_idx = self._resolve_primary_obj_dof_index()
        if primary_dof_idx is None or primary_dof_idx >= dofs_pos.shape[1]:
            return

        _env_cache_masked_set(self._env, "_init_screw_dof_pos", env_ids, dofs_pos[:, primary_dof_idx])
        _env_cache_masked_set(self._env, "_cached_screw_axis_pos", env_ids, 0.0)
        _env_cache_masked_set(self._env, "_cached_screw_dof_vel", env_ids, 0.0)
        _env_cache_masked_set(self._env, "_screw_axis_state_step_token", env_ids, -1)

    def _offset_from_configured_action_offset(self, dof_names: list[str] | tuple[str, ...]) -> torch.Tensor:
        assert self.action_term is not None
        offset = getattr(self.action_term, "offset", 0.0)
        if isinstance(offset, (int, float)):
            return torch.full((1, len(dof_names)), float(offset), device=self.device, dtype=gs.tc_float)
        out = torch.zeros((1, len(dof_names)), device=self.device, dtype=gs.tc_float)
        if not isinstance(offset, dict):
            return out
        for dofs_name, value in offset.items():
            matched_idx, _ = resolve_matching_names(dofs_name, dof_names)
            for idx in matched_idx:
                out[:, idx] = float(value)
        return out

    def _sampled_pos_for_dofs(
        self,
        sampled_joint_pos: torch.Tensor,
        dof_names: list[str] | tuple[str, ...],
    ) -> torch.Tensor | None:
        indices = []
        for name in dof_names:
            grasp_idx = self._robot_dof_name_to_grasp_idx.get(str(name))
            if grasp_idx is None:
                en.logger.info(f"LoadGraspPose: action term DOF {name!r} is missing from grasp joint positions")
                return None
            indices.append(grasp_idx)
        return sampled_joint_pos[:, indices]

    def _sync_action_term_to_grasp(
        self,
        envs_idx: slice | torch.Tensor,
        sampled_joint_pos: torch.Tensor,
    ) -> None:
        if self.action_term is None:
            return

        term_dof_names = tuple(getattr(self.action_term, "dofs_name", ()) or ())
        term_grasp_pos = self._sampled_pos_for_dofs(sampled_joint_pos, term_dof_names)
        off_t = getattr(self.action_term, "_offset", None)
        if term_grasp_pos is not None and isinstance(off_t, torch.Tensor) and off_t.dim() == 2:
            if off_t.shape[0] != self.num_envs:
                self.action_term._offset = off_t.repeat(self.num_envs, 1)
                off_t = self.action_term._offset
            off_t[envs_idx] = term_grasp_pos + self._offset_from_configured_action_offset(term_dof_names)

        frozen_hold_pos = getattr(self.action_term, "_frozen_hold_pos", None)
        frozen_dof_names = tuple(getattr(self.action_term, "frozen_dofs", ()) or ())
        frozen_grasp_pos = self._sampled_pos_for_dofs(sampled_joint_pos, frozen_dof_names)
        if frozen_grasp_pos is not None and isinstance(frozen_hold_pos, torch.Tensor) and frozen_hold_pos.dim() == 2:
            frozen_hold_pos[envs_idx] = frozen_grasp_pos

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        """Sample and apply a random grasp from the precomputed set."""
        if self.num_grasps == 0:
            return

        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, prefer_slice=False, return_n_envs=True)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]

        # Sample random grasps from precomputed set
        if self.obj_is_heterogeneous:
            # Sample grasps matching each environment's geometry
            grasp_indices = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            for i, env_idx in enumerate(envs_idx):
                env_geom_idx = self.env_to_geom_idx[env_idx].item()
                grasp_geom_idx = self.env_to_grasp_geom_idx[env_geom_idx]
                available_grasps = self.grasps_per_geom[grasp_geom_idx]
                random_idx = torch.randint(0, len(available_grasps), (1,), device=self.device)
                grasp_indices[i] = available_grasps[random_idx]
        else:
            # Random sampling for homogeneous objects
            grasp_indices = torch.randint(0, self.num_grasps, (n_envs,), device=self.device)

        if self.has_robot_root_pose:
            sampled_robot_pos = self.grasp_robot_pos[grasp_indices]  # (num_resamples, 3)
            sampled_robot_quat = self.grasp_robot_quat[grasp_indices]  # (num_resamples, 4)
            self._env._sampled_grasp_robot_pos[envs_idx] = sampled_robot_pos
            self._env._sampled_grasp_robot_quat[envs_idx] = sampled_robot_quat
            self._env._sampled_grasp_has_robot_root_pose[envs_idx] = True
            self.robot.set_pos(sampled_robot_pos, envs_idx)
            self.robot.set_quat(sampled_robot_quat, envs_idx)
        else:
            self._env._sampled_grasp_robot_pos[envs_idx] = 0.0
            self._env._sampled_grasp_robot_quat[envs_idx] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float
            )
            self._env._sampled_grasp_has_robot_root_pose[envs_idx] = False

        # Set robot joint positions to sampled grasps
        sampled_joint_pos = self.grasp_joint_pos[grasp_indices]  # (num_resamples, num_dofs)
        self._env._sampled_grasp_indices[envs_idx] = grasp_indices
        self._env._sampled_grasp_joint_pos[envs_idx] = sampled_joint_pos
        self.robot.set_dofs_pos(sampled_joint_pos, self.robot.dofs_idx_local, envs_idx)
        self._sync_action_term_to_grasp(envs_idx, sampled_joint_pos)

        if self.has_obj_pose:
            # Set object pose to sampled grasp poses only when the grasp file stores it.
            sampled_obj_pos = self.grasp_obj_pos[grasp_indices]  # (num_resamples, 3)
            sampled_obj_quat = self.grasp_obj_quat[grasp_indices]  # (num_resamples, 4)
            if self.obj_pose_needs_conversion:
                target_local_quat_inv = self._target_local_quat_inv.expand(n_envs, -1)
                target_local_pos = self._target_local_pos.expand(n_envs, -1)
                sampled_obj_quat = gu.transform_quat_by_quat(target_local_quat_inv, sampled_obj_quat)
                sampled_obj_pos = sampled_obj_pos - gu.transform_by_quat(target_local_pos, sampled_obj_quat)

            self._env._sampled_grasp_obj_pos[envs_idx] = sampled_obj_pos
            self._env._sampled_grasp_obj_quat[envs_idx] = sampled_obj_quat
            self._env._sampled_grasp_has_obj_pose[envs_idx] = True
            self.obj.set_pos(sampled_obj_pos, envs_idx)
            self.obj.set_quat(sampled_obj_quat, envs_idx)
        else:
            self._env._sampled_grasp_obj_pos[envs_idx] = 0.0
            self._env._sampled_grasp_obj_quat[envs_idx] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float
            )
            self._env._sampled_grasp_has_obj_pose[envs_idx] = False

        # Reset articulated object joints alongside the sampled root pose so
        # task observations that read object DOF state start from a finite base.
        self._reset_object_dof_state(envs_idx, n_envs)


@EVENT_TERM_REGISTRY.register()
class ResetRobotFromCachedGrasp(EventTerm):
    """Offset the robot away from the cached grasp target after sampling it."""

    entity_name: str = "robot"
    root_pos_offset: tuple[float, float, float] = (0.0, -0.05, 0.08)
    root_rot_euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos_scale: float = 0.35

    def build(self) -> None:
        super().build()
        self.robot = self._env.entities[self.entity_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]
        if isinstance(envs_idx, slice):
            envs_idx = torch.arange(self.num_envs, device=self.device)[envs_idx]

        has_root_pose = getattr(self._env, "_sampled_grasp_has_robot_root_pose", None)
        grasp_indices = getattr(self._env, "_sampled_grasp_indices", None)
        target_joint_pos = getattr(self._env, "_sampled_grasp_joint_pos", None)
        if has_root_pose is None or grasp_indices is None or target_joint_pos is None:
            return
        if not (grasp_indices[envs_idx] >= 0).any():
            return

        valid_mask = grasp_indices[envs_idx] >= 0

        if has_root_pose[envs_idx].any():
            root_envs_idx = (
                envs_idx[valid_mask & has_root_pose[envs_idx]] if isinstance(envs_idx, torch.Tensor) else envs_idx
            )
            target_root_pos = self._env._sampled_grasp_robot_pos[envs_idx][valid_mask & has_root_pose[envs_idx]]
            target_root_quat = self._env._sampled_grasp_robot_quat[envs_idx][valid_mask & has_root_pose[envs_idx]]
            offset = torch.tensor(self.root_pos_offset, device=self.device, dtype=target_root_pos.dtype).unsqueeze(0)
            offset = offset.expand(target_root_pos.shape[0], -1)
            root_pos = target_root_pos + gu.transform_by_quat(offset, target_root_quat)

            rot_offset_euler = torch.tensor(
                self.root_rot_euler_offset, device=self.device, dtype=target_root_pos.dtype
            ).unsqueeze(0)
            rot_offset_euler = rot_offset_euler.expand(target_root_pos.shape[0], -1)
            rot_offset_quat = gu.xyz_to_quat(rot_offset_euler, rpy=True, degrees=True)
            root_quat = gu.transform_quat_by_quat(rot_offset_quat, target_root_quat)

            self.robot.set_pos(root_pos, root_envs_idx)
            self.robot.set_quat(root_quat, root_envs_idx)

        joint_envs_idx = envs_idx[valid_mask] if isinstance(envs_idx, torch.Tensor) else envs_idx
        joint_pos = self._env._sampled_grasp_joint_pos[envs_idx][valid_mask] * self.joint_pos_scale
        self.robot.set_dofs_pos(joint_pos, self.robot.dofs_idx_local, joint_envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeFrictionRatioWithObs(en.events.RandomizeFrictionRatio):
    obs_name: str = "friction"

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)

        friction_ratio = sample_uniform(
            self.friction_range[0],
            self.friction_range[1],
            (
                self._env.num_envs,
                len(self.links_name),
            ),
            device=self._env.device,
        )[envs_idx]

        self.entity.set_friction(1.0)
        self.entity.set_friction_ratio(
            friction_ratio,
            ls_idx_local=self.ls_idx_local,
            envs_idx=envs_idx,
        )

        obs_term = self._env.observation_manager.get_term(self.obs_name)
        obs_term._cached[envs_idx] = friction_ratio


@EVENT_TERM_REGISTRY.register()
class RandomizeMassShiftWithObs(en.events.RandomizeMassShift):
    obs_name: str = "mass"

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        super().compute(envs_idx)

        mass = self.entity.get_mass()
        obs_term = self._env.observation_manager.get_term(self.obs_name)
        if isinstance(mass, float):
            obs_term._cached[envs_idx] = mass
        else:
            mass = torch.as_tensor(mass, device=self.device, dtype=obs_term.dtype)
            if mass.ndim == 1:
                mass = mass.unsqueeze(-1)
            obs_term._cached[envs_idx] = mass[envs_idx]


# ================== TERMINATIONS ==================


@TERMINATION_TERM_REGISTRY.register()
def obj_below_height(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate if object falls below a certain height."""
    obj: Entity = env.entities[entity_name]
    return obj.get_pos()[:, 2] < threshold


@TERMINATION_TERM_REGISTRY.register()
def obj_tilted_past_threshold(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    local_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    world_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_tilt_deg: float = 78.46,
) -> torch.Tensor:
    """Terminate when the object's ``local_up_axis`` (rotated into the world frame
    by the entity's current quaternion) tilts more than ``max_tilt_deg`` degrees
    away from ``world_up_axis``. ``max_tilt_deg = 78.46`` matches the old cosine
    threshold of 0.2.
    """
    obj = env.entities[entity_name]
    quat = obj.get_quat()
    local = torch.as_tensor(local_up_axis, device=env.device, dtype=quat.dtype).view(1, 3).expand(env.num_envs, 3)
    world = torch.as_tensor(world_up_axis, device=env.device, dtype=quat.dtype).view(1, 3)
    world = world / torch.linalg.vector_norm(world, dim=-1, keepdim=True).clamp_min(1e-6)
    up_w = gu.transform_by_quat(local, quat)
    up_w = up_w / torch.linalg.vector_norm(up_w, dim=-1, keepdim=True).clamp_min(1e-6)
    alignment = (up_w * world).sum(dim=-1).clamp(-1.0, 1.0)
    return alignment < math.cos(math.radians(max_tilt_deg))


@TERMINATION_TERM_REGISTRY.register()
def obj_pos_drift_from_grasp(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    max_distance: float = 0.02,
) -> torch.Tensor:
    """Terminate if the object's pos drifts more than ``max_distance`` (m) from the
    per-env grasp pose cached by :class:`LoadGraspPose` on ``env._sampled_grasp_obj_pos``.
    Envs without a cached obj pose (``_sampled_grasp_has_obj_pose`` False) never trigger.
    """
    obj = env.entities[entity_name]
    obj_pos = obj.get_pos()
    target = getattr(env, "_sampled_grasp_obj_pos", None)
    if target is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    dist = torch.linalg.vector_norm(obj_pos - target.to(dtype=obj_pos.dtype), dim=-1)
    triggered = dist > max_distance
    has_pose = getattr(env, "_sampled_grasp_has_obj_pose", None)
    if has_pose is not None:
        triggered = triggered & has_pose
    return triggered
