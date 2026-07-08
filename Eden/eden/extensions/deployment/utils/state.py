"""Robot state/command dataclasses and state-entity wrapper for deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from eden.utils.isaac_math import quat_apply, quat_apply_inverse, quat_mul

if TYPE_CHECKING:
    from eden.entities.base import Entity


@dataclass
class RobotState:
    stamp: float
    base_quat: np.ndarray | None  # wxyz
    base_ang_vel: np.ndarray | None
    base_lin_acc: np.ndarray | None
    dofs_pos: np.ndarray
    dofs_vel: np.ndarray
    dofs_torque: np.ndarray
    full_dofs_pos: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device = "cpu") -> RobotState:
        return RobotState(
            stamp=self.stamp,
            base_quat=torch.from_numpy(self.base_quat).to(device) if self.base_quat is not None else None,
            base_ang_vel=torch.from_numpy(self.base_ang_vel).to(device) if self.base_ang_vel is not None else None,
            base_lin_acc=torch.from_numpy(self.base_lin_acc).to(device) if self.base_lin_acc is not None else None,
            dofs_pos=torch.from_numpy(self.dofs_pos).to(device),
            dofs_vel=torch.from_numpy(self.dofs_vel).to(device),
            dofs_torque=torch.from_numpy(self.dofs_torque).to(device),
            full_dofs_pos=torch.from_numpy(self.full_dofs_pos).to(device) if self.full_dofs_pos is not None else None,
            extra=self.extra,
        )


@dataclass
class RobotCommand:
    dofs_pos: np.ndarray
    dofs_vel: np.ndarray
    dofs_torque: np.ndarray
    dofs_kp: np.ndarray | None = None
    dofs_kd: np.ndarray | None = None


class RobotStateEntity:
    """Entity-compatible wrapper that exposes RobotState through the Entity getter interface.

    Duck-types the subset of Entity getter methods used by observation terms,
    allowing the ObservationManager to compute observations from real robot state
    without any modification to existing terms.

    Static properties (default_dofs_pos, dofs_name, etc.) are copied from the sim
    Entity at construction time.  Mutable state is updated each step via ``update()``.

    Notes
    -----
    - ``base_ang_vel`` in RobotState is assumed to be in **body frame** (IMU convention).
      ``get_ang(frame="world")`` rotates it into world frame, while
      ``get_ang(frame="body")`` returns it directly.
    - ``base_pos`` is initialized from the sim entity's spawn pose and is
      treated as **unknown** until ``RobotState.extra["base_pos"]`` is
      supplied (e.g. from an external estimator or motion capture). While
      base_pos is unknown, ``update()`` does **not** push a synthetic root
      position into the sim entity's FK so observation terms keep seeing the
      training spawn pose instead of an origin-anchored value.
    - ``base_lin_vel`` defaults to zeros and is updated from
      ``RobotState.extra["base_lin_vel"]`` when supplied.
    - A yaw offset can be installed via :meth:`set_yaw_offset` /
      :meth:`capture_yaw_offset`. Subsequent ``update()`` calls subtract that
      yaw from world-frame quantities (``_quat``, ``_pos``, ``_vel``) so the
      deployment world frame matches the training/motion world frame.
      Body-frame quantities (``_ang_vel``) are unaffected. The spawn-pose
      fallback for ``_pos`` is left in the sim's training world frame and is
      not yaw-rotated, since the policy already saw it that way.
    """

    def __init__(self, sim_entity: Entity, device: torch.device) -> None:
        self.device = device
        # Kept as the kinematics oracle: link-level getters (get_links_pos, etc.)
        # delegate here, with the sim entity's qpos synced from real state in update().
        self._sim_entity = sim_entity

        # -- static properties copied from the sim entity --
        self.num_dofs: int = sim_entity.num_dofs
        self.dofs_name: list[str] = list(sim_entity.dofs_name)
        self.dofs_name_map: dict[str, int] = dict(sim_entity.dofs_name_map)
        self.dofs_idx_local = sim_entity.dofs_idx_local.clone().to(device)
        self.dofs_idx_map: dict[int, int] = dict(sim_entity.dofs_idx_map)
        self.default_dofs_pos = sim_entity.default_dofs_pos.clone().to(device)
        self.default_dofs_vel = sim_entity.default_dofs_vel.clone().to(device)
        self.default_dofs_kp = sim_entity.default_dofs_kp.clone().to(device)
        self.default_dofs_kd = sim_entity.default_dofs_kd.clone().to(device)
        self.is_fixed_base: bool = getattr(sim_entity, "is_fixed_base", False)
        self._is_attaching: bool = getattr(sim_entity, "_is_attaching", False)
        self.material = None

        fv = sim_entity.forward_vec
        if isinstance(fv, (tuple, list)):
            self._forward_vec = torch.tensor(fv, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            self._forward_vec = fv.clone().to(device)
            if self._forward_vec.dim() == 1:
                self._forward_vec = self._forward_vec.unsqueeze(0)

        # -- pre-allocated mutable state buffers (single env) --
        # Default ``_pos`` to the sim entity's spawn pose so that, when no
        # external base_pos estimator is wired up, the wrapper and the sim
        # entity agree on the root translation FK is computed at. This keeps
        # body-frame observation terms internally consistent and avoids
        # silently anchoring world-frame link positions at the origin.
        self._pos = self._initial_base_pos(sim_entity, device)
        self._base_pos_known: bool = False
        self._quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        self._vel = torch.zeros(1, 3, dtype=torch.float32, device=device)
        self._ang_vel = torch.zeros(1, 3, dtype=torch.float32, device=device)
        self._dofs_pos = torch.zeros(1, self.num_dofs, dtype=torch.float32, device=device)
        self._dofs_vel = torch.zeros(1, self.num_dofs, dtype=torch.float32, device=device)
        self._dofs_torque = torch.zeros(1, self.num_dofs, dtype=torch.float32, device=device)

        # Yaw offset removed from the IMU world frame so that deployment matches
        # the yaw-aligned world frame the policy was trained in. Identity by default.
        self._yaw_offset: float = 0.0
        self._yaw_remove_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    @staticmethod
    def _initial_base_pos(sim_entity: Entity, device: torch.device) -> torch.Tensor:
        """Read the sim entity's current base position and shape it to ``(1, 3)``.

        Falls back to zeros if the sim entity hasn't been built yet (no
        ``get_pos``) so existing tests / fixed-base configs keep working.
        """
        try:
            pos = sim_entity.get_pos()
        except Exception:
            return torch.zeros(1, 3, dtype=torch.float32, device=device)
        pos = pos.detach().clone().to(dtype=torch.float32, device=device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        return pos[:1]

    @property
    def forward_vec(self) -> torch.Tensor:
        return self._forward_vec

    @property
    def fixed(self) -> bool:
        return False

    @property
    def base_pos_known(self) -> bool:
        """Return True if a real-world base position was supplied since the last reset.

        The base position is supplied via ``RobotState.extra["base_pos"]``.
        """
        return self._base_pos_known

    def update(self, state: RobotState) -> None:
        """Update internal buffers from a real-robot RobotState reading.

        Also syncs the wrapped sim entity's qpos so that Genesis FK
        (consumed by ``get_links_pos`` / ``get_links_quat``) reflects the
        latest real-robot state when observation terms are computed.

        Any installed yaw offset (see :meth:`set_yaw_offset`) is applied to
        world-frame quantities (``_quat``, ``_pos``, ``_vel``) before the sim
        entity is synced, so FK is computed in the yaw-aligned world frame.
        """
        self._quat[0] = torch.as_tensor(state.base_quat, dtype=torch.float32, device=self.device)
        self._ang_vel[0] = torch.as_tensor(state.base_ang_vel, dtype=torch.float32, device=self.device)
        self._dofs_pos[0] = torch.as_tensor(state.dofs_pos, dtype=torch.float32, device=self.device)
        self._dofs_vel[0] = torch.as_tensor(state.dofs_vel, dtype=torch.float32, device=self.device)
        self._dofs_torque[0] = torch.as_tensor(state.dofs_torque, dtype=torch.float32, device=self.device)
        if "base_pos" in state.extra:
            self._pos[0] = torch.as_tensor(state.extra["base_pos"], dtype=torch.float32, device=self.device)
            self._base_pos_known = True
        if "base_lin_vel" in state.extra:
            self._vel[0] = torch.as_tensor(state.extra["base_lin_vel"], dtype=torch.float32, device=self.device)

        if self._yaw_offset != 0.0:
            self._quat[:] = quat_mul(self._yaw_remove_quat, self._quat)
            self._vel[:] = quat_apply(self._yaw_remove_quat, self._vel)
            # Only yaw-rotate _pos when it represents a real IMU/estimator
            # reading. The spawn-pose fallback already lives in the sim's
            # training world frame and would be corrupted by another rotation.
            if self._base_pos_known:
                self._pos[:] = quat_apply(self._yaw_remove_quat, self._pos)

        if not self.is_fixed_base:
            # Skip set_pos when base_pos is unknown so the sim entity's FK
            # keeps using the training spawn pose instead of an origin-anchored
            # synthetic value. set_quat is always synced because base_quat is
            # always available from the IMU.
            if self._base_pos_known:
                self._sim_entity.set_pos(self._pos)
            self._sim_entity.set_quat(self._quat)
        self._sim_entity.set_dofs_pos(self._dofs_pos)
        self._sim_entity.set_dofs_vel(self._dofs_vel)

    # -- yaw alignment ----------------------------------------------------

    @property
    def yaw_offset(self) -> float:
        """Captured world-frame yaw, in radians, that ``update()`` removes."""
        return self._yaw_offset

    def set_yaw_offset(self, yaw: float) -> None:
        """Install a yaw rotation (radians, around +z) that ``update()`` removes from world-frame quantities.

        The resulting world frame has the same z-axis but its x-axis is rotated
        by ``-yaw`` relative to the IMU's. Setting ``yaw=0`` restores the raw
        IMU world frame.
        """
        self._yaw_offset = float(yaw)
        # Build a yaw-only rotation around -z: q = (cos(-yaw/2), 0, 0, sin(-yaw/2)).
        half = -0.5 * float(yaw)
        self._yaw_remove_quat[0, 0] = float(np.cos(half))
        self._yaw_remove_quat[0, 1] = 0.0
        self._yaw_remove_quat[0, 2] = 0.0
        self._yaw_remove_quat[0, 3] = float(np.sin(half))

    def capture_yaw_offset(self, base_quat_wxyz) -> float:
        """Extract yaw from a (w, x, y, z) quaternion and install it as the offset.

        Returns the captured yaw in radians.
        """
        q = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(-1)
        w, x, y, z = q[0], q[1], q[2], q[3]
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        self.set_yaw_offset(yaw)
        return yaw

    # -- base link getters ------------------------------------------------

    def get_pos(self, envs_idx=None) -> torch.Tensor:
        return self._pos

    def get_quat(self, envs_idx=None) -> torch.Tensor:
        return self._quat

    def get_heading(self, envs_idx=None) -> torch.Tensor:
        forward_w = quat_apply(self._quat, self._forward_vec)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world") -> torch.Tensor:
        if frame == "world":
            return self._vel
        elif frame == "body":
            return quat_apply_inverse(self._quat, self._vel)
        raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world") -> torch.Tensor:
        if frame == "body":
            return self._ang_vel
        elif frame == "world":
            return quat_apply(self._quat, self._ang_vel)
        raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    # -- DOF getters ------------------------------------------------------

    def _select_dofs(self, data: torch.Tensor, dofs_idx_local) -> torch.Tensor:
        if dofs_idx_local is None:
            return data
        local_idx = torch.tensor(
            [self.dofs_idx_map[int(i)] for i in dofs_idx_local],
            dtype=torch.long,
            device=self.device,
        )
        return data[:, local_idx]

    def get_dofs_pos(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_pos, dofs_idx_local)

    def get_dofs_vel(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_vel, dofs_idx_local)

    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_torque, dofs_idx_local)

    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._select_dofs(self._dofs_torque, dofs_idx_local)

    # -- link getters (delegate to sim entity, kept in sync via update()) ----

    def find_named_links_idx_local(self, links_name, name_scope=None, preserve_order=True):
        return self._sim_entity.find_named_links_idx_local(
            links_name, name_scope=name_scope, preserve_order=preserve_order
        )

    def get_links_pos(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._sim_entity.get_links_pos(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_quat(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._sim_entity.get_links_quat(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_vel(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._sim_entity.get_links_vel(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_ang(self, ls_idx_local=None, envs_idx=None) -> torch.Tensor:
        return self._sim_entity.get_links_ang(ls_idx_local=ls_idx_local, envs_idx=envs_idx)

    # -- catch-all delegation to the underlying sim entity -------------------
    #
    # Action terms (e.g. DifferentialIKController, OperationalSpaceController)
    # call kinematics/dynamics methods like ``get_link``, ``get_jacobian``,
    # ``get_mass_mat`` and ``control_dofs_pos`` / ``control_dofs_force``.
    # In normal use those run against the cached ``sim_entity`` reference
    # captured in ``ActionTerm.build()`` (before this wrapper replaces the
    # entities-dict entry), and ``update()`` has already synced the sim
    # entity's qpos from the real state so they return correct values.
    #
    # But anything that reaches the wrapper via ``env.entities[name]`` at
    # runtime would otherwise hit AttributeError. Falling through to the
    # sim entity preserves Genesis's behavior; the wrapper-specific
    # overrides above still take precedence because ``__getattr__`` is
    # only consulted when normal lookup fails.
    def __getattr__(self, name: str):
        # ``__getattr__`` is only called when the attribute is missing from
        # the instance and class dict, so this does not shadow any explicit
        # method defined above.
        if name.startswith("_"):
            # Avoid recursion via ``self._sim_entity`` before it's set in __init__.
            raise AttributeError(name)
        try:
            sim_entity = object.__getattribute__(self, "_sim_entity")
        except AttributeError:
            raise AttributeError(name)
        return getattr(sim_entity, name)
