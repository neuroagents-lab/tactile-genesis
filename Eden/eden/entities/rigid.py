"""Rigid (articulated/actuated) entity wrapper with DOF, IK, and contact helpers.

DOF-ordering gotcha: ``set_qpos`` / ``get_dofs_limit`` / ``get_jacobian`` all use
**dofs_name order** (MuJoCo XML declaration order), whereas ``find_named_dofs_idx_local``
returns indices in Genesis kinematic-tree (BFS) order — use the latter for Genesis-level
API calls, **not** for indexing into ``get_jacobian`` columns. ``get_jacobian`` already
reorders its columns Genesis->dofs_name, so no manual reordering is needed.

For actuated entities, ``post_build`` adds ``kd * substep_dt`` to the armature for
implicit-damping numerical stability, so the effective armature differs from the
configured value.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, Literal

import genesis as gs
import genesis.utils.geom as gu
import torch
from genesis.typing import Vec2FType

import eden as en
from eden.entities._grouped_contacts import (
    aggregate_grouped_contacts as _aggregate_grouped_contacts,
    resolve_local_link_idx as _resolve_local_link_idx,
)
from eden.entities.base import Entity
from eden.entities.support_mixin import SupportSurfaceMixin
from eden.options.actuators import ActuatorSpecOptions
from eden.utils.misc import sanitize_envs_idx
from eden.utils.string import resolve_matching_names, resolve_matching_names_values


class RigidEntity(Entity, SupportSurfaceMixin):
    """Entity subclass for rigid and kinematic materials.

    Provides DOF/link/geom access, FK/IK, Jacobians, contacts,
    and support-surface utilities on top of the base :class:`Entity`.
    """

    dofs_name: list[str] = []
    default_dofs_pos: dict[str, float] = {}
    default_dofs_vel: dict[str, float] = {}
    default_dofs_stiffness: dict[str, float] = {}
    default_dofs_damping: dict[str, float] = {}
    default_dofs_armature: dict[str, float] = {}
    default_dofs_kp: dict[str, float] = {}
    default_dofs_kd: dict[str, float] = {}
    default_dofs_pos_limits: dict[str, Vec2FType] = {}
    soft_dofs_pos_limits: dict[str, Vec2FType] = {}
    default_dofs_force_limits: dict[str, float] = {}

    is_articulated: bool = False
    is_actuated: bool = False
    is_support_enabled: bool = False

    support_sample_method: Literal["uniform", "grid"] = "uniform"
    support_minimum_clearance: float = 0.05
    support_shrink: float = 0.1
    support_num_sample_points: int = 300
    support_grid_size: float = 0.01
    support_links_name: list[str] = []

    @property
    def fixed(self) -> bool:
        return self.is_fixed_base and not self.is_articulated

    @property
    def dofs_spec(self):
        return self._options.dofs_spec

    # ------------------------------------------------------------------------------------
    # ----------------------------------- post_build -------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def _parse_dofs_info(
        self,
        dofs_info: dict[str, float | Vec2FType],
        default_info: torch.Tensor,
    ) -> torch.Tensor:
        info = default_info.clone()
        idx, _, dofs_values = resolve_matching_names_values(dofs_info, self.dofs_name, preserve_order=True)

        for i, val in zip(idx, dofs_values):
            info[i] = val

        return info.to(self.device)

    @gs.assert_built
    def _init_dofs_property(
        self,
        attr_name: str,
        setter_name: str | None,
        default_info: torch.Tensor,
        *,
        apply_to_sim: bool = True,
        force_broadcast: bool = False,
    ) -> torch.Tensor:
        """Initialize a DOF property: parse user dict, store as tensor, optionally apply to sim.

        Works for both scalar-per-DOF (shape ``[1, num_dofs]``) and range-per-DOF
        (shape ``[1, num_dofs, 2]``) properties — ``_parse_dofs_info`` infers the shape
        from the value type (float vs tuple).

        Parameters
        ----------
        attr_name : str
            Attribute name on self, e.g. ``"default_dofs_stiffness"``.
        setter_name : str | None
            Method name to write values to sim, or None to skip.
        default_info : torch.Tensor
            Used as per-key fallback inside ``_parse_dofs_info`` when the user dict
            is set, and as the full silent fallback when the user dict is unset.
        apply_to_sim : bool
            Whether to call the setter after parsing.
        force_broadcast : bool
            Always repeat for all envs, bypassing the ``batch_dofs_info`` check.
            Use for properties whose setter always expects a full ``(num_envs, ...)`` tensor.
        """
        user_dict = getattr(self, attr_name)
        if user_dict:
            parsed = self._parse_dofs_info(user_dict, default_info=default_info).unsqueeze(0)

            # Warn about silent control-gain fallthrough for kp/kd: a partial
            # override dict replaces the robot's default dict entirely, and
            # unmatched DOFs fall back to the sim's current gain. For robots
            # whose XML has no <actuator> block (e.g. YAM), that fallback is 0,
            # leaving the DOF uncontrolled — commands to it silently do nothing.
            if attr_name in ("default_dofs_kp", "default_dofs_kd") and parsed.ndim == 2:
                matched_idx, _, _ = resolve_matching_names_values(user_dict, self.dofs_name, preserve_order=True)
                matched_set = set(matched_idx)
                unmatched_zero = [
                    self.dofs_name[i]
                    for i in range(len(self.dofs_name))
                    if i not in matched_set and float(parsed[0, i].item()) == 0.0
                ]
                if unmatched_zero:
                    gain_kind = "kp" if attr_name.endswith("kp") else "kd"
                    en.logger.warning(
                        f"`{attr_name}` resolves to {gain_kind}=0 for DOFs {unmatched_zero} "
                        f"(not covered by the override dict and no sim-level default set). "
                        f"Position commands to these DOFs will produce no torque. "
                        f"If this is intentional (passive/free-spinning DOFs), you can ignore "
                        f"this message; otherwise add entries for these DOFs to `{attr_name}`."
                    )

            if force_broadcast:
                broadcast = parsed.repeat(self._env.num_envs, *([1] * (parsed.ndim - 1)))
                if apply_to_sim and setter_name is not None:
                    getattr(self, setter_name)(broadcast)
                return broadcast
            if apply_to_sim and setter_name is not None:
                getattr(self, setter_name)(parsed[0])
            return parsed
        else:
            fallback = default_info.unsqueeze(0)
            return fallback

    def pre_build(self) -> None:
        super().pre_build()
        self._apply_collision_link_mask()

    def _apply_collision_link_mask(self) -> None:
        """Restrict collision to geoms on links matching ``collision_link_patterns``.

        Runs after the entity is added to the scene but before ``scene.build()``,
        because Genesis bakes the collision-pair candidate list from per-geom
        ``contype``/``conaffinity`` at build time. Geoms on matching links keep
        collision (``contype = conaffinity = 1``, optionally with
        ``collision_friction``); every other link's geoms are disabled
        (``contype = conaffinity = 0``).

        Genesis defaults ``is_local_collision_mask=True`` for MJCF entities, which
        makes ``contype``/``conaffinity`` a *local self-collision* mask only — so
        zeroing them would NOT stop collisions with other entities (e.g. a grasped
        cube). We set it ``False`` so the mask filters cross-entity pairs (see
        ``collider.py``: ``con_skip = (same_entity | ~has_local_mask) & (con_match == 0)``).
        """
        patterns = getattr(self._options, "collision_link_patterns", None)
        if not patterns:
            return
        geoms = self._entity.geoms
        unique_links = sorted({g.link.name for g in geoms})
        _, matched = resolve_matching_names(patterns, unique_links, preserve_order=True)
        keep_links = set(matched)
        if not keep_links:
            en.logger.warning(
                f"collision_link_patterns {patterns} matched no links on '{self.name}' "
                f"(links: {unique_links}); leaving collision unchanged."
            )
            return
        # Treat contype/conaffinity as a global (cross-entity) filter, not a local one.
        self._entity._is_local_collision_mask = False
        fric = self._options.collision_friction
        for g in geoms:
            if g.link.name in keep_links:
                g._contype = 1
                g._conaffinity = 1
                if fric is not None:
                    g.set_friction(fric)
            else:
                g._contype = 0
                g._conaffinity = 0
        en.logger.info(
            f"[collision mask] '{self.name}': only links {sorted(keep_links)} collide "
            f"({len(keep_links)}/{len(unique_links)} links)."
        )

    @gs.assert_built
    def post_build(self) -> None:
        is_kinematic = isinstance(self.material, en.materials.KinematicMaterialOptions)
        if self.is_articulated or self.is_actuated:
            self.dofs_name, dofs_idx_local = self.find_named_dofs_idx_local(self.dofs_name, preserve_order=True)
            self.dofs_name_map = {name: i for i, name in enumerate(self.dofs_name)}
            self.dofs_idx_map = {idx: i for i, idx in enumerate(dofs_idx_local)}
            self.dofs_idx_local = torch.as_tensor(dofs_idx_local, dtype=gs.tc_int, device=self.device).contiguous()

            # -- Jacobian column reorder: Genesis kinematic-tree → dofs_name --
            # get_jacobian() returns columns in Genesis BFS order, but all other
            # DOF APIs (set_qpos, get_dofs_limit, …) use dofs_name order.  Build
            # a permutation so get_jacobian() returns columns in dofs_name order.
            if self.is_fixed_base:
                self._jac_perm = self.dofs_idx_local.clone()
            else:
                self._jac_perm = torch.cat(
                    [
                        torch.arange(6, dtype=gs.tc_int, device=self.device),
                        self.dofs_idx_local,
                    ]
                )

            dofs_pos_init = self.get_dofs_pos(envs_idx=0)
            self.default_dofs_pos = self._init_dofs_property(
                "default_dofs_pos",
                "set_dofs_pos",
                default_info=dofs_pos_init.squeeze(0),
                force_broadcast=True,
            )
            assert self.default_dofs_pos.ndim == 2, f"default_dofs_pos.shape: {self.default_dofs_pos.shape}"

            # --- pos limits (always read, no setter) ---
            _envs_idx_batch = 0 if self._env.env_options.batch_dofs_info and not is_kinematic else None
            default_pos_limits = torch.stack(self.get_dofs_limit(envs_idx=_envs_idx_batch), dim=-1)
            self.default_dofs_pos_limits = self._init_dofs_property(
                "default_dofs_pos_limits",
                None,
                default_info=default_pos_limits.squeeze(0),
            )
            assert self.default_dofs_pos_limits.ndim == 3, (
                f"default_dofs_pos_limits.shape: {self.default_dofs_pos_limits.shape}"
            )

            if not is_kinematic:
                # --- dofs_vel: zero default, always broadcast to all envs ---
                self.default_dofs_vel = self._init_dofs_property(
                    "default_dofs_vel",
                    "set_dofs_vel",
                    default_info=torch.zeros(self.num_dofs),
                    force_broadcast=True,
                )

                # --- batch-aware DOF properties ---
                spec_stiffness = {key: spec.STIFFNESS for key, spec in self.dofs_spec.items()}
                self.default_dofs_stiffness = self._init_dofs_property(
                    "default_dofs_stiffness",
                    "set_dofs_stiffness",
                    default_info=self._parse_dofs_info(
                        spec_stiffness,
                        default_info=self.get_dofs_stiffness(
                            envs_idx=0 if self._env.env_options.batch_dofs_info else None
                        ).squeeze(0),
                    ),
                    force_broadcast=self._env.env_options.batch_dofs_info,
                )
                spec_damping = {key: spec.DAMPING for key, spec in self.dofs_spec.items()}
                self.default_dofs_damping = self._init_dofs_property(
                    "default_dofs_damping",
                    "set_dofs_damping",
                    default_info=self._parse_dofs_info(
                        spec_damping,
                        default_info=self.get_dofs_damping(
                            envs_idx=0 if self._env.env_options.batch_dofs_info else None
                        ).squeeze(0),
                    ),
                    force_broadcast=self._env.env_options.batch_dofs_info,
                )
                spec_armature = {key: spec.ARMATURE() for key, spec in self.dofs_spec.items()}
                self.default_dofs_armature = self._init_dofs_property(
                    "default_dofs_armature",
                    "set_dofs_armature",
                    default_info=self._parse_dofs_info(
                        spec_armature,
                        default_info=self.get_dofs_armature(
                            envs_idx=0 if self._env.env_options.batch_dofs_info else None
                        ).squeeze(0),
                    ),
                    force_broadcast=self._env.env_options.batch_dofs_info,
                )
                # mjlab/IsaacLab parity: shrink the hard joint range to a soft
                # range by soft_joint_pos_limit_factor=0.9 about the midpoint,
                # rather than using the hard limits directly. This is the
                # threshold the dofs_pos_limits penalty is computed against.
                _hard_limits = self.default_dofs_pos_limits.clone()[0]
                _mid = 0.5 * (_hard_limits[:, 0] + _hard_limits[:, 1])
                _half_range = 0.5 * (_hard_limits[:, 1] - _hard_limits[:, 0])
                _soft_factor = 0.9
                _soft_limits = torch.stack(
                    [_mid - _soft_factor * _half_range, _mid + _soft_factor * _half_range], dim=-1
                )
                self.soft_dofs_pos_limits = self._init_dofs_property(
                    "soft_dofs_pos_limits", None, default_info=_soft_limits
                )

                _SPEC_PROPS = [
                    "FULL_TORQUE_SPEED",
                    "NO_LOAD_SPEED",
                    "DRIVING_TORQUE_LIMIT",
                    "BRAKING_TORQUE_LIMIT",
                    "STATIC_FRICTION",
                    "DYNAMIC_FRICTION",
                    "FRICTION_ACTIVATION_SPEED",
                ]

                for key in _SPEC_PROPS:
                    parsed = self._parse_dofs_info(
                        {dofs_key: getattr(spec, key) for dofs_key, spec in self.dofs_spec.items()},
                        default_info=torch.zeros(self.num_dofs) + getattr(ActuatorSpecOptions, key),
                    ).unsqueeze(0)
                    setattr(self, f"_{key.lower()}", parsed)

                # --- force limits ---
                spec_effort_limit = {key: spec.EFFORT_LIMIT for key, spec in self.dofs_spec.items()}
                self.default_dofs_force_limits = self._init_dofs_property(
                    "default_dofs_force_limits",
                    None,
                    default_info=self._parse_dofs_info(
                        spec_effort_limit,
                        default_info=self.get_dofs_force_range(
                            envs_idx=0 if self._env.env_options.batch_dofs_info else None
                        )[1].squeeze(0),
                    ),
                    apply_to_sim=False,
                )
                if self._env.env_options.batch_dofs_info:
                    upper_limits = self.default_dofs_force_limits.repeat(self._env.num_envs, 1)
                else:
                    upper_limits = self.default_dofs_force_limits[0]
                self.set_dofs_force_range(-upper_limits, upper_limits)
        else:
            self.dofs_name_map = {}
            self.dofs_idx_map = {}
            self.dofs_idx_local = None
            self._jac_perm = None
            self.default_dofs_pos = None
            self.default_dofs_vel = None
            self.default_dofs_stiffness = None
            self.default_dofs_damping = None
            self.default_dofs_armature = None
            self.default_dofs_kp = None
            self.default_dofs_kd = None
            self.default_dofs_pos_limits = None
            self.soft_dofs_pos_limits = None
            self.default_dofs_force_limits = None
            # actuator spec properties
            self._full_torque_speed = None
            self._no_load_speed = None
            self._driving_torque_limit = None
            self._braking_torque_limit = None
            self._static_friction = None
            self._dynamic_friction = None
            self._friction_activation_speed = None

        if self.is_fixed_base:
            self.qs_idx_local = self.dofs_idx_local
        else:
            if self.dofs_idx_local is not None:
                self.qs_idx_local = (
                    torch.cat(
                        [torch.arange(7), self.dofs_idx_local + 1]
                    )  # why + 1: dofs_idx_local has 6 base free dofs -> to add 7 offset, we just add 1
                    .to(self.device)
                    .contiguous()
                )
            else:
                self.qs_idx_local = torch.arange(7).to(self.device)

        if not is_kinematic and self.is_actuated:
            # NOTE: kp, kd is for control (vs. stiffness/damping for simulation)
            def _read_dof_gains(getter, attr_name: str) -> torch.Tensor:
                """Read per-DOF kp/kd from the solver, falling back to zeros on rejection.

                The solver rejects the read when, e.g., MJCF tendon-approximated
                actuators are not PD-reducible.

                The returned tensor is the base for ``_parse_dofs_info``; any DOFs
                listed in the user's ``default_dofs_{kp,kd}`` dict overwrite the
                zeros in place. A zero gain on an unspecified DOF matches the
                passive-follower semantics of tendon-driven joints.
                """
                envs_idx = 0 if self._env.env_options.batch_dofs_info else None
                try:
                    return getter(envs_idx=envs_idx).squeeze(0)
                except gs.GenesisException as e:
                    if "non-PD-reducible" not in str(e):
                        raise
                    en.logger.debug(
                        f"Entity '{self.name}': non-PD-reducible actuator detected; "
                        f"using zero fallback for `{attr_name}` where not explicitly set."
                    )
                    return torch.zeros(len(self.dofs_name), device=self.device)

            self.default_dofs_kp = self._init_dofs_property(
                "default_dofs_kp",
                "set_dofs_kp",
                default_info=_read_dof_gains(self.get_dofs_kp, "default_dofs_kp"),
                force_broadcast=self._env.env_options.batch_dofs_info,
            )
            self.default_dofs_kd = self._init_dofs_property(
                "default_dofs_kd",
                "set_dofs_kd",
                default_info=_read_dof_gains(self.get_dofs_kd, "default_dofs_kd"),
                force_broadcast=self._env.env_options.batch_dofs_info,
            )

            # NOTE: Must update DoF armature to emulate implicit damping for force control.
            # When default_dofs_armature is not given, it will be set to 0.1 by default.
            # This is equivalent to a first-order correction term, which greatly improves numerical stability.
            implicit_damping = self.default_dofs_kd * self._env.scene.sim._substep_dt
            effective_armature = self.default_dofs_armature + implicit_damping
            assert effective_armature.ndim == 2
            self.set_dofs_armature(effective_armature.squeeze(0))
        elif not is_kinematic and self._entity.n_dofs > 0 and not self.is_fixed_base:
            # NOTE: apply damping to stabilize the free joints for non-actuated entities
            _, free_dofs_idx = self.find_named_dofs_idx_local(
                dofs_name="(?:.*_(?:baselink|base)_joint|/?root_joint|/?mesh_joint)"
            )
            if len(free_dofs_idx) > 0:
                self.set_dofs_damping(
                    self.get_dofs_damping(dofs_idx_local=free_dofs_idx) + 0.0002,
                    dofs_idx_local=free_dofs_idx,
                )

        # Set initial root pose
        if not self._is_attaching:
            default_root_pos, default_root_quat = self.get_default_root_pose()
            needs_coup = getattr(self.material, "needs_coup", True)
            coup_type = getattr(self.material, "coup_type", None)
            is_ipc_coupled = self._env._has_ipc_coupler and needs_coup

            if is_ipc_coupled and (coup_type == "external_articulation" or self.is_fixed_base):
                # IPC cases where Genesis rejects direct root-pose writes:
                # - external_articulation (base driven by IPC constraint system)
                # - any IPC coup_type on fixed-base (Genesis set_pos/set_quat guard)
                # Root pose is baked into morph.pos at morph creation.
                pass
            elif is_ipc_coupled and not self.is_fixed_base:
                # Floating-base with IPC coupling (two_way_soft_constraint / ipc_only).
                # Genesis rejects set_pos/set_quat for these, but set_qpos on the
                # free-root prefix (first 7 qs: xyz + wxyz) is allowed.
                self.set_qpos(
                    torch.cat([default_root_pos, default_root_quat], dim=-1),  # (B, 7)
                    qs_idx_local=torch.arange(7, device=self.device),
                )
            else:
                # Non-IPC (fixed or floating base): preserve the original
                # set_pos + set_quat behavior.
                self.set_pos(default_root_pos)
                self.set_quat(default_root_quat)

        if not is_kinematic and self.is_support_enabled:
            self._prepare_support(
                minimum_clearance=self.support_minimum_clearance,
                pre_sample_points=True,
                pre_num_sample_points=self.support_num_sample_points,
                pre_shrink=self.support_shrink,
                pre_sample_method=self.support_sample_method,
                pre_grid_size=self.support_grid_size,
                support_links_name=self.support_links_name,
            )

    def get_default_root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get default root pose for all environments.

        For heterogeneous entities, the default root pose is composed with the GroupedEntityOptions default root pose
        and each variant's default pose as an additional offset.
        """
        default_pos = self.default_root_pos.repeat(self._env.num_envs, 1)
        default_quat = self.default_root_quat.repeat(self._env.num_envs, 1)

        if not self.is_heterogeneous:
            return default_pos, default_quat

        if len(self.links) == 0:
            return default_pos, default_quat

        geoms = self.links[0].geoms
        max_idx = min(len(geoms), len(self._grouped_default_root_pos), len(self._grouped_default_root_quat))
        for i in range(max_idx):
            geom = geoms[i]
            active_envs_idx = getattr(geom, "active_envs_idx", None)
            if active_envs_idx is None or len(active_envs_idx) == 0:
                continue

            envs_idx = torch.tensor(active_envs_idx, dtype=gs.tc_int, device=self.device)
            n = int(envs_idx.shape[0])
            group_pos = default_pos[envs_idx]
            group_quat = default_quat[envs_idx]
            sub_pos = self._grouped_default_root_pos[i].unsqueeze(0).expand(n, -1)
            sub_quat = self._grouped_default_root_quat[i].unsqueeze(0).expand(n, -1)

            default_quat[envs_idx] = gu.transform_quat_by_quat(sub_quat, group_quat)
            default_pos[envs_idx] = group_pos + gu.transform_by_quat(sub_pos, group_quat)

        return default_pos, default_quat

    # ------------------------------------------------------------------------------------
    # ------------------------------------ attach ----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_unbuilt
    def attach_to(self, entity: Entity, link_name: str) -> None:
        """Attach this entity to another entity.

        Parameters
        ----------
        entity: Entity
            The entity to attach to.
        link_name: str
            The name of the link to attach this entity to.
        """

        def _do_attach():
            child = self._entity
            # MJCF entities may have multiple kinematic trees (e.g. a world link
            # in one tree and the actual robot base in another).  Genesis's
            # ``attach`` assumes every link shares the same ``root_idx`` as
            # ``links[0]``.  Merge secondary trees into the first so the
            # assertion holds.
            if len(child.links) > 1:
                base_root = child.links[0].root_idx
                roots_to_merge = set()
                for link in child.links[1:]:
                    if link.root_idx != base_root and link.root_idx != -1:
                        roots_to_merge.add(link.root_idx)
                if roots_to_merge:
                    for link in child.links:
                        if link.root_idx in roots_to_merge:
                            link._root_idx = base_root
            child.attach(entity._entity, link_name)

        self._env.register_pre_build_hook(_do_attach)
        self._is_attaching = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties ------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def dofs_order(self) -> dict[str, int]:
        """Get the mapping from dofs name to its local index."""
        if self.is_built:
            return {dof_name: i for i, dof_name in enumerate(self.dofs_name)}
        raise RuntimeError("Entity is not built yet")

    @property
    def joints_name(self) -> list[str]:
        """Includes `root_joint` besides the dofs_name."""
        return [joint.name for joint in self._entity.joints]

    @property
    def num_dofs(self):
        """Return the number of controllable DOFs specified by ``dofs_name``.

        This may differ from the total number of DOFs of the entity; uncontrollable
        DOFs such as free base joints are not included.
        """
        if self.is_built:
            return len(self.dofs_name)
        raise RuntimeError("Entity is not built yet")

    @property
    def n_dofs(self):
        return self._entity.n_dofs

    @property
    def n_qs(self):
        return self._entity.n_qs

    @property
    def q_limit(self):
        return self._entity.q_limit

    @property
    def num_links(self):
        return self._entity.n_links

    @property
    def links(self):
        return self._entity.links

    @property
    def link_start(self):
        return self._entity.link_start

    @property
    def link_end(self):
        return self._entity.link_end

    @property
    def geoms(self):
        return self._entity.geoms

    @property
    def geom_start(self):
        return self._entity.geom_start

    @property
    def geom_end(self):
        return self._entity.geom_end

    @property
    def dof_start(self):
        return self._entity.dof_start

    @property
    def dof_end(self):
        return self._entity.dof_end

    # ------------------------------------------------------------------------------------
    # ------------------------------- find named indices ---------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def find_named_dofs_idx_local(
        self,
        dofs_name: list[str] | str,
        name_scope: list[str] | None = None,
        preserve_order: bool = True,
    ) -> tuple[list[str], list[int]]:
        """Find DOF indices by DOF/joint names or patterns.

        Supports:
        - Exact names: "joint_1"
        - Glob patterns: "joint_*", "arm?[LR]_*" (fnmatch semantics)
        - Regex patterns: any valid regex (uses fullmatch). If a pattern isn't an exact
          match or glob, we attempt to treat it as a regex.

        Parameters
        ----------
        dofs_name: list[str] | str
            The names of the DOFs to find.
        name_scope: list[str] | None
            The names of the joints to search within. If None, all joints will be searched.
        preserve_order: bool
            Whether to preserve the order of the DOFs. If True, the DOFs will be returned in the same order as they are in the dofs_name.
        """
        joint_names = name_scope or [joint.name for joint in self._entity.joints]

        _, names = resolve_matching_names(dofs_name, joint_names, preserve_order=preserve_order)

        res_dofs_name: list[str] = []
        res_dofs_idx: list[int] = []
        for name in names:
            joint = self._entity.get_joint(name)
            res_dofs_name.extend([name] * len(joint.dofs_idx_local))
            res_dofs_idx.extend(joint.dofs_idx_local)

        return res_dofs_name, res_dofs_idx

    @gs.assert_built
    def find_named_links_idx_local(
        self,
        links_name: list[str] | str,
        name_scope: list[str] | None = None,
        preserve_order: bool = True,
    ) -> tuple[list[str], list[int]]:
        """Find link indices by link names or patterns.

        Supports:
        - Exact names: "link_1"
        - Glob patterns: "link_*", "arm?[LR]_*" (fnmatch semantics)
        - Regex patterns: any valid regex (uses fullmatch). If a pattern isn't an exact
          match or glob, we attempt to treat it as a regex.

        Parameters
        ----------
        links_name: list[str] | str
            The names of the links to find.
        name_scope: list[str] | None
            The names of the links to search within. If None, all links will be searched.
        preserve_order: bool
            Whether to preserve the order of the links. If True, the links will be returned in the same order as they are in the links_name.
        """
        link_names = name_scope or [link.name for link in self._entity.links]
        _, names = resolve_matching_names(links_name, link_names, preserve_order=preserve_order)

        res_links_name: list[str] = []
        res_links_idx: list[int] = []
        for name in names:
            link = self._entity.get_link(name)
            res_links_name.extend([name])
            res_links_idx.extend([link.idx_local])

        return res_links_name, res_links_idx

    # ------------------------------------------------------------------------------------
    # --------------------------------- FK / IK / Jacobian ------------------------------
    # ------------------------------------------------------------------------------------

    def forward_kinematics(self, qpos, qs_idx_local=None, links_idx_local=None, envs_idx=None):
        """Compute forward kinematics for a single link or multiple links."""
        return self._entity.forward_kinematics(qpos, qs_idx_local, links_idx_local, envs_idx)

    def inverse_kinematics(
        self,
        links_name,
        poss=None,
        quats=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for a single link or multiple links.

        Parameters
        ----------
        links_name: str | list[str]
            The name of the link to compute inverse kinematics for.
        poss: array_like | list[array_like], optional
            The position of the link to compute inverse kinematics for. If empty, position error will not be considered. Defaults to None.
        quats: array_like | list[array_like], optional
            The orientation of the link to compute inverse kinematics for. If empty, orientation error will not be considered. Defaults to None.
        init_qpos: array_like, optional
            Initial joint positions used to seed the solver. Defaults to None.
        respect_joint_limit: bool, optional
            Whether to clamp the solution to the joint limits. Defaults to True.
        max_samples: int, optional
            Maximum number of random restarts attempted by the solver. Defaults to 50.
        max_solver_iters: int, optional
            Maximum number of iterations per sample. Defaults to 20.
        damping: float, optional
            Damping factor for the damped least-squares update. Defaults to 0.01.
        pos_tol: float, optional
            Position tolerance for convergence, in meters. Defaults to 5e-4.
        rot_tol: float, optional
            Rotation tolerance for convergence, in radians. Defaults to 5e-3.
        pos_mask: list[bool], optional
            Per-axis ``(x, y, z)`` mask selecting which position components to match. Defaults to ``[True, True, True]``.
        rot_mask: list[bool], optional
            Per-axis ``(x, y, z)`` mask selecting which rotation components to match. Defaults to ``[True, True, True]``.
        max_step_size: float, optional
            Maximum joint update applied per iteration. Defaults to 0.5.
        dofs_idx_local: array_like, optional
            Local indices of the DOFs the solver is allowed to move. If None, all DOFs are used. Defaults to None.
        return_error: bool, optional
            If True, also return the final residual error alongside the solution. Defaults to False.
        envs_idx: array_like, optional
            The indices of the environments to solve for. If None, all environments are used. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        """
        if isinstance(links_name, str):
            return self._entity.inverse_kinematics(
                link=self.get_link(links_name),
                pos=poss,
                quat=quats,
                init_qpos=init_qpos,
                respect_joint_limit=respect_joint_limit,
                max_samples=max_samples,
                max_solver_iters=max_solver_iters,
                damping=damping,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                pos_mask=pos_mask,
                rot_mask=rot_mask,
                max_step_size=max_step_size,
                dofs_idx_local=dofs_idx_local,
                return_error=return_error,
                envs_idx=envs_idx,
            )
        else:
            return self._entity.inverse_kinematics_multilink(
                links=[self.get_link(name) for name in links_name],
                poss=poss,
                quats=quats,
                init_qpos=init_qpos,
                respect_joint_limit=respect_joint_limit,
                max_samples=max_samples,
                max_solver_iters=max_solver_iters,
                damping=damping,
                pos_tol=pos_tol,
                rot_tol=rot_tol,
                pos_mask=pos_mask,
                rot_mask=rot_mask,
                max_step_size=max_step_size,
                dofs_idx_local=dofs_idx_local,
                return_error=return_error,
                envs_idx=envs_idx,
            )

    def get_jacobian(self, link, local_point=None):
        """
        Get the spatial Jacobian for a point on a target link.

        Columns are returned in ``dofs_name`` order (matching ``set_qpos`` /
        ``get_dofs_limit``), **not** the Genesis kinematic-tree order.

        Parameters
        ----------
        link : RigidLink
            The target link.
        local_point : torch.Tensor or None, shape (3,)
            Coordinates of the point in the link's *local* frame.
            If None, the link origin is used (back-compat).

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        J = self._entity.get_jacobian(link, local_point)
        if self._jac_perm is not None:
            J = J[..., self._jac_perm]
        return J

    # ------------------------------------------------------------------------------------
    # ------------------------------------ getters --------------------------------------
    # ------------------------------------------------------------------------------------

    def get_link(self, link_name: str):
        return self._entity.get_link(link_name)

    def get_AABB(self):
        return self._entity.get_AABB()

    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        if qs_idx_local is None:
            qs_idx_local = self.qs_idx_local
        return self._entity.get_qpos(qs_idx_local, envs_idx=envs_idx)

    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_control_force(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx)

    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_force(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx)

    def get_dofs_vel(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' velocity.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        velocity : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' velocity.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_velocity(dofs_idx_local, envs_idx)

    def get_dofs_pos(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' position.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        position : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' position.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_position(dofs_idx_local, envs_idx)

    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_kp(dofs_idx_local, envs_idx)

    def get_dofs_kd(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_kv(dofs_idx_local, envs_idx)

    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the force range for the entity's dofs.
        """
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_force_range(dofs_idx_local, envs_idx)

    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_limit(dofs_idx_local, envs_idx)

    def get_dofs_stiffness(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_stiffness(dofs_idx_local, envs_idx)

    def get_dofs_invweight(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_invweight(dofs_idx_local, envs_idx)

    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_armature(dofs_idx_local, envs_idx)

    def get_dofs_damping(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_damping(dofs_idx_local, envs_idx)

    def get_dofs_frictionloss(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        return self._entity.get_dofs_frictionloss(dofs_idx_local, envs_idx)

    def get_mass_mat(self, envs_idx=None, decompose=False):
        return self._entity.get_mass_mat(envs_idx, decompose)

    def get_links_pos(self, ls_idx_local=None, envs_idx=None):
        return self._entity.get_links_pos(links_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_quat(self, ls_idx_local=None, envs_idx=None):
        return self._entity.get_links_quat(links_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_vel(self, ls_idx_local=None, envs_idx=None, *, ref="link_origin"):
        return self._entity.get_links_vel(links_idx_local=ls_idx_local, envs_idx=envs_idx, ref=ref)

    def get_links_ang(self, ls_idx_local=None, envs_idx=None):
        return self._entity.get_links_ang(links_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_acc(self, ls_idx_local=None, envs_idx=None):
        return self._entity.get_links_acc(links_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_acc_ang(self, ls_idx_local=None, envs_idx=None):
        return self._entity.get_links_acc_ang(links_idx_local=ls_idx_local, envs_idx=envs_idx)

    def get_links_net_contact_force(self):
        return self._entity.get_links_net_contact_force()

    def get_contacts(self, with_entity: Entity | None = None, exclude_self_contact: bool = False):
        """Return contact information computed during the most recent ``scene.step()``.

        If ``with_entity`` is provided, only returns contact information involving the caller and the specified entity.
        Otherwise, returns all contact information involving the caller entity.
        When ``with_entity`` is ``self``, it will return the self-collision only.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : A boolean mask indicating whether the contact information is valid.
                        (Only when scene is parallelized)

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs
                               and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : Entity, optional
            The entity to check contact with. Defaults to None.
        exclude_self_contact: bool
            Exclude the self collision from the returning contacts. Defaults to False.

        Returns
        -------
        contact_info : dict
            The contact information.
        """
        if isinstance(with_entity, Entity):
            with_entity = with_entity._entity
        return self._entity.get_contacts(with_entity=with_entity, exclude_self_contact=exclude_self_contact)

    def get_grouped_contacts(
        self,
        other_entity: "RigidEntity",
        *,
        links_a_idx_local: torch.Tensor | list[int] | None = None,
        links_b_idx_local: torch.Tensor | list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Force-weighted contact aggregation per (self link, other link) pair.

        Layered on top of :meth:`get_contacts`: per (link_a, link_b) pair,
        compute the contact-force-norm-weighted mean of contact positions
        across every contact between *this* entity (``self``) and
        ``other_entity`` in the most recent ``scene.step()``.

        The output canonicalises orientation: axis 1 always indexes
        ``self``'s links and axis 2 always indexes ``other_entity``'s
        links, regardless of which side Genesis happened to label "a" or
        "b" for any given contact.

        Port of ``dexmachina/envs/contacts.get_grouped_contact_pos``.

        Parameters
        ----------
        other_entity:
            The contact partner. Must be a different entity (``other is not
            self``); self-contact is not supported by this layer — use
            :meth:`get_contacts` directly with ``with_entity=self``.
        links_a_idx_local:
            Local link indices on ``self`` to include on axis 1. Defaults
            to all of ``self``'s links (``[0, num_links)``). For per-step
            consumers, pre-resolve once at ``build()`` time and pass a
            cached ``torch.long`` tensor on ``self.device`` to avoid a
            tiny H2D copy each call (no-op early-return path).
        links_b_idx_local:
            Local link indices on ``other_entity`` to include on axis 2.
            Defaults to all of ``other_entity``'s links. Same hot-path
            note as above.

        Returns
        -------
        dict with:

        - ``position`` : ``(num_envs, n_a, n_b, 3)`` — contact-force-norm
          weighted mean position in world frame; zero where ``valid`` is
          False.
        - ``valid`` : ``(num_envs, n_a, n_b)`` bool — True iff the pair's
          aggregate force-norm is strictly positive (``force_norm_sum >
          0``), matching the DexMachina reference. Pairs whose contacts
          all carry zero force — a quasi-static initial overlap or the
          frame of make/break — are reported invalid because no
          force-weighted mean position is defined for them. Callers that
          want pure geometric "are they touching" detection should
          bool-AND on the raw ``get_contacts(...)["valid_mask"]``
          themselves.
        - ``force_norm_sum`` : ``(num_envs, n_a, n_b)`` — sum of contact
          force magnitudes (``Σ ‖f_i‖``, **not** ``‖Σ f_i‖``;
          opposing-direction contacts do not cancel) across the pair.
          The denominator used to weight the mean; useful as a
          contact-strength feature in its own right.

        Notes
        -----
        Requires a parallelised scene (``num_envs >= 1``); the
        non-parallelised path is not implemented since the only consumers
        — RL reward / observation terms — always run parallelised.
        """
        if not isinstance(other_entity, RigidEntity):
            raise TypeError(f"other_entity must be a RigidEntity, got {type(other_entity).__name__}")
        if other_entity._entity.idx == self._entity.idx:
            raise ValueError(
                "get_grouped_contacts does not support self-contact; use get_contacts(with_entity=self) directly."
            )

        contacts = self.get_contacts(with_entity=other_entity)
        if "valid_mask" not in contacts:
            raise NotImplementedError(
                "get_grouped_contacts requires the env-leading contact layout "
                "(`valid_mask` field present in get_contacts output). Genesis "
                "scenes built with `n_envs == 0` return non-parallelised "
                "contacts without this field; rebuild with `num_envs >= 1` if "
                "you need this aggregation."
            )

        device = contacts["position"].device
        a_local = _resolve_local_link_idx(links_a_idx_local, self.num_links, device)
        b_local = _resolve_local_link_idx(links_b_idx_local, other_entity.num_links, device)
        a_global = a_local + self.link_start
        b_global = b_local + other_entity.link_start
        return _aggregate_grouped_contacts(
            link_a=contacts["link_a"],
            link_b=contacts["link_b"],
            position=contacts["position"],
            force=contacts["force_a"],
            valid_mask=contacts["valid_mask"],
            a_global=a_global,
            b_global=b_global,
        )

    def detect_collision(self, env_idx=0):
        """Detect collision for the entity. This only supports a single environment."""
        return self._entity.detect_collision(env_idx=env_idx)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ setters --------------------------------------
    # ------------------------------------------------------------------------------------

    def set_qpos(
        self,
        qpos,
        qs_idx_local=None,
        envs_idx=None,
        *,
        zero_velocity=True,
        skip_forward=False,
    ) -> None:
        if qs_idx_local is None:
            qs_idx_local = self.qs_idx_local
        self._entity.set_qpos(
            qpos,
            qs_idx_local,
            envs_idx=envs_idx,
            zero_velocity=zero_velocity,
            skip_forward=skip_forward,
        )

    def set_friction(self, friction: float) -> None:
        """Set the friction coefficient for all of the entity's geoms.

        The friction coefficient must be in range [1e-2, 5.0] for simulation stability.

        Parameters
        ----------
        friction : float
            The friction coefficient to set for all the entity's links.
        """
        self._entity.set_friction(friction)

    def set_friction_ratio(self, friction_ratio, ls_idx_local, envs_idx=None) -> None:
        self._entity.set_friction_ratio(friction_ratio, ls_idx_local, envs_idx=envs_idx)

    def set_mass_shift(self, mass_shift, ls_idx_local, envs_idx=None) -> None:
        self._entity.set_mass_shift(mass_shift, ls_idx_local, envs_idx=envs_idx)

    def set_COM_shift(self, com_shift, ls_idx_local, envs_idx=None) -> None:
        self._entity.set_COM_shift(com_shift, ls_idx_local, envs_idx=envs_idx)

    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_force_range(lower, upper, dofs_idx_local, envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_stiffness(stiffness, dofs_idx_local, envs_idx)

    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_armature(armature, dofs_idx_local, envs_idx)

    def set_dofs_damping(self, damping, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_damping(damping, dofs_idx_local, envs_idx)

    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_kp(kp, dofs_idx_local, envs_idx)

    def set_dofs_kd(self, kd, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_kv(kd, dofs_idx_local, envs_idx)

    def set_dofs_vel(
        self,
        velocity=None,
        dofs_idx_local=None,
        envs_idx=None,
        *,
        skip_forward=False,
    ) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_velocity(velocity, dofs_idx_local, envs_idx, skip_forward=skip_forward)

    def set_dofs_frictionloss(self, frictionloss, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_frictionloss(frictionloss, dofs_idx_local, envs_idx)

    def set_dofs_pos(
        self,
        position,
        dofs_idx_local=None,
        envs_idx=None,
        *,
        zero_velocity=True,
    ) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.set_dofs_position(
            position,
            dofs_idx_local,
            envs_idx,
            zero_velocity=zero_velocity,
        )

    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.control_dofs_force(force, dofs_idx_local, envs_idx)

    def control_dofs_vel(self, velocity, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.control_dofs_velocity(velocity, dofs_idx_local, envs_idx)

    def control_dofs_pos(self, position, dofs_idx_local=None, envs_idx=None) -> None:
        if dofs_idx_local is None:
            dofs_idx_local = self.dofs_idx_local
        self._entity.control_dofs_position(position, dofs_idx_local, envs_idx)

    def zero_all_dofs_velocity(self, envs_idx=None) -> None:
        self._entity.zero_all_dofs_velocity(envs_idx)

    # ------------------------------------------------------------------------------------
    # ---------------------------- support surface related -------------------------------
    # ------------------------------------------------------------------------------------
    def place_on_to(
        self,
        entity: RigidEntity,
        sampler: Callable,
        envs_idx: slice | torch.Tensor | None = None,
    ):
        """Place this entity onto the support surface of another entity.

        Parameters
        ----------
        entity: RigidEntity
            the entity on which this object will be placed
        sampler: Callable
            the sampler to use for location sampling
        envs_idx: None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self._env.num_envs, return_n_envs=True)

        AABB = self.get_AABB()
        AABB = AABB[envs_idx]
        # NOTE: support bool envs_idx
        if AABB.shape[0] == 0:
            return
        clearance = (AABB[:, 1, 2] - AABB[:, 0, 2]).max(dim=0)[0]
        support = entity.current_support(
            minimum_clearance=clearance,
            gravity=self._env.rigid_solver.get_gravity(envs_idx=envs_idx),
            envs_idx=envs_idx,
        )
        # NOTE: force to update collision state
        entity.solver.collider.clear()
        # NOTE: update slightly bigger AABBs
        entity._kernel_update_aabbs(
            self.solver.geoms_state.pos,
            self.solver.geoms_state.quat,
            self.solver.geoms_init_AABB,
            self.solver.geoms_state.aabb_min,
            self.solver.geoms_state.aabb_max,
        )
        # NOTE: use broad phase to check the occupied space on the support surface
        from genesis.engine.solvers.rigid.collider import func_broad_phase

        func_broad_phase(
            self.solver.links_state,
            self.solver.links_info,
            self.solver.geoms_state,
            self.solver.geoms_info,
            self.solver._rigid_global_info,
            self.solver._static_rigid_sim_config,
            self.solver.constraint_solver.constraint_state,
            self.solver.collider._collider_state,
            self.solver.equalities_info,
            self.solver.collider._collider_info,
            self.solver._errno,
        )
        tensor = torch.zeros(
            (self._env.num_envs, self.solver.n_entities, 2, 3),
            dtype=gs.tc_float,
            device=self.device,
        )
        tensor[..., 0, :] = torch.inf
        tensor[..., 1, :] = -torch.inf
        valid_mask = torch.zeros(
            (self._env.num_envs, self.solver.n_entities),
            dtype=gs.tc_int,
            device=self.device,
        )
        # NOTE: get set of AABBs that is already occupied
        entity._kernel_filter_detection(
            tensor,
            valid_mask,
            self.solver.collider._collider_state.n_broad_pairs,
            self.solver.collider._collider_state.broad_collision_pairs,
            self.solver.geoms_info.link_idx,
            self.solver.links_info.entity_idx,
            self.solver.entities_info.geom_start,
            self.solver.entities_info.geom_end,
            self.solver.geoms_state.aabb_min,
            self.solver.geoms_state.aabb_max,
        )
        AABBs = tensor[:, (valid_mask == 1).any(dim=0)]
        AABBs = AABBs[envs_idx]

        pos, offset_direction, valid_mask = sampler(
            support,
            asset_AABB=AABB,
            holes=AABBs,
            batch_size=n_envs,
        )
        pos += torch.sum(self.offset * offset_direction, dim=-1, keepdim=True) * offset_direction
        if self.is_fixed_base and not self.morph.batch_fixed_verts:
            self.set_pos(pos)
        else:
            if valid_mask is not None:
                pos = pos[valid_mask]
                if isinstance(envs_idx, slice):
                    envs_idx = torch.arange(*envs_idx.indices(self._env.num_envs))[valid_mask.cpu()]
                else:
                    envs_idx = envs_idx[valid_mask]
            if isinstance(envs_idx, torch.Tensor):
                envs_idx = envs_idx.to(self.device)
            self.set_pos(pos, envs_idx=envs_idx)

    @cached_property
    def offset(self):
        if not self.is_heterogeneous:
            offset = torch.zeros(3, device=self.device, dtype=gs.tc_float)
            if self.is_fixed_base and not self.morph.batch_fixed_verts:
                pos_ = self.get_pos()
                self.set_pos(torch.zeros(self._env.num_envs, 3))
            else:
                pos_ = self.get_pos(envs_idx=0)
                self.set_pos(torch.zeros(1, 3), envs_idx=0)
            AABB = self.get_AABB()
            offset[:2] = -AABB[0, :, :2].mean(dim=-2)
            offset[2] = -AABB[0, 0, 2] + 0.001
            if self.is_fixed_base and not self.morph.batch_fixed_verts:
                self.set_pos(pos_)
            else:
                self.set_pos(pos_, envs_idx=0)
            return offset.unsqueeze(0)
        else:
            offset = torch.zeros(self._env.num_envs, 3, device=self.device, dtype=gs.tc_float)
            pos_ = self.get_pos()
            self.set_pos(torch.zeros(self._env.num_envs, 3))
            AABB = self.get_AABB()
            offset[:, :2] = -AABB[:, :, :2].mean(dim=-2)
            offset[:, 2] = -AABB[:, 0, 2] + 0.001
            self.set_pos(pos_)
            return offset
