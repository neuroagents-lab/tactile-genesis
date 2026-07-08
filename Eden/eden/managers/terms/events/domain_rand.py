"""Domain-randomization event terms (pushes, external wrenches, DOF randomization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.typing import Vec2FType

from eden.constants import EventMode, ReferenceSource
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.options.managers.events import EventTermOptions
from eden.utils.isaac_math import quat_apply, quat_apply_inverse
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.managers.modifiers.actions.actuators import (
        ConstantTorqueKick,
        Deadband,
        GearBacklash,
        MotorStrength,
        TorqueOffset,
    )
    from eden.managers.modifiers.base import ActionModifier
    from eden.managers.terms.actions.joint_actions import (
        ExplicitPDController,
        VelocityFeedforwardPDController,
        _JointPDControllerBase,
    )


# Define proxies for fast lookup
_STARTUP, _RESET, _INTERVAL = EventMode


def _require_modifier(term, modifier_cls: type, event_cls_name: str, action_term_name: str) -> "ActionModifier":
    """Look up ``modifier_cls`` on ``term._modifier`` and raise a clear error if missing."""
    root = getattr(term, "_modifier", None)
    if root is None:
        raise ValueError(
            f"{event_cls_name} requires action term '{action_term_name}' to have a "
            f"{modifier_cls.__name__} modifier, but the term has no modifier configured. "
            f"Add {modifier_cls.__name__} to its modifier chain."
        )
    modifier = root.get(modifier_cls)
    if modifier is None:
        raise ValueError(
            f"{event_cls_name} requires action term '{action_term_name}' to have a "
            f"{modifier_cls.__name__} modifier. "
            f"Add {modifier_cls.__name__} to its modifier chain."
        )
    return modifier


@EVENT_TERM_REGISTRY.register()
class PushByVelocity(EventTerm):
    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    entity_name: str = "robot"
    base_joint_name: str = "floating_base_joint"
    velocity_range: dict[str, Vec2FType] = {}

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.entity: Entity | None = None
        self.base_dofs_index: torch.Tensor | None = None

        range_list = [self.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=self._env.device)

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        _, base_dofs_index = self.entity.find_named_dofs_idx_local(self.base_joint_name)
        self.base_dofs_index = torch.as_tensor(base_dofs_index, dtype=gs.tc_int, device=self.device).contiguous()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        lin_vel_w = self.entity.get_vel(envs_idx=envs_idx)
        ang_vel_w = self.entity.get_ang(envs_idx=envs_idx)
        quat_w = self.entity.get_quat(envs_idx=envs_idx)

        rand_vel = sample_uniform(
            self.ranges[:, 0],
            self.ranges[:, 1],
            (self._env.num_envs, 6),
            device=self._env.device,
        )[envs_idx]
        lin_vel_w += rand_vel[:, :3]
        # ang_vel_b = inv_transform_by_quat(ang_vel_w, quat_w)
        ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
        ang_vel_b += rand_vel[:, 3:]
        # ang_vel_w = transform_by_quat(ang_vel_b, quat_w)
        ang_vel_w = quat_apply(quat_w, ang_vel_b)
        self.entity.set_dofs_vel(
            torch.cat([lin_vel_w, ang_vel_w], dim=1),
            dofs_idx_local=self.base_dofs_index,
            envs_idx=envs_idx,
        )


@EVENT_TERM_REGISTRY.register()
class ApplyExternalForce(EventTerm):
    """
    Apply a randomized external force on specified links.

    Parameters
    ----------
    entity_name: str
        The name of the entity to apply the force to.
    links_name: list[str]
        The names of the links to apply the force to.
    force_x_range, force_y_range, force_z_range: Vec2FType
        Per-axis force range (N), sampled uniformly per env and link.
    ref: str
        Reference frame for the force application. One of ``"link_origin"``, ``"link_com"``, or ``"root_com"``.
    local: bool
        If true, the force is interpreted in the link's local frame; otherwise in the world frame.
    prob: float
        Per-env probability of applying the wrench at each interval fire. ``1.0`` (default) reproduces the prior
        all-envs behavior. ``< 1.0`` independently samples a Bernoulli(prob) mask per env per fire and zeros the
        wrench on envs that fail the draw.

    Notes
    -----
    Each term instance samples its own mask. Pairing ``ApplyExternalForce`` with ``ApplyExternalTorque`` at the config
    layer therefore gates them **independently** — the same env may receive force in one fire and not torque (or vice
    versa). This diverges from DexMachina's ``_random_force_torque``
    (``references/dexmachina/dexmachina/envs/randomizations.py:99``), which shares one mask across both signals so the
    same envs are zeroed for force AND torque atomically. Both terms draw from the global ``torch`` RNG, so seeding
    alone cannot synchronize them — anything that consumes RNG state between the two ``compute()`` calls (including
    the in-call ``sample_uniform`` draws) shifts the mask draws out of phase. If synchronized gating is required for
    parity, the only correct fix is a single event term that samples one mask and applies both wrenches in one call;
    that is left to a downstream PR.
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    entity_name: str = "robot"
    links_name: list[str] = []
    force_x_range: Vec2FType = (0.0, 0.0)
    force_y_range: Vec2FType = (0.0, 0.0)
    force_z_range: Vec2FType = (0.0, 0.0)
    ref: str = "link_origin"  # "link_origin" | "link_com" | "root_com"
    local: bool = False
    prob: float = 1.0

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.links_idx: torch.Tensor | None = None
        self.entity: RigidEntity | None = None

    def build(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"{type(self).__name__}.prob must be in [0.0, 1.0], got {self.prob}.")
        self.entity = self._env.entities[self.entity_name]
        resolved_links_name, _ = self.entity.find_named_links_idx_local(self.links_name)
        assert len(resolved_links_name) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.links_idx = torch.tensor(
            [self.entity._entity.get_link(name).idx for name in resolved_links_name],
            dtype=gs.tc_int,
            device=self.device,
        ).contiguous()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_links = len(self.links_idx)
        force = torch.stack(
            [
                sample_uniform(
                    self.force_x_range[0],
                    self.force_x_range[1],
                    (self._env.num_envs, n_links),
                    device=self._env.device,
                ),
                sample_uniform(
                    self.force_y_range[0],
                    self.force_y_range[1],
                    (self._env.num_envs, n_links),
                    device=self._env.device,
                ),
                sample_uniform(
                    self.force_z_range[0],
                    self.force_z_range[1],
                    (self._env.num_envs, n_links),
                    device=self._env.device,
                ),
            ],
            dim=-1,
        )[envs_idx]
        if self.prob < 1.0:
            mask = torch.rand((self._env.num_envs,), device=self._env.device) < self.prob
            force[~mask[envs_idx]] = 0.0
        self._env.rigid_solver.apply_links_external_force(
            force=force,
            links_idx=self.links_idx,
            envs_idx=envs_idx,
            ref=self.ref,
            local=self.local,
        )


@EVENT_TERM_REGISTRY.register()
class ApplyExternalTorque(EventTerm):
    """
    Apply a randomized external torque on specified links.

    Parameters
    ----------
    entity_name: str
        The name of the entity to apply the torque to.
    links_name: list[str]
        The names of the links to apply the torque to.
    torque_x_range, torque_y_range, torque_z_range: Vec2FType
        Per-axis torque range (N·m), sampled uniformly per env and link.
    ref: str
        Reference frame for the torque application. One of ``"link_origin"``, ``"link_com"``, or ``"root_com"``.
    local: bool
        If true, the torque is interpreted in the link's local frame; otherwise in the world frame.
    prob: float
        Per-env probability of applying the wrench at each interval fire. ``1.0`` (default) reproduces the prior
        all-envs behavior. ``< 1.0`` independently samples a Bernoulli(prob) mask per env per fire and zeros the
        wrench on envs that fail the draw.

    Notes
    -----
    See :class:`ApplyExternalForce` for the cross-term independence caveat: pairing this term with
    ``ApplyExternalForce`` gates force and torque independently, NOT through one shared mask as DexMachina does.
    Sharing a seed across the two terms does not synchronize them; only folding both into a single event term does.
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    entity_name: str = "robot"
    links_name: list[str] = []
    torque_x_range: Vec2FType = (0.0, 0.0)
    torque_y_range: Vec2FType = (0.0, 0.0)
    torque_z_range: Vec2FType = (0.0, 0.0)
    ref: str = "link_origin"  # "link_origin" | "link_com" | "root_com"
    local: bool = False
    prob: float = 1.0

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.links_idx: torch.Tensor | None = None
        self.entity: RigidEntity | None = None

    def build(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"{type(self).__name__}.prob must be in [0.0, 1.0], got {self.prob}.")
        self.entity = self._env.entities[self.entity_name]
        resolved_links_name, _ = self.entity.find_named_links_idx_local(self.links_name)
        assert len(resolved_links_name) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.links_idx = torch.tensor(
            [self.entity._entity.get_link(name).idx for name in resolved_links_name],
            dtype=gs.tc_int,
            device=self.device,
        ).contiguous()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_links = len(self.links_idx)
        torque = torch.stack(
            [
                sample_uniform(
                    self.torque_x_range[0],
                    self.torque_x_range[1],
                    (self._env.num_envs, n_links),
                    device=self._env.device,
                ),
                sample_uniform(
                    self.torque_y_range[0],
                    self.torque_y_range[1],
                    (self._env.num_envs, n_links),
                    device=self._env.device,
                ),
                sample_uniform(
                    self.torque_z_range[0],
                    self.torque_z_range[1],
                    (self._env.num_envs, n_links),
                    device=self._env.device,
                ),
            ],
            dim=-1,
        )[envs_idx]
        if self.prob < 1.0:
            mask = torch.rand((self._env.num_envs,), device=self._env.device) < self.prob
            torque[~mask[envs_idx]] = 0.0
        self._env.rigid_solver.apply_links_external_torque(
            torque=torque,
            links_idx=self.links_idx,
            envs_idx=envs_idx,
            ref=self.ref,
            local=self.local,
        )


@EVENT_TERM_REGISTRY.register()
class SetRandomDofsPos(EventTerm):
    """
    Set the joint positions of an entity's DOFs to a random position (default mode: reset).

    Parameters
    ----------
    entity_name: str
        The name of the entity to randomize.
    dofs_pos_range: Vec2FType
        When ``apply_as_ratio`` is false, absolute joint positions (radians) sampled uniformly per env and DOF.
        When ``apply_as_ratio`` is true, fractions of each DOF's limit range — e.g. ``(0.0, 0.5)`` samples
        positions in ``[lower, lower + 0.5 * (upper - lower)]`` per DOF; ``(0.0, 1.0)`` covers the full range.
        DOFs without finite limits fall back to a ``2 * pi`` range centered at 0.
    apply_as_ratio: bool
        If false, ``dofs_pos_range`` sets absolute positions in radians.
        If true, ``dofs_pos_range`` is interpreted as a fraction of each DOF's configured limit range.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    dofs_pos_range: Vec2FType = (0.0, 0.2)  # sample within the first 20% of the dofs limits
    apply_as_ratio: bool = True

    if TYPE_CHECKING:
        entity: RigidEntity

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        if self.apply_as_ratio:
            # Per-DOF lower limit and (upper - lower); unbounded joints fall back to a 2*pi range centered at 0
            # so the ratio still produces a sensible position. Shape is normalized to (1, n_dofs) for broadcasting.
            lower, upper = self.entity.get_dofs_limit()
            lower = lower.to(device=self._env.device)
            upper = upper.to(device=self._env.device)
            dof_range = upper - lower
            finite = torch.isfinite(dof_range)
            self._dof_range = torch.where(finite, dof_range, torch.full_like(dof_range, 2.0 * torch.pi))
            self._dof_lower = torch.where(finite, lower, torch.full_like(lower, -torch.pi))
            if self._dof_range.ndim == 1:
                self._dof_range = self._dof_range.unsqueeze(0)
                self._dof_lower = self._dof_lower.unsqueeze(0)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        sampled = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (self._env.num_envs, self.entity.num_dofs),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            # Slice per-env cached limits to match the subset; (1, n_dofs) broadcasts directly.
            dof_lower = self._dof_lower if self._dof_lower.shape[0] == 1 else self._dof_lower[envs_idx]
            dof_range = self._dof_range if self._dof_range.shape[0] == 1 else self._dof_range[envs_idx]
            dofs_pos = dof_lower + sampled * dof_range
        else:
            dofs_pos = sampled
        self.entity.set_dofs_pos(dofs_pos, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeDofsPosOffset(EventTerm):
    """
    Randomize the per-env DOF position offset baked into a PD controller's ``_offset`` cache.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term whose ``_offset`` should be randomized.
    dofs_pos_range: Vec2FType
        Per-DOF offset range (radians) sampled uniformly per env and DOF.
    """

    mode: EventMode = _RESET
    entity_name: str = ""
    action_term_name: str = ""
    dofs_pos_range: Vec2FType = (
        -0.01745,
        0.01745,
    )  # radians, about +/- 1 degree

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        self.term = self._env.action_manager.get_term(self.action_term_name)
        # NOTE: expand the offset for the randomization
        if self.term._offset.shape[0] != self._env.num_envs:
            self.term._offset = self.term._offset.repeat(self._env.num_envs, 1)
        # Cache DOF indices for slicing default_dofs_pos to the controlled subset
        self._dofs_idx = [self.entity.dofs_name_map[name] for name in self.term.dofs_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        offset = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (
                self._env.num_envs,
                len(self.term.dofs_name),
            ),
            device=self._env.device,
        )[envs_idx]

        if self.term.reference_source == "default":
            self.term._offset[envs_idx] = self.entity.default_dofs_pos[envs_idx][:, self._dofs_idx] + offset
        else:
            self.term._offset[envs_idx] = offset


@EVENT_TERM_REGISTRY.register()
class RandomizeStartupDofsPos(EventTerm):
    """
    Randomize the absolute joint positions of an entity's DOFs (default mode: reset).

    Writes joint positions directly via ``set_dofs_pos`` — useful for spawning each episode with a randomized
    initial pose. Despite the name, defaults to ``mode=_RESET`` so it fires on every episode reset; override to
    ``_STARTUP`` for a one-time draw.

    Parameters
    ----------
    entity_name: str
        The name of the entity to randomize.
    dofs_pos_range: Vec2FType
        When ``apply_as_ratio`` is false, absolute joint positions (radians) sampled uniformly per env and DOF.
        When ``apply_as_ratio`` is true, fractions of each DOF's limit range — e.g. ``(0.0, 0.5)`` samples
        positions in ``[lower, lower + 0.5 * (upper - lower)]`` per DOF; ``(0.0, 1.0)`` covers the full range.
        DOFs without finite limits fall back to a ``2 * pi`` range centered at 0.
    apply_as_ratio: bool
        If false (default), ``dofs_pos_range`` sets absolute positions in radians. If true, ``dofs_pos_range`` is
        interpreted as a fraction of each DOF's configured limit range.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    dofs_pos_range: Vec2FType = (-0.01745, 0.01745)  # radians, about +/- 1 degree
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        if self.apply_as_ratio:
            # Per-DOF lower limit and (upper - lower); unbounded joints fall back to a 2*pi range centered at 0
            # so the ratio still produces a sensible position. Shape is normalized to (1, n_dofs) for broadcasting.
            lower, upper = self.entity.get_dofs_limit()
            lower = lower.to(device=self._env.device)
            upper = upper.to(device=self._env.device)
            dof_range = upper - lower
            finite = torch.isfinite(dof_range)
            self._dof_range = torch.where(finite, dof_range, torch.full_like(dof_range, 2.0 * torch.pi))
            self._dof_lower = torch.where(finite, lower, torch.full_like(lower, -torch.pi))
            if self._dof_range.ndim == 1:
                self._dof_range = self._dof_range.unsqueeze(0)
                self._dof_lower = self._dof_lower.unsqueeze(0)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        sampled = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (self._env.num_envs, self.entity.num_dofs),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            # Slice per-env cached limits to match the subset; (1, n_dofs) broadcasts directly.
            dof_lower = self._dof_lower if self._dof_lower.shape[0] == 1 else self._dof_lower[envs_idx]
            dof_range = self._dof_range if self._dof_range.shape[0] == 1 else self._dof_range[envs_idx]
            dofs_pos = dof_lower + sampled * dof_range
        else:
            dofs_pos = sampled
        self.entity.set_dofs_pos(dofs_pos, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeStartupDofsPosBias(EventTerm):
    """
    Bake a per-env constant joint-zero bias into ``entity.default_dofs_pos`` at startup.

    Simulates motor calibration drift: each env gets a small fixed offset on every DOF of the entity, sampled once at
    startup and held constant for the lifetime of the env. The bias persists across resets, unlike the per-reset
    jitter from :class:`RandomizeDofsPosOffset`. The two compose: a calibration drift baked in at startup + small
    per-episode jitter on top.

    When ``action_terms_name`` lists one or more PD controllers, each controller's ``_offset`` cache (folded once at
    build time as ``default_offset + configured_offset``, see ``_JointPDControllerBase._build_scale_and_offset``) is
    updated in lockstep by adding the same per-env bias delta on the DOFs that controller owns. This avoids the
    "stale cache" trap where mutating ``entity.default_dofs_pos`` would diverge from a controller's pre-folded static
    offset.

    **Build-time contract.** ``build()`` performs strict validation so that any stale-cache footgun surfaces loudly at
    startup rather than as silent drift at runtime:

    1. ``action_terms_name`` must contain no duplicates.
    2. Each listed term must be a :class:`_JointPDControllerBase` instance, target the same entity, and have
       ``reference_source == ReferenceSource.DEFAULT``.
    3. Every same-entity action term that caches ``default_dofs_pos`` at build must be either listed (and updated in
       lockstep) or rejected with a clear error. Specifically:

       - ``_JointPDControllerBase`` with ``reference_source=DEFAULT`` not in the listed set: error (its ``_offset``
         would silently go stale).
       - :class:`NullJointAction` and :class:`OperationalSpaceController` cache ``default_dofs_pos.clone()`` at build
         with no refresh path; if either is present on the same entity, ``build()`` errors out.

    Parameters
    ----------
    entity_name: str
        The entity to bias.
    action_terms_name: list[str]
        Names of PD controller action terms whose ``_offset`` caches should be kept in sync with the bias. Each must
        be a :class:`_JointPDControllerBase` on ``entity_name`` with ``reference_source=DEFAULT``. Must list every
        such controller on the entity — see the build-time contract above. Empty list is fine if and only if no
        DEFAULT-source PD controllers exist on this entity.
    dofs_pos_range: Vec2FType
        Per-DOF bias range, sampled uniformly. Default ±1 degree.
    """

    mode: EventMode = _STARTUP
    entity_name: str = "robot"
    action_terms_name: list[str] = []
    dofs_pos_range: Vec2FType = (-0.01745, 0.01745)  # radians, about ±1 degree

    if TYPE_CHECKING:
        entity: RigidEntity

    def build(self) -> None:
        # Function-local imports avoid a load-time cycle through eden.managers.terms
        # (events is imported alongside actions; runtime isinstance checks are safe
        # because action_manager has fully built by the time event_manager builds).
        from eden.managers.terms.actions.joint_actions import (
            NullJointAction,
            _JointPDControllerBase,
        )
        from eden.managers.terms.actions.task_space_actions import OperationalSpaceController

        self.entity = self._env.entities[self.entity_name]

        # (1) reject duplicates — they would double-apply the bias to one
        # controller's `_offset` while only single-applying it to the entity default.
        duplicates = {name for name in self.action_terms_name if self.action_terms_name.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"{type(self).__name__}.action_terms_name has duplicate entries {sorted(duplicates)}; "
                f"each PD controller must appear at most once or its `_offset` would be biased twice "
                f"while `default_dofs_pos` is biased only once (recreating the double-counting bug "
                f"this term is designed to avoid)."
            )

        # (2) resolve each listed controller, validating type / entity / reference_source.
        # Failures here surface as a clear ValueError rather than as a downstream
        # AttributeError on a non-PD term's missing `reference_source` / `_offset`.
        listed_set = set(self.action_terms_name)
        self._targets: list[tuple[_JointPDControllerBase, list[int]]] = []
        for name in self.action_terms_name:
            term = self._env.action_manager.get_term(name)
            if not isinstance(term, _JointPDControllerBase):
                raise ValueError(
                    f"{type(self).__name__}.action_terms_name lists '{name}' which is a "
                    f"{type(term).__name__}, not a `_JointPDControllerBase`. Lockstep "
                    f"`_offset` updates only make sense for PD controllers that fold "
                    f"`default_dofs_pos` into a tensor `_offset` cache."
                )
            if term.entity_name != self.entity_name:
                raise ValueError(
                    f"{type(self).__name__}.action_terms_name lists '{name}' which targets "
                    f"entity '{term.entity_name}', but this term is biasing entity "
                    f"'{self.entity_name}'. Lockstep `_offset` updates only make sense for "
                    f"controllers on the same entity whose `default_dofs_pos` is being mutated."
                )
            if term.reference_source != ReferenceSource.DEFAULT:
                raise ValueError(
                    f"{type(self).__name__}.action_terms_name lists '{name}' but its "
                    f"reference_source={term.reference_source!r} != ReferenceSource.DEFAULT. "
                    f"Only default-source PD controllers fold `default_dofs_pos` into `_offset`; "
                    f"updating non-default controllers' `_offset` here would be incorrect."
                )
            dofs_idx = [self.entity.dofs_name_map[dof] for dof in term.dofs_name]
            # Lazy-expand the broadcast (1, n_dofs) offset to per-env (num_envs, n_dofs)
            # so the bias write below has the right shape. Matches RandomizeDofsPosOffset.
            if term._offset.shape[0] != self._env.num_envs:
                term._offset = term._offset.repeat(self._env.num_envs, 1)
            self._targets.append((term, dofs_idx))

        # (3) scan all same-entity action terms for unhandled `default_dofs_pos` cache
        # holders. Any DEFAULT-source PD controller not listed, or any action term that
        # snapshots defaults at build with no refresh API (NullJointAction,
        # OperationalSpaceController), would silently drift after the bias mutates the
        # entity defaults. Fail loud rather than ship a half-coupled config.
        for action_name, action_term in self._env.action_manager._terms.items():
            if getattr(action_term, "entity_name", None) != self.entity_name:
                continue
            if action_name in listed_set:
                continue
            if (
                isinstance(action_term, _JointPDControllerBase)
                and action_term.reference_source == ReferenceSource.DEFAULT
            ):
                raise ValueError(
                    f"{type(self).__name__}: PD controller '{action_name}' on entity "
                    f"'{self.entity_name}' has reference_source=DEFAULT (folds "
                    f"`default_dofs_pos` into `_offset` at build), but is not listed in "
                    f"`action_terms_name`. Add it to keep its cache in sync, or change "
                    f"its reference_source so it doesn't snapshot the pre-bias defaults."
                )
            if isinstance(action_term, (NullJointAction, OperationalSpaceController)):
                raise ValueError(
                    f"{type(self).__name__}: action term '{action_name}' "
                    f"({type(action_term).__name__}) on entity '{self.entity_name}' "
                    f"caches `default_dofs_pos` at build time and provides no refresh path. "
                    f"Mutating the entity default would leave its cache stale. Either remove "
                    f"the action term or stop biasing this entity."
                )

        self._applied = False

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Idempotence: startup terms are invoked once today, but make the contract explicit.
        if self._applied:
            return
        self._applied = True

        if envs_idx is None:
            envs_idx = slice(None)

        bias_full = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (self._env.num_envs, self.entity.default_dofs_pos.shape[1]),
            device=self._env.device,
        )[envs_idx]

        # 1. Bake the bias into the entity's default DOF positions (additive).
        self.entity.default_dofs_pos[envs_idx] = self.entity.default_dofs_pos[envs_idx] + bias_full

        # 2. Keep each listed PD controller's pre-folded `_offset` cache in lockstep,
        #    on the subset of DOFs that controller owns.
        for term, dofs_idx in self._targets:
            term._offset[envs_idx] = term._offset[envs_idx] + bias_full[:, dofs_idx]


@EVENT_TERM_REGISTRY.register()
class RandomizeMotorStrength(EventTerm):
    """
    Randomize the motor strength scalar of a :class:`MotorStrength` action modifier (default factor 0.9 to 1.1).

    Requires
    --------
    The action term must have a ``MotorStrength`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``MotorStrength`` modifier.
    motor_strength_range: Vec2FType
        Multiplicative range applied to the modifier's strength, sampled uniformly per env (one scalar per env).
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    action_term_name: str = ""
    motor_strength_range: Vec2FType = (0.9, 1.1)

    if TYPE_CHECKING:
        entity: RigidEntity
        term: ExplicitPDController | VelocityFeedforwardPDController
        modifier: MotorStrength

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import MotorStrength

        self.entity = self._env.entities[self.entity_name]
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, MotorStrength, type(self).__name__, self.action_term_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        motor_strength = sample_uniform(
            self.motor_strength_range[0],
            self.motor_strength_range[1],
            (self._env.num_envs, 1),
            device=self._env.device,
        )[envs_idx]
        self.modifier._motor_strength[envs_idx] = motor_strength


@EVENT_TERM_REGISTRY.register()
class RandomizeGearBacklash(EventTerm):
    """
    Randomize the one-sided target offset of a :class:`GearBacklash` action modifier.

    Requires
    --------
    The action term must have a ``GearBacklash`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``GearBacklash`` modifier.
    backlash_range: Vec2FType
        When ``apply_as_ratio`` is false, the range of one-sided backlash offsets (rad) sampled uniformly per env and
        DOF. When ``apply_as_ratio`` is true, the range of multipliers applied to the modifier's configured per-DOF
        baseline (see ``apply_as_ratio``).
    apply_as_ratio: bool
        If false (default), ``backlash_range`` sets absolute backlash offsets in radians. If true, ``backlash_range``
        is interpreted as ``(low, high)`` multipliers (uniform per env and DOF): the backlash becomes
        ``m * backlash_i``, where ``backlash_i`` is that DOF's configured nominal backlash from
        :class:`~eden.managers.modifiers.actions.actuators.GearBacklash` at build time.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    action_term_name: str = ""
    backlash_range: Vec2FType = (0.0, 0.01)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase
        modifier: GearBacklash

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import GearBacklash

        self.entity = self._env.entities[self.entity_name]
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, GearBacklash, type(self).__name__, self.action_term_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_ctrl = len(self.term.dofs_name)
        sampled = sample_uniform(
            self.backlash_range[0],
            self.backlash_range[1],
            (self._env.num_envs, n_ctrl),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            baseline = self.modifier._backlash_row.to(device=sampled.device, dtype=sampled.dtype)
            backlash = sampled * baseline.unsqueeze(0)
        else:
            backlash = sampled
        self.modifier._backlash[envs_idx] = backlash.clamp(min=0.0)


@EVENT_TERM_REGISTRY.register()
class RandomizeConstantTorqueKick(EventTerm):
    """
    Randomize the torque magnitude of a :class:`ConstantTorqueKick` action modifier.

    Requires
    --------
    The action term must have a ``ConstantTorqueKick`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``ConstantTorqueKick`` modifier.
    torque_kick_range: Vec2FType
        When ``apply_as_ratio`` is false, the range of torque-kick magnitudes (N·m) sampled uniformly per env and
        DOF. When ``apply_as_ratio`` is true, the range of multipliers applied to the modifier's configured per-DOF
        baseline (see ``apply_as_ratio``).
    apply_as_ratio: bool
        If false (default), ``torque_kick_range`` sets absolute torque-kick magnitudes. If true, ``torque_kick_range``
        is interpreted as ``(low, high)`` multipliers (uniform per env and DOF): the kick becomes
        ``m * torque_kick_i``, where ``torque_kick_i`` is that DOF's configured nominal magnitude from
        :class:`~eden.managers.modifiers.actions.actuators.ConstantTorqueKick` at build time.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    action_term_name: str = ""
    torque_kick_range: Vec2FType = (0.0, 0.01)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase
        modifier: ConstantTorqueKick

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import ConstantTorqueKick

        self.entity = self._env.entities[self.entity_name]
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, ConstantTorqueKick, type(self).__name__, self.action_term_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_ctrl = len(self.term.dofs_name)
        sampled = sample_uniform(
            self.torque_kick_range[0],
            self.torque_kick_range[1],
            (self._env.num_envs, n_ctrl),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            baseline = self.modifier._torque_kick_row.to(device=sampled.device, dtype=sampled.dtype)
            torque_kick = sampled * baseline.unsqueeze(0)
        else:
            torque_kick = sampled
        self.modifier._torque_kick[envs_idx] = torque_kick.clamp(min=0.0)


@EVENT_TERM_REGISTRY.register()
class RandomizeKpKdGains(EventTerm):
    """
    Randomize the PD gains of an entity's joints by a multiplicative factor (default 0.9 to 1.1).

    Requires
    --------
    ``env_options.batch_dofs_info=True`` to support per-env PD gain randomization.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str | None
        If set, the gain factors are written into the named PD controller's ``_kp`` / ``_kd`` caches. If ``None``,
        the gains are written directly to the entity via ``set_dofs_kp`` / ``set_dofs_kd``.
    kp_range: Vec2FType
        Multiplicative range applied to ``entity.default_dofs_kp``, sampled uniformly per env and DOF.
    kd_range: Vec2FType
        Multiplicative range applied to ``entity.default_dofs_kd``, sampled uniformly per env and DOF.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    action_term_name: str | None = None
    kp_range: Vec2FType = (0.9, 1.1)
    kd_range: Vec2FType = (0.9, 1.1)

    if TYPE_CHECKING:
        entity: RigidEntity
        term: ExplicitPDController | VelocityFeedforwardPDController

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        if self.action_term_name is not None:
            self.term = self._env.action_manager.get_term(self.action_term_name)
        else:
            self.term = None
        assert self._env.env_options.batch_dofs_info, (
            f"{self.__class__.__name__} requires `batch_dofs_info=True` in env_options "
            "to support per-env PD gain randomization."
        )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_dofs = len(self.entity.dofs_name)

        kp_factor = sample_uniform(
            self.kp_range[0],
            self.kp_range[1],
            (self._env.num_envs, n_dofs),
            device=self._env.device,
        )[envs_idx]
        kd_factor = sample_uniform(
            self.kd_range[0],
            self.kd_range[1],
            (self._env.num_envs, n_dofs),
            device=self._env.device,
        )[envs_idx]

        # NOTE: default_dofs_kp/kd may be (1, n_dofs) even with batch_dofs_info=True
        base_kp = self.entity.default_dofs_kp.expand(self._env.num_envs, -1)[envs_idx]
        base_kd = self.entity.default_dofs_kd.expand(self._env.num_envs, -1)[envs_idx]

        if self.term is not None:
            self.term._kp[envs_idx] = base_kp * kp_factor
            self.term._kd[envs_idx] = base_kd * kd_factor
        else:
            self.entity.set_dofs_kp(
                base_kp * kp_factor,
                dofs_idx_local=self.entity.dofs_idx_local,
                envs_idx=envs_idx,
            )
            self.entity.set_dofs_kd(
                base_kd * kd_factor,
                dofs_idx_local=self.entity.dofs_idx_local,
                envs_idx=envs_idx,
            )


@EVENT_TERM_REGISTRY.register()
class RandomizeDeadbandEpsilon(EventTerm):
    """
    Randomize the position-error threshold of a :class:`Deadband` action modifier.

    Requires
    --------
    The action term must have a ``Deadband`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``Deadband`` modifier.
    deadband_epsilon_range: Vec2FType
        When ``apply_as_ratio`` is false, the range of the deadband (position error, rad) sampled uniformly per env
        and DOF. When ``apply_as_ratio`` is true, the range of multipliers applied to the modifier's configured
        per-DOF baseline (see ``apply_as_ratio``).
    apply_as_ratio: bool
        If false (default), ``deadband_epsilon_range`` sets absolute deadband values in radians. If true,
        ``deadband_epsilon_range`` is interpreted as ``(low, high)`` multipliers (uniform per env and DOF): the
        deadband becomes ``m * epsilon_i``, where ``epsilon_i`` is that DOF's configured nominal deadband from
        :class:`~eden.managers.modifiers.actions.actuators.Deadband` at build time.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    action_term_name: str = ""
    deadband_epsilon_range: Vec2FType = (0.0, 0.01)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase
        modifier: Deadband

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import Deadband

        self.entity = self._env.entities[self.entity_name]
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, Deadband, type(self).__name__, self.action_term_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_ctrl = len(self.term.dofs_name)
        sampled = sample_uniform(
            self.deadband_epsilon_range[0],
            self.deadband_epsilon_range[1],
            (self._env.num_envs, n_ctrl),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            baseline = self.modifier._deadband_epsilon_row.to(device=sampled.device, dtype=sampled.dtype)
            eps = sampled * baseline.unsqueeze(0)
        else:
            eps = sampled
        self.modifier._deadband_epsilon[envs_idx] = eps


@EVENT_TERM_REGISTRY.register()
class RandomizeFrictionRatio(EventTerm):
    mode: EventMode = _RESET
    entity_name: str = "robot"
    links_name: list[str] = []
    friction_range: Vec2FType = (0.3, 1.2)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: Entity | None = None

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        self.links_name, ls_idx_local = entity.find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = entity

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # NOTE: set the friction of the terrain and objects small so that the robot geometry can decide the friction
        # NOTE: the friction between the object A and B is defined by the maximum friction of the two objects
        # NOTE: friction_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
        # NOTE: friction_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_a]
        # NOTE: contact_data.friction[i_c, i_b] = ti.max(ti.max(friction_a, friction_b), 1e-2)
        # NOTE: geoms_info.friction defaults to 1.0 and has to be in range [1e-2, 5.0] for simulation stability.

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


@EVENT_TERM_REGISTRY.register()
class RandomizeComShift(EventTerm):
    mode: EventMode = _RESET
    entity_name: str = "robot"
    links_name: list[str] = []
    com_shift_x_range: Vec2FType = (-0.01, 0.01)
    com_shift_y_range: Vec2FType = (-0.01, 0.01)
    com_shift_z_range: Vec2FType = (-0.01, 0.01)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: Entity | None = None

    def build(self) -> None:
        _, ls_idx_local = self._env.entities[self.entity_name].find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = self._env.entities[self.entity_name]

    def _sample_com_shift_axis(self, axis_range: Vec2FType) -> torch.Tensor:
        return sample_uniform(
            axis_range[0],
            axis_range[1],
            (
                self._env.num_envs,
                len(self.ls_idx_local),
                1,
            ),
            device=self._env.device,
        )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        com_shift = torch.cat(
            [
                self._sample_com_shift_axis(self.com_shift_x_range),
                self._sample_com_shift_axis(self.com_shift_y_range),
                self._sample_com_shift_axis(self.com_shift_z_range),
            ],
            dim=-1,
        )[envs_idx]
        self.entity.set_COM_shift(com_shift, ls_idx_local=self.ls_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeMassShift(EventTerm):
    mode: EventMode = _RESET
    entity_name: str = "robot"
    links_name: list[str] = []
    mass_shift_range: Vec2FType = (-1.0, 3.0)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: Entity | None = None

    def build(self) -> None:
        _, ls_idx_local = self._env.entities[self.entity_name].find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = self._env.entities[self.entity_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        mass_shift = sample_uniform(
            self.mass_shift_range[0],
            self.mass_shift_range[1],
            (
                self._env.num_envs,
                len(self.ls_idx_local),
            ),
            device=self._env.device,
        )[envs_idx]
        self.entity.set_mass_shift(mass_shift, ls_idx_local=self.ls_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeLinkMassScale(EventTerm):
    """
    Multiplicative per-link mass randomization.

    Scales each specified link's mass by a random factor sampled from ``[mass_scale_range[0], mass_scale_range[1]]``.
    Internally uses ``set_mass_shift`` with ``shift = mass * (scale - 1)``.

    Parameters
    ----------
    entity_name: str
        The entity to randomize.
    links_name: list[str]
        Link name patterns to randomize. Use ``["*"]`` for all links.
    mass_scale_range: Vec2FType
        Multiplicative scale range, e.g. ``(0.9, 1.2)`` for 90-120%.
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    mass_scale_range: Vec2FType = (0.9, 1.1)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: RigidEntity | None = None
        self._base_mass: torch.Tensor | None = None

    def build(self) -> None:
        _, ls_idx_local = self._env.entities[self.entity_name].find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = self._env.entities[self.entity_name]
        # Cache the default (unshifted) mass per link for env 0 and broadcast
        full_mass = self.entity.get_mass()  # (num_envs, num_links)
        # Select only the links we care about and take env 0 as the reference
        self._base_mass = full_mass[0, self.ls_idx_local.long()].clone()  # (num_selected_links,)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Sample at full env count and index — mirrors the rest of this module
        # and avoids shape mismatches when ``envs_idx`` is a slice subset.
        scale = sample_uniform(
            self.mass_scale_range[0],
            self.mass_scale_range[1],
            (self._env.num_envs, len(self.ls_idx_local)),
            device=self._env.device,
        )[envs_idx]
        # shift = base_mass * (scale - 1)
        mass_shift = self._base_mass.unsqueeze(0) * (scale - 1.0)
        self.entity.set_mass_shift(mass_shift, ls_idx_local=self.ls_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeTorqueNoise(EventTerm):
    """
    Residual Force Injection (RFI) — adds random torque noise to actuated joints.

    At each physics step, samples ``[-rfi_scale, rfi_scale]`` per DOF and writes it into the ``TorqueOffset``
    modifier's ``_torque_offset`` buffer, which the action term reads when applying control torques. Simulates
    actuator noise and unmodeled dynamics.

    Must be used with ``mode = "interval"`` and an ``interval_range_s`` matching the control timestep (set
    automatically if left as ``None``).

    Requires
    --------
    The action term must have a ``TorqueOffset`` modifier in its chain.

    Parameters
    ----------
    action_term_name: str
        Name of the action term whose ``TorqueOffset`` modifier should receive noise.
    rfi_scale: float
        Maximum magnitude of the random torque noise (N·m).
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    action_term_name: str = ""
    rfi_scale: float = 0.5

    if TYPE_CHECKING:
        term: ExplicitPDController | VelocityFeedforwardPDController
        modifier: TorqueOffset

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import TorqueOffset

        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, TorqueOffset, type(self).__name__, self.action_term_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Always sample for the full env count and index — matches the rest of
        # this module's pattern and avoids subtle shape mismatches when ``envs_idx``
        # is a slice subset (`n_sample == num_envs` but only a slice writes back).
        noise = sample_uniform(
            -self.rfi_scale,
            self.rfi_scale,
            (self._env.num_envs, self.term._n_dofs),
            device=self._env.device,
        )
        self.modifier._torque_offset[envs_idx] = noise[envs_idx]


@EVENT_TERM_REGISTRY.register()
class ERFI50(EventTerm):
    """
    ERFI-50 noise: 50% Random Actuation Offset (RAO) + 50% Random Force Injection (RFI).

    For each environment, a random value *r* in [0, 1] is drawn at reset:

    - **r < erfi_ratio (RAO):** a fixed scalar offset ``r * 3 * upper_bound`` is broadcast to all DOFs and held
      constant for the entire episode.
    - **r >= erfi_ratio (RFI):** independent per-DOF torque noise drawn uniformly from ``[lower_bound, upper_bound]``.

    Requires
    --------
    The action term must have a ``TorqueOffset`` modifier.

    Parameters
    ----------
    interval_range_s: tuple[float, float] | None
        The interval range in seconds. Expected to be set to the control timestep.
    action_term_name: str
        The name of the action term to apply the noise to.
    lower_bound: float
        The lower bound of the torque noise (N·m).
    upper_bound: float
        The upper bound of the torque noise (N·m).
    erfi_ratio: float
        The fraction of envs that use RAO instead of RFI.

    Reference
    ---------
    - Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization
      (https://arxiv.org/pdf/2209.12878)
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    action_term_name: str = ""
    lower_bound: float = -7.0 / 9.0
    upper_bound: float = 7.0 / 9.0
    erfi_ratio: float = 0.5

    if TYPE_CHECKING:
        term: ExplicitPDController | VelocityFeedforwardPDController
        modifier: TorqueOffset

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import TorqueOffset

        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, TorqueOffset, type(self).__name__, self.action_term_name)
        self._erfi_random = sample_uniform(0.0, 1.0, (self._env.num_envs,), device=self._env.device)
        self._is_rao = (self._erfi_random < self.erfi_ratio).unsqueeze(1)
        self._rao_noise = (self._erfi_random * 3.0 * self.upper_bound).unsqueeze(1).repeat(1, self.term._n_dofs)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Resample per-env RAO / RFI selector.
        self._erfi_random[envs_idx] = sample_uniform(
            0.0,
            1.0,
            (self._env.num_envs,),
            device=self._env.device,
        )[envs_idx]
        r = self._erfi_random[envs_idx]
        self._is_rao[envs_idx] = (r < self.erfi_ratio).unsqueeze(1)

        # RAO: fixed scalar noise broadcast to all DOFs.
        self._rao_noise[envs_idx] = (r * 3.0 * self.upper_bound).unsqueeze(1).expand(-1, self.term._n_dofs)

        # Apply initial offset so first post-reset step is consistent.
        rfi_noise = sample_uniform(
            self.lower_bound, self.upper_bound, (self._env.num_envs, self.term._n_dofs), device=self._env.device
        )[envs_idx]
        self.modifier._torque_offset[envs_idx] = torch.where(
            self._is_rao[envs_idx], self._rao_noise[envs_idx], rfi_noise
        )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)

        # RFI: independent per-DOF noise.
        rfi_noise = sample_uniform(
            self.lower_bound,
            self.upper_bound,
            (self._env.num_envs, self.term._n_dofs),
            device=self._env.device,
        )[envs_idx]

        self.modifier._torque_offset[envs_idx] = torch.where(
            self._is_rao[envs_idx], self._rao_noise[envs_idx], rfi_noise
        )
