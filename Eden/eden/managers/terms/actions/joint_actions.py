"""Joint-space PD and velocity action terms (implicit/explicit controllers)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

import eden as en
from eden.constants import ReferenceSource
from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.modifiers import ACTION_MODIFIER_REGISTRY, ActionModifier

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.modifiers import ActionModifierOptions


class _JointPDControllerBase(ActionTerm):
    """
    Base class for joint PD controllers.

    The processed target is built additively as::

        target = raw_action * scale + offset + reference_offset

    where ``reference_offset`` is selected by ``reference_source`` (see
    :class:`eden.constants.ReferenceSource`):

    - ``ReferenceSource.ZERO``: no reference offset.
    - ``ReferenceSource.DEFAULT`` (class default): the entity's
      ``default_dofs_pos`` for the controlled DOFs.
    - ``ReferenceSource.DELTA``: per-step joint position captured from the
      entity at the start of each control step (``entity.get_dofs_pos(...)``);
      the action is then interpreted as a delta on top of the previous step's
      DOF position. The captured offset is constant across decimation substeps.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: str | list[str]
        The names of the DOFs to control.
    reference_source: ReferenceSource
        Source of the reference offset added on top of ``raw_action * scale + offset``.
    scale: float | dict[str, float]
        The scale to apply to the actions.
    scale_ratio: float | None
        If set, replace ``scale`` with ``scale_ratio * (upper - lower)`` per
        controlled DOF, using the entity's DOF position limits.
    offset: float | dict[str, float]
        The offset to apply to the actions.
    modifier: ActionModifierOptions | None
        Optional composable action modifier (e.g., delay, friction, effort clip).
    """

    reference_source: ReferenceSource = ReferenceSource.DEFAULT
    scale: float | dict[str, float] = 1.0
    scale_ratio: float | None = None
    offset: float | dict[str, float] = 0.0
    modifier: ActionModifierOptions | None = None

    if TYPE_CHECKING:
        entity: RigidEntity

    def __init__(
        self,
        env: EnvBase,
        options: ActionTermOptions,
    ):
        super().__init__(env=env, options=options)
        self._scale: float | torch.Tensor | None = None
        self._offset: torch.Tensor | None = None
        self._dofs_pos_offset: torch.Tensor | None = None
        self._modifier: ActionModifier | None = None
        if self.modifier is not None:
            self._modifier = ACTION_MODIFIER_REGISTRY.get(self.modifier.name)(env=env, options=self.modifier)

    def build(self) -> None:
        super().build()
        self._build_scale_and_offset()

        if self.reference_source == ReferenceSource.DELTA:
            self._dofs_pos_offset = torch.zeros(self.num_envs, self._n_dofs, device=self.device)

        # Build action modifier if configured
        if self._modifier is not None:
            self._modifier.build(
                num_envs=self.num_envs,
                device=self.device,
                entity=self.entity,
                dofs_idx_local=self.dofs_idx_local,
            )

    def _has_nonzero_offset(self) -> bool:
        """Return True if the configured ``offset`` parameter is non-zero on any DOF."""
        if isinstance(self.offset, float):
            return self.offset != 0.0
        if isinstance(self.offset, dict):
            return any(v != 0.0 for v in self.offset.values())
        return False

    def _build_scale_and_offset(self) -> None:
        if self.scale_ratio is not None:
            if isinstance(self.scale_ratio, bool) or not isinstance(self.scale_ratio, int | float):
                raise ValueError(f"Unsupported scale_ratio type: {type(self.scale_ratio)}.")
            lower, upper = self.entity.get_dofs_limit(dofs_idx_local=self.dofs_idx_local)
            dof_range = upper - lower
            revolute_fallback = torch.full_like(dof_range, 2.0 * torch.pi)
            finite_range = torch.isfinite(dof_range)
            nonpositive_finite_range = finite_range & (dof_range <= 0.0)
            if nonpositive_finite_range.any().item():
                warnings.warn(
                    "scale_ratio encountered finite DOF limits with upper <= lower; clamping those DOF ranges to 1e-6.",
                    stacklevel=2,
                )
            dof_range = torch.where(finite_range, dof_range, revolute_fallback).clamp(min=1e-6)
            self._scale = float(self.scale_ratio) * dof_range
            if self._scale.ndim == 1:
                self._scale = self._scale.unsqueeze(0)
        elif isinstance(self.scale, float):
            self._scale = self.scale
        elif isinstance(self.scale, dict):
            self._scale = torch.ones(1, self.action_dim, device=self.device)
            for dofs_name, scale in self.scale.items():
                _, dofs_idx = self.entity.find_named_dofs_idx_local(dofs_name, name_scope=self.dofs_name)
                dofs_idx = [self.dofs_idx_map[idx] for idx in dofs_idx]
                self._scale[:, dofs_idx] = scale
        else:
            raise ValueError(f"Unsupported scale type: {type(self.scale)}.")

        # ``_offset`` is consumed by ``_compute_pos_target(raw_pos)`` where
        # ``raw_pos`` has shape (batch, n_dofs) — for sub-controllers like
        # ``VelocityFeedforwardPDController`` (action_dim = 2 * n_dofs)
        # sizing this to ``action_dim`` would break broadcasting on the
        # position half. Match the position-slice width instead.
        if isinstance(self.offset, float):
            self._offset = torch.full((1, self._n_dofs), self.offset, device=self.device)
        elif isinstance(self.offset, dict):
            self._offset = torch.zeros(1, self._n_dofs, device=self.device)
            for dofs_name, offset in self.offset.items():
                _, dofs_idx = self.entity.find_named_dofs_idx_local(dofs_name, name_scope=self.dofs_name)
                dofs_idx = [self.dofs_idx_map[idx] for idx in dofs_idx]
                self._offset[:, dofs_idx] = offset
        else:
            raise ValueError(f"Unsupported offset type: {type(self.offset)}.")

        # Fold default DOF positions into the static offset for `reference_source="default"`.
        if self.reference_source == ReferenceSource.DEFAULT:
            if self._has_nonzero_offset():
                en.logger.warning(
                    f"`offset` is non-zero on {type(self).__name__} with "
                    f"reference_source='default'; the static offset stacks additively "
                    f"on top of `default_dofs_pos`."
                )
            dofs_idx = [self.entity.dofs_name_map[name] for name in self.dofs_name]
            default_offset = self.entity.default_dofs_pos[:, dofs_idx].clone()
            self._offset = default_offset + self._offset

        if self.reference_source == ReferenceSource.DELTA:
            if self._has_nonzero_offset():
                en.logger.warning(
                    f"`offset` is non-zero on {type(self).__name__} with "
                    f"reference_source='delta'; the static offset stacks additively "
                    f"on top of the previous step's DOF positions."
                )

    def _compute_pos_target(self, raw_pos: torch.Tensor) -> torch.Tensor:
        """Build a position target from a raw action slice: ``raw * scale + offset (+ ref)``."""
        target = raw_pos * self._scale + self._offset
        if self._dofs_pos_offset is not None:
            target = target + self._dofs_pos_offset
        return target

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        if self._dofs_pos_offset is not None:
            # Capture once per env step; constant across decimation substeps.
            self._dofs_pos_offset[:] = self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local)
        self._processed_action = self._compute_pos_target(self._raw_action)

        # Apply modifier to processed action (e.g., delay).
        # The target is already absolute (delta + dofs_pos), so state-dependent
        # modifiers (e.g. EnvelopeClip) receive correct absolute joint positions.
        if self._modifier is not None:
            self._processed_action = self._modifier.modify_processed_action(self._processed_action)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0
        if self._dofs_pos_offset is not None:
            # Overwritten in the next compute(); zeroed here for parity with _raw_action.
            self._dofs_pos_offset[envs_idx] = 0.0

        if self._modifier is not None:
            self._modifier.reset(envs_idx)


@ACTION_TERM_REGISTRY.register()
class ImplicitPDController(_JointPDControllerBase):
    """
    Implicit joint PD controller using Genesis's built-in position control.

    Uses ``control_dofs_pos`` which leverages Genesis's implicit integration scheme:
    the PD target is recomputed at every physics substep and an implicit damping term
    (kd * substep_dt) is added to the mass matrix for numerical stability. This matches
    the behavior of MuJoCo, making it suitable for sim2sim transfer.
    """

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.dofs_idx_local)


@ACTION_TERM_REGISTRY.register()
class ExplicitPDController(_JointPDControllerBase):
    """
    Explicit joint PD controller using manual torque computation.

    Computes PD torques explicitly and applies them via ``control_dofs_force``.
    This is more physically accurate as it matches real-world motor control where
    only torque commands are valid. Better suited for sim2real transfer and easier
    to extend with domain randomization (e.g., randomizing kp/kd, adding motor delays).

    An optional ``modifier`` can be provided to compose post-processing steps
    (delay, backlash, DC saturation, friction, effort clipping, motor strength scaling)
    on top of the PD controller::

        ExplicitPDController.configure(
            entity_name="robot",
            dofs_name=["joint1", "joint2"],
            modifier=Compose.configure(
                modifiers=[
                    ActionDelay.configure(min_delay=1, max_delay=3),
                    FrictionModel.configure(static_friction=2.4, dynamic_friction=0.24),
                    EffortClip.configure(driving_torque_limit=111.0, ...),
                    MotorStrength.configure(motor_strength=0.8),
                ]
            ),
        )

    Note
    ----
    Requires ``sim_substeps=1`` so that PD torques are recomputed at every physics
    substep. Adjust ``decimation`` accordingly to maintain the desired control frequency.

    """

    def build(self) -> None:
        super().build()

        self._batch_dofs_info = self._env.env_options.batch_dofs_info

        # Cache PD gains
        self._kp = self.entity.get_dofs_kp(dofs_idx_local=self.dofs_idx_local)
        self._kd = self.entity.get_dofs_kd(dofs_idx_local=self.dofs_idx_local)
        if not self._batch_dofs_info:
            self._kp = self._kp.unsqueeze(0).repeat(self.num_envs, 1)
            self._kd = self._kd.unsqueeze(0).repeat(self.num_envs, 1)
        self._dofs_idx = [self.entity.dofs_name_map[name] for name in self.dofs_name]

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        if self._batch_dofs_info:
            # Update PD gains per-env for domain randomization support
            self._kp[envs_idx] = self.entity.get_dofs_kp(dofs_idx_local=self.dofs_idx_local, envs_idx=envs_idx)
            self._kd[envs_idx] = self.entity.get_dofs_kd(dofs_idx_local=self.dofs_idx_local, envs_idx=envs_idx)

    def apply_actions(self) -> None:
        dofs_vel = self.entity.get_dofs_vel(dofs_idx_local=self.dofs_idx_local)
        pos_err = self._processed_action - self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local)

        # PD control
        ctrl_torque = self._kp * pos_err - self._kd * dofs_vel

        # Apply modifiers to the computed torques (e.g., backlash, friction, effort clip)
        if self._modifier is not None:
            ctrl_torque = self._modifier.modify_ctrl_torque(ctrl_torque, dofs_vel, pos_err=pos_err)

        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.dofs_idx_local)


@ACTION_TERM_REGISTRY.register()
class VelocityFeedforwardPDController(_JointPDControllerBase):
    r"""
    Explicit joint PD controller with velocity feedforward.

    Extends the explicit PD controller by accepting both target joint positions
    and target joint velocities as actions. The torque is computed as:

    .. math::
        \\tau = k_p (q_{target} + q_{offset} - q_{current} + q_{motor\\_offset})
              + k_d (-\\dot{q}_{current} + \\dot{q}_{target} \\cdot r_{ff})

    where :math:`r_{ff}` is the velocity feedforward ratio. Post-processing
    (motor strength scaling, torque offset injection, effort clipping, etc.)
    is opt-in via the ``modifier`` parameter — see the composable modifiers in
    :mod:`eden.managers.modifiers.actions.actuators`.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: str | list[str]
        The names of the DOFs to control.
    reference_source: ReferenceSource
        Source of the reference offset added on top of the position action.
    scale: float | dict[str, float]
        The scale to apply to the position actions.
    offset: float | dict[str, float]
        The offset to apply to the position actions.
    velocity_scale: float
        The scale to apply to the velocity actions.
    feed_forward_ratio: float
        The ratio for the velocity feedforward term.

    .. note::
        Requires ``sim_substeps=1`` so that PD torques are recomputed at every physics
        substep. Adjust ``decimation`` accordingly to maintain the desired control frequency.
    """

    velocity_scale: float = 1.0
    feed_forward_ratio: float = 1.0

    @property
    def action_dim(self) -> int:
        return self._n_dofs * 2

    def build(self) -> None:
        super().build()

        # Re-allocate _processed_action to hold both position and velocity
        self._processed_action = torch.zeros(self.num_envs, self._n_dofs * 2, device=self.device)

        self._batch_dofs_info = self._env.env_options.batch_dofs_info

        # Cache PD gains
        self._kp = self.entity.get_dofs_kp(dofs_idx_local=self.dofs_idx_local)
        self._kd = self.entity.get_dofs_kd(dofs_idx_local=self.dofs_idx_local)
        self._dofs_idx = [self.entity.dofs_name_map[name] for name in self.dofs_name]

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        if self._batch_dofs_info:
            # Update PD gains per-env for domain randomization support
            self._kp[envs_idx] = self.entity.get_dofs_kp(dofs_idx_local=self.dofs_idx_local, envs_idx=envs_idx)
            self._kd[envs_idx] = self.entity.get_dofs_kd(dofs_idx_local=self.dofs_idx_local, envs_idx=envs_idx)

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        n_dofs = self._n_dofs
        if self._dofs_pos_offset is not None:
            # Capture once per env step; constant across decimation substeps.
            self._dofs_pos_offset[:] = self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local)
        joint_pos = self._compute_pos_target(self._raw_action[:, :n_dofs])
        joint_vel = self._raw_action[:, n_dofs:] * self.velocity_scale

        # Apply modifier to processed action (e.g., delay)
        if self._modifier is not None:
            joint_pos = self._modifier.modify_processed_action(joint_pos)

        self._processed_action[:, :n_dofs] = joint_pos
        self._processed_action[:, n_dofs:] = joint_vel

    def apply_actions(self) -> None:
        n_dofs = self._n_dofs

        target_pos = self._processed_action[:, :n_dofs]
        target_vel = self._processed_action[:, n_dofs:]
        dofs_vel = self.entity.get_dofs_vel(dofs_idx_local=self.dofs_idx_local)
        pos_err = target_pos - self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local)

        ctrl_torque = self._kp * pos_err + self._kd * (target_vel * self.feed_forward_ratio - dofs_vel)

        # Apply modifier to ctrl torque (e.g., backlash, friction, effort clip)
        if self._modifier is not None:
            ctrl_torque = self._modifier.modify_ctrl_torque(ctrl_torque, dofs_vel, pos_err=pos_err)

        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.dofs_idx_local)


@ACTION_TERM_REGISTRY.register()
class ImplicitVelocityController(_JointPDControllerBase):
    """Implicit joint velocity controller using Genesis's built-in velocity control."""

    # Override the parent's DEFAULT default — velocity controllers have no position reference.
    reference_source: ReferenceSource = ReferenceSource.ZERO

    def build(self) -> None:
        if self.reference_source != ReferenceSource.ZERO:
            raise ValueError(
                f"`reference_source='{self.reference_source}'` is not supported on {type(self).__name__}; "
                f"the action is interpreted as a target velocity."
            )
        super().build()

    def apply_actions(self) -> None:
        self.entity.control_dofs_vel(self._processed_action, dofs_idx_local=self.dofs_idx_local)


@ACTION_TERM_REGISTRY.register()
class ExplicitVelocityController(_JointPDControllerBase):
    r"""
    Explicit joint velocity controller using manual torque computation.

    The policy action is interpreted as a target joint velocity. Torque is computed
    as a proportional law on the velocity error:

    .. math::
        \\tau = k_p (\\dot{q}_{target} - \\dot{q})

    This matches the behavior of MuJoCo's ``velocity`` actuator and the implicit
    counterpart :class:`ImplicitVelocityController`. Use this variant when an
    explicit, auditable torque is required (e.g., sim2real, modifier composition,
    domain randomization of the gain).

    The gain is derived from the DOF force range so it is automatically scaled to
    each joint's effort capacity:

    .. math::
        k_p = (F_{max} - F_{min}) \\cdot \\text{velocity\\_gain}

    Parameters
    ----------
    entity_name : str
        The name of the entity to control.
    dofs_name : str | list[str]
        The names of the DOFs to control. Supports glob patterns.
    velocity_gain : float
        Scales the force range to produce :math:`k_p`.
    modifier : ActionModifierOptions | None
        Optional composable action modifier (e.g., effort clip, friction).

    .. note::
        Requires ``sim_substeps=1`` so that PD torques are recomputed at every physics
        substep. Adjust ``decimation`` accordingly to maintain the desired control frequency.

    .. note::
        ``reference_source`` must remain ``"zero"`` (this is a velocity
        controller). The inherited ``offset`` parameter operates in velocity
        units (rad/s).
    """

    # Override the parent's DEFAULT default — velocity controllers have no position reference.
    reference_source: ReferenceSource = ReferenceSource.ZERO
    velocity_gain: float = 0.25

    def _compute_gain(self, envs_idx=None) -> torch.Tensor:
        lower, upper = self.entity.get_dofs_force_range(dofs_idx_local=self.dofs_idx_local, envs_idx=envs_idx)
        return (upper - lower) * self.velocity_gain

    def build(self) -> None:
        if self.reference_source != ReferenceSource.ZERO:
            raise ValueError(
                f"`reference_source='{self.reference_source}'` is not supported on {type(self).__name__}; "
                f"the action is interpreted as a target velocity."
            )

        super().build()

        self._batch_dofs_info = self._env.env_options.batch_dofs_info

        self._kp = self._compute_gain()
        if not self._batch_dofs_info:
            self._kp = self._kp.unsqueeze(0).repeat(self.num_envs, 1)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        if self._batch_dofs_info:
            self._kp[envs_idx] = self._compute_gain(envs_idx=envs_idx)

    def apply_actions(self) -> None:
        dofs_vel = self.entity.get_dofs_vel(dofs_idx_local=self.dofs_idx_local)
        ctrl_torque = self._kp * (self._processed_action - dofs_vel)

        if self._modifier is not None:
            ctrl_torque = self._modifier.modify_ctrl_torque(ctrl_torque, dofs_vel)

        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.dofs_idx_local)


@ACTION_TERM_REGISTRY.register()
class NullJointAction(ActionTerm):
    """
    Virtually fix the DoFs to the default positions.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The names of the DOFs to control.
    """

    def __init__(
        self,
        env: EnvBase,
        options: ActionTermOptions,
    ):
        super().__init__(env=env, options=options)
        self._processed_action: torch.Tensor | None = None
        self._raw_action: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None

    def build(self) -> None:
        super().build()

        dofs_idx = [self.dofs_name_map[name] for name in self.dofs_name]
        self._processed_action = self.entity.default_dofs_pos[:, dofs_idx].clone()
        self._raw_action = torch.zeros_like(self._processed_action)
        self._prev_action = torch.zeros_like(self._processed_action)

    def compute(self, actions: torch.Tensor) -> None:
        pass

    def apply_actions(self) -> None:
        # NOTE: must provide self._dofs_idx_local as it might be different (or ordered differently) from the one in the entity
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.dofs_idx_local)


# TODO: add JointTorqueController, ActuatorNetMLPController, ActuatorNetLSTMController
