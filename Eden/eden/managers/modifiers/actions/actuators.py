"""Composable action modifiers for action terms.

Action modifiers sit between action terms and the physics engine. They
intercept and transform processed actions (target positions/velocities)
and/or control torques via composable operations like clipping, delay,
friction, and motor strength scaling.

Modifiers are designed to be composed together using :class:`Compose`,
similar to ``torchvision.transforms.Compose``::

    modifiers = Compose.configure(
        modifiers=[
            ActionDelay.configure(min_delay=1, max_delay=3),
            FrictionModel.configure(),
            EffortClip.configure(),
        ]
    )

Each modifier implements two hooks:

- ``modify_processed_action``: transforms the target positions/velocities
  before PD control (e.g., delay).
- ``modify_ctrl_torque``: transforms the computed torques after PD control
  (e.g., friction, effort clipping, motor strength scaling).

Modifiers that need motor spec parameters (friction, T-N curve limits) read
them automatically from the entity's actuator spec (``dofs_spec``) at build
time.  Explicit overrides can still be passed to use custom values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.typing import LaxFArrayType

from eden.managers.modifiers.base import ACTION_MODIFIER_REGISTRY, ActionModifier
from eden.options.managers.modifiers import ActionModifierOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

CURVE_SPEED_DENOM_MIN = 1e-6
MOTOR_STALL_VEL_THRESHOLD = 0.01


def _per_dof_float_row(
    values: LaxFArrayType,
    n_dof: int,
    device: str,
    *,
    param_name: str,
) -> torch.Tensor:
    """Broadcast length-1 inputs or match ``n_dof`` entries (see ``LaxFArrayType``)."""
    flat = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)
    n = flat.numel()
    if n == 1:
        return torch.full((n_dof,), flat.item(), dtype=torch.float32, device=device)
    if n != n_dof:
        raise ValueError(
            f"{param_name} length ({n}) must be 1 (broadcast) or match number of controlled DOFs ({n_dof})."
        )
    return flat


def _get_param_tensor_from_entity(
    param_value: LaxFArrayType | None,
    *,
    spec_attr: str,
    entity,
    dofs_idx_local,
    device: str,
) -> torch.Tensor:
    """Explicit ``LaxFArrayType`` → ``(1, n_dof)`` row(s); *None* → entity spec slice."""
    if param_value is None:
        return _get_entity_spec(entity, dofs_idx_local, spec_attr)
    if dofs_idx_local is None:
        flat = torch.as_tensor(param_value, dtype=torch.float32, device=device).reshape(-1)
        if flat.numel() != 1:
            raise ValueError(
                f"{spec_attr} requires dofs_idx_local when passing a per-DOF iterable (got length {flat.numel()})."
            )
        return flat.squeeze()
    n_dof = len(dofs_idx_local)
    row = _per_dof_float_row(param_value, n_dof, device, param_name=spec_attr)
    return row.unsqueeze(0)


def _explicit_lax_to_tensor(
    values: LaxFArrayType,
    *,
    param_name: str,
    dofs_idx_local,
    device: str,
) -> torch.Tensor:
    """Explicit ``LaxFArrayType`` only: ``(1, n_dof)`` or a scalar tensor when ``dofs_idx_local`` is absent."""
    if dofs_idx_local is None:
        flat = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)
        if flat.numel() != 1:
            raise ValueError(
                f"{param_name} requires dofs_idx_local when passing a per-DOF iterable (got length {flat.numel()})."
            )
        return flat.squeeze()
    n_dof = len(dofs_idx_local)
    row = _per_dof_float_row(values, n_dof, device, param_name=param_name)
    return row.unsqueeze(0)


# ---------------------------------------------------------------------------
# Helper: read a per-DOF spec property from the entity
# ---------------------------------------------------------------------------


def _get_entity_spec(entity, dofs_idx_local, attr: str) -> torch.Tensor:
    """Return entity actuator-spec tensor sliced to the controlled DOFs.

    The entity stores spec properties as ``entity._<attr>`` with shape
    ``(1, entity.num_dofs)``.  ``dofs_idx_local`` are genesis-solver-local
    indices (e.g. 6..34 for a floating-base robot with 29 actuated DOFs).
    We convert them to entity-local indices (0..28) via ``entity.dofs_idx_map``
    before slicing.
    """
    if entity is None:
        raise ValueError(
            "entity must be provided to read actuator spec. "
            "Make sure the modifier is used with a controller that has an entity."
        )
    if dofs_idx_local is None:
        raise ValueError(
            "dofs_idx_local must be provided to read actuator spec. "
            "Make sure the modifier is used with a controller that has dofs_idx_local."
        )
    tensor = getattr(entity, f"_{attr}", None)
    if tensor is None:
        raise AttributeError(
            f"Entity '{entity.name}' has no actuator spec attribute '_{attr}'. "
            "Make sure the entity has a 'dofs_spec' configured."
        )
    # Map genesis-local DOF indices to entity-local (0-based) indices.
    entity_local_idx = [entity.dofs_idx_map[idx.item() if hasattr(idx, "item") else idx] for idx in dofs_idx_local]
    return tensor[:, entity_local_idx]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _tanh_friction(
    ctrl_torque: torch.Tensor,
    dofs_vel: torch.Tensor,
    dofs_static_friction: torch.Tensor,
    dofs_dynamic_friction: torch.Tensor,
    friction_activation_vel: torch.Tensor,
    friction_offset: torch.Tensor,
) -> torch.Tensor:
    ctrl_torque -= (
        dofs_static_friction * torch.tanh(dofs_vel / friction_activation_vel)
        + dofs_dynamic_friction * dofs_vel
        + friction_offset
    )
    return ctrl_torque


def _clip_effort(
    dofs_torque: torch.Tensor,
    dofs_vel: torch.Tensor,
    driving_torque_limit: torch.Tensor,
    braking_torque_limit: torch.Tensor,
    full_torque_speed: torch.Tensor,
    no_load_speed: torch.Tensor,
) -> torch.Tensor:
    same_direction = (dofs_vel * dofs_torque) > 0
    max_effort = torch.where(same_direction, driving_torque_limit, braking_torque_limit)
    max_effort = torch.where(
        dofs_vel.abs() < full_torque_speed,
        max_effort,
        _compute_effort_limit(max_effort, dofs_vel, full_torque_speed, no_load_speed),
    )
    return torch.clip(dofs_torque, -max_effort, max_effort)


def _compute_effort_limit(
    max_effort: torch.Tensor,
    dofs_vel: torch.Tensor,
    full_torque_speed: torch.Tensor,
    no_load_speed: torch.Tensor,
):
    k = -max_effort / (no_load_speed - full_torque_speed)
    limit = k * (dofs_vel.abs() - full_torque_speed) + max_effort
    return limit.clip(min=0.0)


# ---------------------------------------------------------------------------
# Composable action modifiers
# ---------------------------------------------------------------------------


@ACTION_MODIFIER_REGISTRY.register()
class Compose(ActionModifier):
    """Chains multiple action modifiers in sequence.

    Parameters
    ----------
    modifiers : tuple[ActionModifierOptions, ...] | list[ActionModifierOptions]
        Ordered sequence of modifier configurations. Each modifier's
        ``modify_processed_action`` and ``modify_ctrl_torque`` are called
        in sequence.

    Example
    -------
    ::

        Compose.configure(
            modifiers=[
                ActionDelay.configure(min_delay=1, max_delay=3),
                FrictionModel.configure(),
                EffortClip.configure(),
            ]
        )
    """

    modifiers: tuple[ActionModifierOptions, ...] | list[ActionModifierOptions] = ()

    def __init__(self, env: EnvBase, options: ActionModifierOptions):
        super().__init__(env=env, options=options)
        self._modifiers: list[ActionModifier] = []
        for mod_options in self.modifiers:
            # Normalize plain dicts (e.g. from YAML/JSON config loading) to ActionModifierOptions.
            if isinstance(mod_options, dict):
                mod_options = ActionModifierOptions(**mod_options)
            mod = ACTION_MODIFIER_REGISTRY.get(mod_options.name)(env=env, options=mod_options)
            self._modifiers.append(mod)

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        for mod in self._modifiers:
            mod.build(num_envs, device, entity=entity, dofs_idx_local=dofs_idx_local)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        for mod in self._modifiers:
            mod.reset(envs_idx)

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        for mod in self._modifiers:
            processed_action = mod.modify_processed_action(processed_action)
        return processed_action

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for mod in self._modifiers:
            ctrl_torque = mod.modify_ctrl_torque(ctrl_torque, dofs_vel, pos_err=pos_err)
        return ctrl_torque

    def get(self, cls: type) -> ActionModifier | None:
        """Return the first child modifier that is an instance of ``cls``, or ``None``."""
        for mod in self._modifiers:
            if isinstance(mod, cls):
                return mod
        return None


@ACTION_MODIFIER_REGISTRY.register()
class ActionDelay(ActionModifier):
    """Delays processed actions by a stochastic number of physics steps.

    At each reset, a new random delay is sampled uniformly from
    ``[min_delay, max_delay]`` for the reset environments.

    Parameters
    ----------
    min_delay : int
        Minimum command delay in physics steps (inclusive).
    max_delay : int
        Maximum command delay in physics steps (inclusive).
    """

    min_delay: int = 0
    max_delay: int = 0

    def __init__(self, env: EnvBase, options: ActionModifierOptions):
        super().__init__(env=env, options=options)
        self._delay_buffer = None

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if self.max_delay > 0:
            from eden.utils.buffers.delay_buffer import DelayBuffer

            self._delay_buffer = DelayBuffer(
                min_lag=self.min_delay,
                max_lag=self.max_delay,
                batch_size=num_envs,
                device=device,
            )

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        if self._delay_buffer is not None:
            self._delay_buffer.reset(envs_idx)

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        if self._delay_buffer is not None:
            self._delay_buffer.append(processed_action)
            return self._delay_buffer.compute()
        return processed_action


@ACTION_MODIFIER_REGISTRY.register()
class FrictionModel(ActionModifier):
    """Applies tanh-based static + viscous dynamic friction to torques.

    By default, friction parameters are read from the entity's actuator
    spec (``dofs_spec``).  Explicit values override the entity defaults.

    Parameters
    ----------
    static_friction : LaxFArrayType | None
        Static friction coefficient per DOF, or a scalar broadcast.
        *None* → read from entity spec.
    dynamic_friction : LaxFArrayType | None
        Dynamic (viscous) friction coefficient. *None* → read from entity spec.
    friction_activation_vel : LaxFArrayType | None
        Velocity at which static friction is fully activated [rad/s].
        *None* → read from entity spec.
    friction_offset : LaxFArrayType
        Constant friction offset torque per DOF (or scalar broadcast).
    """

    static_friction: LaxFArrayType | None = None
    dynamic_friction: LaxFArrayType | None = None
    friction_activation_vel: LaxFArrayType | None = None
    friction_offset: LaxFArrayType = 0.0

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        self._static_friction = _get_param_tensor_from_entity(
            self.static_friction,
            spec_attr="static_friction",
            entity=entity,
            dofs_idx_local=dofs_idx_local,
            device=device,
        )
        self._dynamic_friction = _get_param_tensor_from_entity(
            self.dynamic_friction,
            spec_attr="dynamic_friction",
            entity=entity,
            dofs_idx_local=dofs_idx_local,
            device=device,
        )
        self._friction_activation_vel = _get_param_tensor_from_entity(
            self.friction_activation_vel,
            spec_attr="friction_activation_speed",
            entity=entity,
            dofs_idx_local=dofs_idx_local,
            device=device,
        )
        self._friction_offset_tensor = _explicit_lax_to_tensor(
            self.friction_offset,
            param_name="friction_offset",
            dofs_idx_local=dofs_idx_local,
            device=device,
        )

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _tanh_friction(
            ctrl_torque,
            dofs_vel,
            self._static_friction,
            self._dynamic_friction,
            self._friction_activation_vel,
            self._friction_offset_tensor,
        )


@ACTION_MODIFIER_REGISTRY.register()
class EffortClip(ActionModifier):
    """Clips torques to motor torque-speed (T-N) curve limits.

    By default, T-N curve parameters are read from the entity's actuator
    spec (``dofs_spec``).  Explicit values override the entity defaults.

    Parameters
    ----------
    driving_torque_limit : LaxFArrayType | None
        Maximum torque when torque and velocity are in the same direction.
        *None* → read from entity spec.
    braking_torque_limit : LaxFArrayType | None
        Maximum torque when torque and velocity are in opposite directions.
        *None* → read from entity spec.
    full_torque_speed : LaxFArrayType | None
        Speed below which full torque is available [rad/s].
        *None* → read from entity spec.
    no_load_speed : LaxFArrayType | None
        Speed at which available torque drops to zero [rad/s].
        *None* → read from entity spec.
    """

    driving_torque_limit: LaxFArrayType | None = None
    braking_torque_limit: LaxFArrayType | None = None
    full_torque_speed: LaxFArrayType | None = None
    no_load_speed: LaxFArrayType | None = None

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        def _resolve(param_value, spec_attr):
            return _get_param_tensor_from_entity(
                param_value,
                spec_attr=spec_attr,
                entity=entity,
                dofs_idx_local=dofs_idx_local,
                device=device,
            )

        self._driving_torque_limit = _resolve(self.driving_torque_limit, "driving_torque_limit")
        self._braking_torque_limit = _resolve(self.braking_torque_limit, "braking_torque_limit")
        self._full_torque_speed = _resolve(self.full_torque_speed, "full_torque_speed")
        self._no_load_speed = _resolve(self.no_load_speed, "no_load_speed")

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _clip_effort(
            ctrl_torque,
            dofs_vel,
            self._driving_torque_limit,
            self._braking_torque_limit,
            self._full_torque_speed,
            self._no_load_speed,
        )


@ACTION_MODIFIER_REGISTRY.register()
class MotorStrength(ActionModifier):
    """Scales control torques by a motor strength multiplier.

    Parameters
    ----------
    motor_strength : LaxFArrayType
        Per-DOF multiplier, or a scalar broadcast to every controlled DOF.

    Note
    ----
    This modifier provides a deterministic motor strength multiplier.
    If you want to randomize the motor strength, you can use the :class:`RandomizeMotorStrength` event term instead of this modifier.
    """

    motor_strength: LaxFArrayType = 1.0

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if dofs_idx_local is None:
            raise ValueError("MotorStrength requires dofs_idx_local to be passed in build().")
        n_dof = len(dofs_idx_local)
        self._motor_strength_row = _per_dof_float_row(self.motor_strength, n_dof, device, param_name="motor_strength")
        self._motor_strength = self._motor_strength_row.unsqueeze(0).repeat(num_envs, 1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._motor_strength[envs_idx] = self._motor_strength_row

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return ctrl_torque * self._motor_strength


@ACTION_MODIFIER_REGISTRY.register()
class TorqueOffset(ActionModifier):
    """Adds a constant offset to the control torques.

    Parameters
    ----------
    torque_offset : LaxFArrayType
        Per-DOF offset, or a scalar broadcast to every controlled DOF.
    """

    torque_offset: LaxFArrayType = 0.0

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if dofs_idx_local is None:
            raise ValueError("TorqueOffset requires dofs_idx_local to be passed in build().")
        n_dof = len(dofs_idx_local)
        self._torque_offset_row = _per_dof_float_row(self.torque_offset, n_dof, device, param_name="torque_offset")
        self._torque_offset = self._torque_offset_row.unsqueeze(0).repeat(num_envs, 1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._torque_offset[envs_idx] = self._torque_offset_row

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return ctrl_torque + self._torque_offset


@ACTION_MODIFIER_REGISTRY.register()
class ConstantTorqueKick(ActionModifier):
    """Adds a flat directional torque on top of computed control torque.

    This acts like a minimum drive effort: when the joint has a nonzero position
    error, the modifier adds ``torque_kick`` in the direction of that error. It
    is useful for matching actuators that move sharply through small errors or
    have firmware/static-friction compensation that a pure proportional term
    cannot reproduce.

    Parameters
    ----------
    torque_kick : LaxFArrayType
        Nonnegative torque magnitude per DOF, or a scalar broadcast.
    activation_epsilon : LaxFArrayType
        Position-error magnitude (rad) below which no torque kick is added.
        This only applies when ``pos_err`` is available.
    direction_source : str
        Direction signal for the torque kick. ``"pos_err"`` pushes toward the
        target when a position error is supplied and falls back to
        ``"ctrl_torque"`` without one. ``"ctrl_torque"`` follows the sign of the
        already computed torque.
    """

    torque_kick: LaxFArrayType = 0.0
    activation_epsilon: LaxFArrayType = 0.0
    direction_source: str = "pos_err"

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if dofs_idx_local is None:
            raise ValueError("ConstantTorqueKick requires dofs_idx_local to be passed in build().")
        if self.direction_source not in {"pos_err", "ctrl_torque"}:
            raise ValueError("ConstantTorqueKick direction_source must be 'pos_err' or 'ctrl_torque'.")
        n_dof = len(dofs_idx_local)
        torque_kick_row = _per_dof_float_row(self.torque_kick, n_dof, device, param_name="torque_kick")
        if (torque_kick_row < 0.0).any().item():
            raise ValueError("ConstantTorqueKick torque_kick must be nonnegative.")
        self._torque_kick_row = torque_kick_row
        self._activation_epsilon_row = _per_dof_float_row(
            self.activation_epsilon,
            n_dof,
            device,
            param_name="activation_epsilon",
        ).abs()
        self._torque_kick = self._torque_kick_row.unsqueeze(0).repeat(num_envs, 1)
        self._activation_epsilon = self._activation_epsilon_row.unsqueeze(0).repeat(num_envs, 1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._torque_kick[envs_idx] = self._torque_kick_row
        self._activation_epsilon[envs_idx] = self._activation_epsilon_row

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.direction_source == "pos_err" and pos_err is not None:
            direction_signal = pos_err
            active = pos_err.abs() > self._activation_epsilon
        else:
            direction_signal = ctrl_torque
            active = direction_signal != 0.0
        return ctrl_torque + torch.sign(direction_signal) * self._torque_kick * active.to(ctrl_torque.dtype)


@ACTION_MODIFIER_REGISTRY.register()
class Deadband(ActionModifier):
    """Zeroes PD control torque inside a position-error deadband.

    No torque is applied when the instantaneous position error ``|pos_err|``
    is smaller than ``deadband_epsilon``. Only applicable to PD-position controllers that pass
    ``pos_err`` into :meth:`modify_ctrl_torque` — velocity-only controllers do
    not and the modifier is a no-op there.

    The threshold is stored as a per-env per-DOF tensor so it can be
    domain-randomized (see :class:`~eden.managers.terms.events.domain_rand.RandomizeDeadbandEpsilon`).

    Parameters
    ----------
    deadband_epsilon : LaxFArrayType
        Position-error magnitude (rad) below which torque is zeroed. Per
        :class:`~genesis.typing.LaxFArrayType`, a scalar broadcasts; an
        iterable must match ``len(dofs_idx_local)``.
    """

    deadband_epsilon: LaxFArrayType = 0.0

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if dofs_idx_local is None:
            raise ValueError("Deadband requires dofs_idx_local to be passed in build().")
        n_dof = len(dofs_idx_local)
        row = _per_dof_float_row(self.deadband_epsilon, n_dof, device, param_name="deadband_epsilon")
        self._deadband_epsilon_row = row
        self._deadband_epsilon = row.unsqueeze(0).repeat(num_envs, 1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._deadband_epsilon[envs_idx] = self._deadband_epsilon_row

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_err is None:
            return ctrl_torque
        return ctrl_torque.masked_fill(pos_err.abs() < self._deadband_epsilon, 0.0)


@ACTION_MODIFIER_REGISTRY.register()
class GearBacklash(ActionModifier):
    """Adds direction-dependent gear slop to position targets before PD control.

    This models lost motion in the drivetrain, where the load settles on the
    side of the gear gap selected by the latest commanded motion. After a
    positive target move the effective motor-side target becomes
    ``target + backlash``; after a negative move it becomes
    ``target - backlash``. Small target changes below ``reversal_threshold`` do
    not change the selected side, which prevents command noise from flipping the
    gap state.

    Parameters
    ----------
    backlash : LaxFArrayType
        One-sided position offset in radians. A scalar broadcasts to all
        controlled DOFs; an iterable must match ``len(dofs_idx_local)``.
    reversal_threshold : LaxFArrayType
        Minimum target-position change in radians required to select a new
        backlash side.
    takeup_rate : LaxFArrayType
        Fraction of the remaining gap state to take up on each action update.
        ``1.0`` switches sides instantly; values in ``[0, 1)`` create a smooth
        slosh transition over multiple control updates.
    initial_side : LaxFArrayType
        Initial gap side on reset. ``0.0`` starts centered until the first
        command movement selects a side.
    """

    backlash: LaxFArrayType = 0.0
    reversal_threshold: LaxFArrayType = 0.0
    takeup_rate: LaxFArrayType = 1.0
    initial_side: LaxFArrayType = 0.0

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if dofs_idx_local is None:
            raise ValueError("GearBacklash requires dofs_idx_local to be passed in build().")
        n_dof = len(dofs_idx_local)
        self._backlash_row = _per_dof_float_row(self.backlash, n_dof, device, param_name="backlash")
        if (self._backlash_row < 0.0).any().item():
            raise ValueError("GearBacklash backlash must be nonnegative.")
        self._reversal_threshold_row = _per_dof_float_row(
            self.reversal_threshold,
            n_dof,
            device,
            param_name="reversal_threshold",
        ).abs()
        self._takeup_rate_row = _per_dof_float_row(self.takeup_rate, n_dof, device, param_name="takeup_rate")
        if ((self._takeup_rate_row < 0.0) | (self._takeup_rate_row > 1.0)).any().item():
            raise ValueError("GearBacklash takeup_rate must be in [0, 1].")
        self._initial_side_row = _per_dof_float_row(self.initial_side, n_dof, device, param_name="initial_side").clamp(
            min=-1.0,
            max=1.0,
        )

        self._backlash = self._backlash_row.unsqueeze(0).repeat(num_envs, 1)
        self._reversal_threshold = self._reversal_threshold_row.unsqueeze(0).repeat(num_envs, 1)
        self._takeup_rate = self._takeup_rate_row.unsqueeze(0).repeat(num_envs, 1)
        self._initial_side = self._initial_side_row.unsqueeze(0).repeat(num_envs, 1)
        self._backlash_side = self._initial_side.clone()
        self._prev_target = torch.zeros(num_envs, n_dof, device=device)
        self._initialized = torch.zeros(num_envs, n_dof, dtype=torch.bool, device=device)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._backlash[envs_idx] = self._backlash_row
        self._reversal_threshold[envs_idx] = self._reversal_threshold_row
        self._takeup_rate[envs_idx] = self._takeup_rate_row
        self._initial_side[envs_idx] = self._initial_side_row
        self._backlash_side[envs_idx] = self._initial_side[envs_idx]
        self._prev_target[envs_idx] = 0.0
        self._initialized[envs_idx] = False

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        first_update = ~self._initialized
        delta = processed_action - self._prev_target
        moving_positive = delta > self._reversal_threshold
        moving_negative = delta < -self._reversal_threshold
        requested_side = torch.where(
            moving_positive,
            torch.ones_like(self._backlash_side),
            torch.where(moving_negative, -torch.ones_like(self._backlash_side), self._backlash_side),
        )
        requested_side = torch.where(first_update, self._initial_side, requested_side)
        self._backlash_side += (requested_side - self._backlash_side) * self._takeup_rate
        self._prev_target[:] = processed_action
        self._initialized[:] = True
        return processed_action + self._backlash_side * self._backlash


@ACTION_MODIFIER_REGISTRY.register()
class EnvelopeClip(ActionModifier):
    r"""Clips target positions so PD torques stay within the motor torque-speed envelope.

    Back-solves the motor torque-speed curve to find the range of target positions
    that yield feasible torques, then clips targets into that range. This ensures
    the PD controller never commands torques outside the motor's physical capability.

    The torque-speed envelope is defined by four parameters::

            Torque, N·m
                ^
        Y2──────|
                |──────────────Y1
                |              │\\
                |              │ \\
                |              │  \\
                |              |   \\
        --------+--------------|------> velocity, rad/s
                              X1   X2

    Given current joint state (q, dq) and PD gains (kp, kd), the modifier
    computes the maximum and minimum feasible torques from the T-N curve,
    then back-solves for the corresponding target position bounds::

        target_low  = (tau_low  + kd * dq) / kp + q
        target_high = (tau_high + kd * dq) / kp + q

    By default, T-N curve parameters are read from the entity's actuator
    spec (``dofs_spec``).  Explicit values override the entity defaults.

    Parameters
    ----------
    driving_torque_limit : LaxFArrayType | None
        (Y1) Maximum torque when torque and velocity are in the same direction.
        *None* → read from entity spec.
    braking_torque_limit : LaxFArrayType | None
        (Y2) Maximum torque when torque and velocity are in opposite directions.
        *None* → read from entity spec.
    full_torque_speed : LaxFArrayType | None
        (X1) Speed below which full torque is available [rad/s].
        *None* → read from entity spec.
    no_load_speed : LaxFArrayType | None
        (X2) Speed at which available torque drops to zero [rad/s].
        *None* → read from entity spec.
    """

    driving_torque_limit: LaxFArrayType | None = None
    braking_torque_limit: LaxFArrayType | None = None
    full_torque_speed: LaxFArrayType | None = None
    no_load_speed: LaxFArrayType | None = None

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if entity is None:
            raise ValueError("EnvelopeClip requires entity to be passed in build().")
        self._entity = entity
        self._dofs_idx_local = dofs_idx_local

        # Cache PD gains
        self._kp = entity.get_dofs_kp(dofs_idx_local=dofs_idx_local)
        self._kd = entity.get_dofs_kd(dofs_idx_local=dofs_idx_local)

        # T-N curve parameters — read from entity spec unless explicitly overridden
        def _resolve(param_value, spec_attr):
            return _get_param_tensor_from_entity(
                param_value,
                spec_attr=spec_attr,
                entity=entity,
                dofs_idx_local=dofs_idx_local,
                device=device,
            )

        self._motor_y1 = _resolve(self.driving_torque_limit, "driving_torque_limit")
        self._motor_y2 = _resolve(self.braking_torque_limit, "braking_torque_limit")
        self._motor_x1 = _resolve(self.full_torque_speed, "full_torque_speed")
        self._motor_x2 = _resolve(self.no_load_speed, "no_load_speed")

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._kp[envs_idx] = self._entity.get_dofs_kp(dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx)
        self._kd[envs_idx] = self._entity.get_dofs_kd(dofs_idx_local=self._dofs_idx_local, envs_idx=envs_idx)

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        joint_pos = self._entity.get_dofs_pos(dofs_idx_local=self._dofs_idx_local)
        joint_vel = self._entity.get_dofs_vel(dofs_idx_local=self._dofs_idx_local)

        abs_dq = joint_vel.abs()
        over = torch.clamp(abs_dq - self._motor_x1, min=0.0)
        denom = torch.clamp(self._motor_x2 - self._motor_x1, min=CURVE_SPEED_DENOM_MIN)

        # Positive torque limit (tau_high >= 0)
        base_pos = torch.where(
            abs_dq <= MOTOR_STALL_VEL_THRESHOLD,
            self._motor_y2,
            torch.where(joint_vel >= 0.0, self._motor_y1, self._motor_y2),
        )
        tau_high = torch.clamp(base_pos - (base_pos / denom) * over, min=0.0)

        # Negative torque limit (tau_low <= 0)
        base_neg = torch.where(
            abs_dq <= MOTOR_STALL_VEL_THRESHOLD,
            -self._motor_y2,
            torch.where(joint_vel >= 0.0, -self._motor_y2, -self._motor_y1),
        )
        tau_low = torch.clamp(base_neg + (-base_neg / denom) * over, max=0.0)

        # Back-solve: tau = kp * (target - q) - kd * dq
        kd_dq = self._kd * joint_vel
        target_low = (tau_low + kd_dq) / self._kp + joint_pos
        target_high = (tau_high + kd_dq) / self._kp + joint_pos

        return torch.clip(processed_action, target_low, target_high)
