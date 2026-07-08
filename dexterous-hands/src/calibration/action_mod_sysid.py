"""Bridge PD ``Compose`` action modifiers to Eden sysid ``Parameter`` application.

``apply_parameters`` only writes entity solver fields. We monkeypatch
``eden.extensions.sysid.modifier._apply_one`` so synthetic ``property``
strings update tensors on ``Deadband``, ``GearBacklash``, ``ConstantTorqueKick``,
``MotorStrength``, ``EffortClip``, ``EnvelopeClip``, and ``FrictionModel`` (matching
``HAND_CONTROLLER`` in ``shared_terms.py``).

T-N curve scalars (``driving_torque_limit``, …) are written to both
``EffortClip`` and ``EnvelopeClip`` so they stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch
from eden.managers.modifiers.actions.actuators import (
    Compose,
    ConstantTorqueKick,
    Deadband,
    EffortClip,
    EnvelopeClip,
    FrictionModel,
    GearBacklash,
    MotorStrength,
)
from eden.managers.terms.actions.joint_actions import ExplicitPDController

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.extensions.sysid.parameter import Parameter

JOINT_POS_TERM = "dofs_pos_controller"


def get_dofs_pos_controller(env: EnvBase) -> ExplicitPDController:
    terms = env.action_manager._terms
    if JOINT_POS_TERM not in terms:
        raise KeyError(f"Expected action term {JOINT_POS_TERM!r} on env.action_manager; got {list(terms)!r}.")
    term = terms[JOINT_POS_TERM]
    if not isinstance(term, ExplicitPDController):
        raise TypeError(f"Expected ExplicitPDController for {JOINT_POS_TERM}, got {type(term)}.")
    return term


def resolve_compose(term: ExplicitPDController) -> Compose:
    mod = term._modifier
    if isinstance(mod, Compose):
        return mod
    raise TypeError(f"Expected Compose modifier stack on PD term (HAND_CONTROLLER); got {type(mod).__name__}.")


def resolve_deadband(term: ExplicitPDController) -> Deadband:
    inner = resolve_compose(term).get(Deadband)
    if inner is None:
        raise RuntimeError("Compose stack has no Deadband.")
    return inner


def resolve_gear_backlash(term: ExplicitPDController) -> GearBacklash:
    inner = resolve_compose(term).get(GearBacklash)
    if inner is None:
        raise RuntimeError("Compose stack has no GearBacklash.")
    return inner


def resolve_constant_torque_kick(term: ExplicitPDController) -> ConstantTorqueKick:
    inner = resolve_compose(term).get(ConstantTorqueKick)
    if inner is None:
        raise RuntimeError("Compose stack has no ConstantTorqueKick.")
    return inner


def _local_indices_for_param(term: ExplicitPDController, dof_names: Sequence[str]) -> list[int]:
    order = term.dofs_order
    out: list[int] = []
    for name in dof_names:
        if name not in order:
            raise KeyError(f"DOF {name!r} not in controller dofs_order {list(order)!r}.")
        out.append(order[name])
    return out


def _values_for_columns(
    env: EnvBase,
    param: Parameter,
    local_idx: torch.Tensor,
    per_env_values: np.ndarray | None,
) -> torch.Tensor:
    """``(num_envs, len(local_idx))`` tensor from ``param`` / parallel candidates."""
    if per_env_values is not None:
        K, width = per_env_values.shape
        if K != env.num_envs:
            raise ValueError(f"per_env_values rows {K} != env.num_envs {env.num_envs}.")
        n_dofs = len(param.dof_names)
        if param.per_dof:
            if width != n_dofs:
                raise ValueError(f"per_env_values width {width} != len(dof_names) {n_dofs}.")
            mat = per_env_values.astype(np.float64, copy=False)
        else:
            if width != 1:
                raise ValueError(f"shared param expects per_env_values width 1, got {width}.")
            mat = np.broadcast_to(per_env_values, (K, n_dofs)).copy()
        return torch.as_tensor(mat, dtype=torch.float32, device=env.device)
    values = param.as_dof_vector(len(param.dof_names))
    v = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    return v.unsqueeze(0).expand(env.num_envs, -1)


def _assign_modifier_slice(
    modifier: object,
    attr: str,
    env: EnvBase,
    param: Parameter,
    term: ExplicitPDController,
    per_env_values: np.ndarray | None,
    row_attr: str | None = None,
) -> None:
    """Write ``param`` into ``modifier.{attr}`` columns for controlled DOFs (subset ``param.dof_names``)."""
    cur = getattr(modifier, attr)
    if cur.ndim != 2:
        raise ValueError(f"{attr} expected 2-D tensor, got shape {tuple(cur.shape)}.")
    li = torch.as_tensor(_local_indices_for_param(term, param.dof_names), dtype=torch.long, device=env.device)
    vals = _values_for_columns(env, param, li, per_env_values)
    n_ctrl = cur.shape[-1]
    base = cur[0].unsqueeze(0).expand(env.num_envs, n_ctrl).clone()
    base[:, li] = vals
    setattr(modifier, attr, base)
    if row_attr is not None and per_env_values is None and hasattr(modifier, row_attr):
        row = getattr(modifier, row_attr).clone()
        if row.ndim != 1:
            raise ValueError(f"{row_attr} expected 1-D tensor, got shape {tuple(row.shape)}.")
        row[li] = vals[0]
        setattr(modifier, row_attr, row)


def write_deadband_epsilon(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    deadband = resolve_deadband(term)
    _assign_modifier_slice(
        deadband,
        "_deadband_epsilon",
        env,
        param,
        term,
        per_env_values,
        row_attr="_deadband_epsilon_row",
    )


def write_gear_backlash(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    _assign_modifier_slice(gear, "_backlash", env, param, term, per_env_values, row_attr="_backlash_row")


def write_gear_reversal_threshold(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    _assign_modifier_slice(
        gear,
        "_reversal_threshold",
        env,
        param,
        term,
        per_env_values,
        row_attr="_reversal_threshold_row",
    )


def write_gear_takeup_rate(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    _assign_modifier_slice(gear, "_takeup_rate", env, param, term, per_env_values, row_attr="_takeup_rate_row")


def write_gear_initial_side(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    _assign_modifier_slice(gear, "_initial_side", env, param, term, per_env_values, row_attr="_initial_side_row")
    li = torch.as_tensor(_local_indices_for_param(term, param.dof_names), dtype=torch.long, device=env.device)
    gear._backlash_side[:, li] = gear._initial_side[:, li]
    gear._initialized[:, li] = False


def write_torque_kick(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    kick = resolve_constant_torque_kick(term)
    _assign_modifier_slice(kick, "_torque_kick", env, param, term, per_env_values, row_attr="_torque_kick_row")


def write_activation_epsilon(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    kick = resolve_constant_torque_kick(term)
    _assign_modifier_slice(
        kick,
        "_activation_epsilon",
        env,
        param,
        term,
        per_env_values,
        row_attr="_activation_epsilon_row",
    )


def write_motor_strength(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None = None,
) -> None:
    term = get_dofs_pos_controller(env)
    ms = resolve_compose(term).get(MotorStrength)
    if ms is None:
        raise RuntimeError("Compose stack has no MotorStrength.")
    li = torch.as_tensor(_local_indices_for_param(term, param.dof_names), dtype=torch.long, device=env.device)
    vals = _values_for_columns(env, param, li, per_env_values)
    ms._motor_strength[:, li] = vals


def _write_tn_pair(
    env: EnvBase,
    param: Parameter,
    *,
    per_env_values: np.ndarray | None,
    effort_attr: str,
    envelope_attr: str,
) -> None:
    term = get_dofs_pos_controller(env)
    compose = resolve_compose(term)
    ec = compose.get(EffortClip)
    ev = compose.get(EnvelopeClip)
    if ec is None or ev is None:
        raise RuntimeError("Compose stack needs EffortClip and EnvelopeClip for T-N sysid.")
    _assign_modifier_slice(ec, effort_attr, env, param, term, per_env_values)
    _assign_modifier_slice(ev, envelope_attr, env, param, term, per_env_values)


def write_driving_torque_limit(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    _write_tn_pair(
        env, param, per_env_values=per_env_values, effort_attr="_driving_torque_limit", envelope_attr="_motor_y1"
    )


def write_braking_torque_limit(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    _write_tn_pair(
        env, param, per_env_values=per_env_values, effort_attr="_braking_torque_limit", envelope_attr="_motor_y2"
    )


def write_full_torque_speed(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    _write_tn_pair(
        env, param, per_env_values=per_env_values, effort_attr="_full_torque_speed", envelope_attr="_motor_x1"
    )


def write_no_load_speed(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    _write_tn_pair(env, param, per_env_values=per_env_values, effort_attr="_no_load_speed", envelope_attr="_motor_x2")


def write_friction_static(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    _assign_modifier_slice(fm, "_static_friction", env, param, term, per_env_values)


def write_friction_dynamic(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    _assign_modifier_slice(fm, "_dynamic_friction", env, param, term, per_env_values)


def write_friction_activation_vel(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    _assign_modifier_slice(fm, "_friction_activation_vel", env, param, term, per_env_values)


def write_friction_offset(env: EnvBase, param: Parameter, *, per_env_values: np.ndarray | None = None) -> None:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    _assign_modifier_slice(fm, "_friction_offset_tensor", env, param, term, per_env_values)


_MODIFIER_WRITERS: dict[str, Callable[..., None]] = {
    "deadband_epsilon": write_deadband_epsilon,
    "gear_backlash": write_gear_backlash,
    "gear_reversal_threshold": write_gear_reversal_threshold,
    "gear_takeup_rate": write_gear_takeup_rate,
    "gear_initial_side": write_gear_initial_side,
    "torque_kick": write_torque_kick,
    "activation_epsilon": write_activation_epsilon,
    "motor_strength": write_motor_strength,
    "driving_torque_limit": write_driving_torque_limit,
    "braking_torque_limit": write_braking_torque_limit,
    "full_torque_speed": write_full_torque_speed,
    "no_load_speed": write_no_load_speed,
    "friction_static": write_friction_static,
    "friction_dynamic": write_friction_dynamic,
    "friction_activation_vel": write_friction_activation_vel,
    "friction_offset": write_friction_offset,
}


def _read_slice_row(
    modifier: object, attr: str, term: ExplicitPDController, dof_names: Sequence[str], env_row: int
) -> np.ndarray:
    cur = getattr(modifier, attr)
    li = _local_indices_for_param(term, dof_names)
    row = cur[int(env_row), li].detach().cpu().numpy().astype(np.float64)
    return row


def read_deadband_epsilon_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    db = resolve_deadband(term)
    return _read_slice_row(db, "_deadband_epsilon", term, dof_names, env_row)


def read_gear_backlash_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    return _read_slice_row(gear, "_backlash", term, dof_names, env_row)


def read_gear_reversal_threshold_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    return _read_slice_row(gear, "_reversal_threshold", term, dof_names, env_row)


def read_gear_takeup_rate_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    return _read_slice_row(gear, "_takeup_rate", term, dof_names, env_row)


def read_gear_initial_side_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    gear = resolve_gear_backlash(term)
    return _read_slice_row(gear, "_initial_side", term, dof_names, env_row)


def read_torque_kick_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    kick = resolve_constant_torque_kick(term)
    return _read_slice_row(kick, "_torque_kick", term, dof_names, env_row)


def read_activation_epsilon_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    kick = resolve_constant_torque_kick(term)
    return _read_slice_row(kick, "_activation_epsilon", term, dof_names, env_row)


def read_motor_strength_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    ms = resolve_compose(term).get(MotorStrength)
    if ms is None:
        raise RuntimeError("Compose stack has no MotorStrength.")
    return _read_slice_row(ms, "_motor_strength", term, dof_names, env_row)


def read_driving_torque_limit_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    ec = resolve_compose(term).get(EffortClip)
    if ec is None:
        raise RuntimeError("Compose stack has no EffortClip.")
    return _read_slice_row(ec, "_driving_torque_limit", term, dof_names, env_row)


def read_braking_torque_limit_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    ec = resolve_compose(term).get(EffortClip)
    if ec is None:
        raise RuntimeError("Compose stack has no EffortClip.")
    return _read_slice_row(ec, "_braking_torque_limit", term, dof_names, env_row)


def read_full_torque_speed_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    ec = resolve_compose(term).get(EffortClip)
    if ec is None:
        raise RuntimeError("Compose stack has no EffortClip.")
    return _read_slice_row(ec, "_full_torque_speed", term, dof_names, env_row)


def read_no_load_speed_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    ec = resolve_compose(term).get(EffortClip)
    if ec is None:
        raise RuntimeError("Compose stack has no EffortClip.")
    return _read_slice_row(ec, "_no_load_speed", term, dof_names, env_row)


def read_friction_static_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    return _read_slice_row(fm, "_static_friction", term, dof_names, env_row)


def read_friction_dynamic_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    return _read_slice_row(fm, "_dynamic_friction", term, dof_names, env_row)


def read_friction_activation_vel_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    return _read_slice_row(fm, "_friction_activation_vel", term, dof_names, env_row)


def read_friction_offset_row(env: EnvBase, dof_names: Sequence[str], env_row: int = 0) -> np.ndarray:
    term = get_dofs_pos_controller(env)
    fm = resolve_compose(term).get(FrictionModel)
    if fm is None:
        raise RuntimeError("Compose stack has no FrictionModel.")
    return _read_slice_row(fm, "_friction_offset_tensor", term, dof_names, env_row)


@dataclass(frozen=True)
class TunableProperty:
    """Single source of truth for a sysid-tunable per-DOF property.

    Used by ``identify._build_parameters`` to construct nominals, by the
    sysid optimiser (via ``PROPERTIES[name].bounds``) to pick search ranges,
    and by the manual calibration GUI to render sliders and route reads/
    writes through one dispatch table.

    ``read_row`` returns the entity's current per-DOF value vector for a
    list of joint names (env-0 only — sysid sims are num_envs ∈ {1, batch}
    where row 0 is the canonical nominal). Writes go through
    ``eden.extensions.sysid.modifier.apply_parameters``: for solver fields
    Eden's ``_apply_one`` writes to ``entity.set_dofs_<prop>`` directly;
    for modifier fields the monkeypatch installed by
    :func:`install_action_mod_sysid_patch` redirects to the per-property
    writer in ``_MODIFIER_WRITERS``.
    """

    name: str
    bounds: tuple[float, float]
    read_row: Callable[["EnvBase", Sequence[str]], np.ndarray]
    is_modifier: bool
    needs_pd_refresh: bool = False


def _read_solver_row(prop_name: str) -> Callable[["EnvBase", Sequence[str]], np.ndarray]:
    """Build a reader returning the entity's current per-DOF values for a solver field."""
    getter_name = f"get_dofs_{prop_name}"

    def reader(env: "EnvBase", dof_names: Sequence[str]) -> np.ndarray:
        entity = env.entities["robot"]
        _, local_idx = entity.find_named_dofs_idx_local(list(dof_names), preserve_order=True)
        vals = getattr(entity, getter_name)(dofs_idx_local=local_idx)
        if hasattr(vals, "ndim") and vals.ndim == 2:
            vals = vals[0]
        return vals.detach().cpu().numpy().astype(np.float64)

    return reader


def _row0(reader: Callable[..., np.ndarray]) -> Callable[["EnvBase", Sequence[str]], np.ndarray]:
    """Adapt a modifier ``read_*_row(env, dof_names, env_row=0)`` to TunableProperty's signature."""
    return lambda env, dof_names: reader(env, dof_names, env_row=0)


# fmt: off
PROPERTIES: dict[str, TunableProperty] = {
    # Solver fields (Eden's _apply_one writes via entity.set_dofs_<name>).
    "damping":      TunableProperty("damping",      (0.0, 20.0),  _read_solver_row("damping"),      is_modifier=False),
    "armature":     TunableProperty("armature",     (1e-4, 1.0),  _read_solver_row("armature"),     is_modifier=False),
    "frictionloss": TunableProperty("frictionloss", (1e-4, 2.0),  _read_solver_row("frictionloss"), is_modifier=False),
    "stiffness":    TunableProperty("stiffness",    (0.0, 20.0),  _read_solver_row("stiffness"),    is_modifier=False),
    "kp":           TunableProperty("kp",           (1.0, 200.0), _read_solver_row("kp"),           is_modifier=False, needs_pd_refresh=True),
    "kd":           TunableProperty("kd",           (0.1, 100.0), _read_solver_row("kd"),           is_modifier=False, needs_pd_refresh=True),
    # Action modifiers (routed through _MODIFIER_WRITERS by the monkeypatch).
    "deadband_epsilon":        TunableProperty("deadband_epsilon",        (0.0, 0.1),    _row0(read_deadband_epsilon_row),        is_modifier=True),
    "gear_backlash":           TunableProperty("gear_backlash",           (0.0, 0.1),    _row0(read_gear_backlash_row),           is_modifier=True),
    "gear_reversal_threshold": TunableProperty("gear_reversal_threshold", (0.0, 0.2),    _row0(read_gear_reversal_threshold_row), is_modifier=True),
    "gear_takeup_rate":        TunableProperty("gear_takeup_rate",        (0.0, 1.0),    _row0(read_gear_takeup_rate_row),        is_modifier=True),
    "gear_initial_side":       TunableProperty("gear_initial_side",       (-1.0, 1.0),   _row0(read_gear_initial_side_row),       is_modifier=True),
    "torque_kick":             TunableProperty("torque_kick",             (0.0, 20.0),   _row0(read_torque_kick_row),             is_modifier=True),
    "activation_epsilon":      TunableProperty("activation_epsilon",      (0.0, 0.2),    _row0(read_activation_epsilon_row),      is_modifier=True),
    "motor_strength":          TunableProperty("motor_strength",          (0.5, 2.0),    _row0(read_motor_strength_row),          is_modifier=True),
    "driving_torque_limit":    TunableProperty("driving_torque_limit",    (0.1, 200.0),  _row0(read_driving_torque_limit_row),    is_modifier=True),
    "braking_torque_limit":    TunableProperty("braking_torque_limit",    (0.1, 200.0),  _row0(read_braking_torque_limit_row),    is_modifier=True),
    "full_torque_speed":       TunableProperty("full_torque_speed",       (0.01, 100.0), _row0(read_full_torque_speed_row),       is_modifier=True),
    "no_load_speed":           TunableProperty("no_load_speed",           (1.0, 500.0),  _row0(read_no_load_speed_row),           is_modifier=True),
    "friction_static":         TunableProperty("friction_static",         (0.0, 10.0),   _row0(read_friction_static_row),         is_modifier=True),
    "friction_dynamic":        TunableProperty("friction_dynamic",        (0.0, 10.0),   _row0(read_friction_dynamic_row),        is_modifier=True),
    "friction_activation_vel": TunableProperty("friction_activation_vel", (1e-3, 50.0),  _row0(read_friction_activation_vel_row), is_modifier=True),
    "friction_offset":         TunableProperty("friction_offset",         (-5.0, 5.0),   _row0(read_friction_offset_row),         is_modifier=True),
}
# fmt: on

BOUNDS: dict[str, tuple[float, float]] = {name: p.bounds for name, p in PROPERTIES.items()}
_MODIFIER_PARAMETER_PROPERTIES: frozenset[str] = frozenset(name for name, p in PROPERTIES.items() if p.is_modifier)


def install_action_mod_sysid_patch() -> None:
    """Monkeypatch Eden sysid ``_apply_one`` once per process."""
    from eden.extensions.sysid import modifier as m

    if getattr(m, "_modifier_sysid_patched", False):
        return

    _orig = m._apply_one

    def _apply_one_with_modifiers(
        env: EnvBase,
        param: Parameter,
        envs_idx: torch.Tensor | Sequence[int] | None = None,
        per_env_values: np.ndarray | None = None,
    ) -> None:
        prop = param.property
        if prop in _MODIFIER_PARAMETER_PROPERTIES:
            if envs_idx is not None:
                raise NotImplementedError(f"{prop} apply with envs_idx is not supported.")
            writer = _MODIFIER_WRITERS[str(prop)]
            writer(env, param, per_env_values=per_env_values)
            return
        return _orig(env, param, envs_idx, per_env_values)

    m._apply_one = _apply_one_with_modifiers
    m._modifier_sysid_patched = True
