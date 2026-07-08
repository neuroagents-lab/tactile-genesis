"""Sim-twin config for dexterous-hand system identification.

A minimal ``EdenRLConfig`` that spawns a single fixed-base dexterous hand
controlled by an ``ExplicitPDController``. No rewards, no commands, no
termination logic: the config is only used as a rollout substrate for
``eden.extensions.sysid``. Explicit (rather than implicit) PD avoids the
``kd * substep_dt`` armature correction Genesis applies under implicit
damping, which would otherwise inject ~100× phantom inertia and bias
the identification.

The config is robot-agnostic: :func:`make_sim_twin_config` takes any hand
registered in ``ROBOT_REGISTRY`` (``xhand1``, ``shadow``, ``allegro``,
``sharpa``, …). Only the *real* hardware side (``collect_data.py``'s
deployment) is XHand1-specific today; the sim twin works for any hand.

The PD path includes ``Deadband``, ``GearBacklash``, and ``ConstantTorqueKick``
(zero initial effect) so modifier parameters in ``identify.py`` can match real
drivetrain slop and minimum-drive behavior.

The ``num_envs`` field is exposed so the identification script can spin
up a single env for serial fits (scipy least-squares, CMA-ES serial) or
an env sized to the CMA-ES population for batched evaluation.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Sequence, cast

import eden as en
import genesis.utils.geom as gu
import numpy as np
import yaml
from eden.extensions.sysid import Parameter
from eden.extensions.sysid import ParameterSet as _EdenParameterSet
from eden.managers.terms.recorders.sysid import SysIDRecorder
from eden.options import ObservationGroupOptions
from eden.options.managers.recorders import RecorderManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.terms import ObsTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    ObservationManagerOptions,
    SceneOptions,
)

import entities.robots  # noqa: F401  -- registers hands into ROBOT_REGISTRY

from calibration.action_mod_sysid import PROPERTIES
from registry import ROBOT_REGISTRY
from shared_terms import ACTION_MODIFIERS, DECIMATION, SIM_DT


class ParameterSet(_EdenParameterSet):
    """Slim-YAML ``ParameterSet`` for dexterous-hand sysid artefacts.

    Bounds belong to the optimisation setup (``action_mod_sysid.PROPERTIES``),
    not to the fitted result — persisting them in ``params_identified.yaml``
    would just freeze stale limits next to the artefacts. This subclass omits
    ``nominal`` / ``min_value`` / ``max_value`` on save and fills them with
    ``value`` on load so the constructed ``Parameter`` objects remain
    internally consistent.
    """

    def save_yaml(self, path: str | pathlib.Path) -> None:
        payload = {}
        for p in self:
            state = p.__getstate__()
            del state["nominal"]
            del state["min_value"]
            del state["max_value"]
            payload[p.name] = state
        pathlib.Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

    @classmethod
    def load_yaml(cls, path: str | pathlib.Path) -> "ParameterSet":
        payload = yaml.safe_load(pathlib.Path(path).read_text())
        params: list[Parameter] = []
        for name, state in payload.items():
            value = state["value"]
            state["nominal"] = value
            state["min_value"] = value
            state["max_value"] = value
            p = Parameter.__new__(Parameter)
            p.__setstate__(state)
            if p.name != name:
                p.name = name
            params.append(p)
        return cls(params)


def make_parameter(name: str, dof_names: Sequence[str], values: np.ndarray) -> Parameter:
    """Build a ``Parameter`` with value-only semantics (bounds collapsed to value).

    Used by the manual GUI's apply path and the slim YAML save path — neither
    cares about optimization bounds. ``apply_parameters`` only reads ``.value``;
    ``ParameterSet.save_yaml`` strips the bounds before persisting.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    p = Parameter(
        name=name,
        property=cast(Any, name),
        dof_names=tuple(dof_names),
        entity_name="robot",
        per_dof=True,
        nominal=arr,
        min_value=arr.copy(),
        max_value=arr.copy(),
    )
    p.value = arr.copy()
    return p


def save_params_yaml(path: pathlib.Path, dof_names: Sequence[str], values_by_name: dict[str, np.ndarray]) -> None:
    """Save the GUI's per-property slider values to a slim ParameterSet YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = ParameterSet([make_parameter(name, dof_names, vals) for name, vals in values_by_name.items()])
    params.save_yaml(path)


def load_params_yaml(path: pathlib.Path, dof_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Load a slim ParameterSet YAML and return ``{property_name: per_dof_values}``.

    Values are remapped onto ``dof_names`` order; the YAML may have a different
    joint order or be a strict superset. Missing joints raise ``ValueError`` —
    silently slotting wrong joint values is the failure mode this guards.
    """
    params = ParameterSet.load_yaml(path)
    out: dict[str, np.ndarray] = {}
    for param in params:
        name = param.name if param.name in PROPERTIES else str(param.property)
        if name not in PROPERTIES:
            continue
        values = param.as_dof_vector(len(param.dof_names))
        by_name = {dn: float(v) for dn, v in zip(param.dof_names, values, strict=True)}
        missing = [dn for dn in dof_names if dn not in by_name]
        if missing:
            raise ValueError(f"parameter {name!r} in {path} is missing DOFs: {missing}")
        out[name] = np.asarray([by_name[dn] for dn in dof_names], dtype=np.float64)
    if not out:
        raise ValueError(f"{path} did not contain any properties listed in PROPERTIES.")
    return out


def make_sim_twin_config(
    robot: str = "xhand1",
    *,
    num_envs: int = 1,
    sim_dt: float = SIM_DT,
    decimation: int = DECIMATION,
) -> EdenRLConfig:
    """Build the sysid sim-twin ``EdenRLConfig`` for any registered hand.

    ``robot`` is a key in ``ROBOT_REGISTRY`` (``xhand1``, ``shadow``,
    ``allegro``, ``sharpa``, …). The hand is spawned fixed-base, raised off
    the floor and rotated palm-down, with all joints at 0 and gravity
    compensation disabled so the identified dynamics aren't masked. The
    ``ExplicitPDController`` acts on every DOF (``dofs_name=["*"]``).

    ``num_envs`` is exposed so the identification script can spin up a
    single env for serial fits (scipy least-squares, CMA-ES serial) or an
    env sized to the CMA-ES population for batched evaluation.
    """
    base = ROBOT_REGISTRY.get(robot)()
    robot_cfg = base.model_copy(
        update=dict(
            is_fixed_base=True,
            default_root_pos=(0.0, 0.0, 0.5),  # raise the hand up to avoid collision with the floor
            default_root_quat=gu.euler_to_quat((0.0, -90.0, 0.0)),
            default_dofs_pos=dict.fromkeys(base.dofs_name, 0.0),
            material=RigidMaterialOptions(gravity_compensation=0.0),
        )
    )

    return EdenRLConfig(
        env_options=EnvOptions(
            num_envs=num_envs,
            num_eval_envs=num_envs,
            episode_length_s=9999.0,
            sim_dt=sim_dt,
            decimation=decimation,
            sim_substeps=1,
            enable_multi_contact=True,
        ),
        scene_options=SceneOptions(robot=robot_cfg),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                dofs_pos=ObsTerm.configure(func=en.observations.dofs_pos, params={"entity_name": "robot"}),
                enable_corruption=False,
            ),
        ),
        action_options=ActionManagerOptions(
            dofs_pos_controller=en.actions.ExplicitPDController.configure(
                entity_name="robot",
                dofs_name=["*"],
                scale=1.0,
                modifier=ACTION_MODIFIERS,
            ),
        ),
        recorder_options=RecorderManagerOptions(
            sysid=SysIDRecorder.configure(
                entity_name="robot",
                include_torque=True,
                include_base=False,  # fixed-base hand — no floating-base signals.
            ),
        ),
    )


def make_argparser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--robot",
        default="xhand1",
        choices=sorted(ROBOT_REGISTRY.keys()),
        help="Registered dexterous hand to build the sysid sim twin for.",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=SIM_DT,
        help="Sim physics timestep for the sysid sim twin.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=DECIMATION,
        help="Control decimation for the sysid sim twin. The replay step is sim_dt * decimation.",
    )
    return parser
