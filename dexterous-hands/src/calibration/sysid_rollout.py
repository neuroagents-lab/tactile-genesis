"""Shared sim-twin rollout for dexterous-hand calibration scripts.

``identify.py``, ``verify.py``, and ``manual_calibration_gui.py`` all replay
recorded action trajectories through the same sim twin. Centralising the
rollout here on ``RLEnvBase.step`` + ``RLEnvBase.reset`` — rather than the
narrow physics-only stepper in ``eden.extensions.sysid.rollout`` — keeps
three things in agreement:

1. The cost the optimiser converged to and the RMSE the verify script
   reports are produced by the same physics path. Without this, the basic
   "best cost reproduces on verification" sanity check is meaningless.
2. ``ExplicitPDController`` re-reads its cached kp/kd from the entity on
   ``env.reset()`` (via ``action_manager.reset`` → ``term.reset``), so
   kp/kd parameter candidates actually reach the controller. The narrow
   stepper bypasses ``_reset_idx`` entirely and silently uses stale gains.
3. The manual calibration GUI steps the same primitives, so what it plots
   matches what ``identify`` and ``verify`` produce on the same trace.

For interactive editing where a full reset would drop replay state, call
:func:`refresh_pd_gains` directly after writing to ``set_dofs_kp/kd``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from eden.extensions.sysid.modifier import apply_candidates, apply_parameters
from eden.extensions.sysid.parameter import ParameterSet
from eden.extensions.sysid.rollout import reset_to_initial_state
from eden.extensions.sysid.trajectory import Trajectory

from calibration.action_mod_sysid import get_dofs_pos_controller

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


_SIGNAL_DIMS: dict[str, int | str] = {
    "dofs_pos": "num_dofs",
    "dofs_vel": "num_dofs",
    "dofs_torque": "num_dofs",
    "base_quat": 4,
    "base_ang_vel": 3,
}


def refresh_pd_gains(env: "RLEnvBase") -> None:
    """Force the active ``ExplicitPDController`` to re-read kp/kd from the entity.

    ``ExplicitPDController`` caches kp/kd in ``build()`` and only refreshes
    that cache inside ``reset()``. Direct writes like ``robot.set_dofs_kp(...)``
    are therefore invisible to the controller until the next ``env.reset()``.
    The rollout below pays that cost; the manual GUI does not, so call this
    after every kp/kd write to make the change effective on the next step.
    """
    term = get_dofs_pos_controller(env)
    term._kp = term.entity.get_dofs_kp(dofs_idx_local=term.dofs_idx_local)
    term._kd = term.entity.get_dofs_kd(dofs_idx_local=term.dofs_idx_local)
    if not term._batch_dofs_info:
        term._kp = term._kp.unsqueeze(0).repeat(term.num_envs, 1)
        term._kd = term._kd.unsqueeze(0).repeat(term.num_envs, 1)


def rollout(
    env: "RLEnvBase",
    trajectory: Trajectory,
    entity_name: str = "robot",
    signals: Sequence[str] = ("dofs_pos",),
) -> dict[str, torch.Tensor]:
    """Reset env, place the recorded initial state, replay ``trajectory.action``.

    Callers should apply any candidate parameters first via
    :func:`apply_parameters` / :func:`apply_candidates` (or direct writes for
    the manual GUI). The ``env.reset()`` here is what lets the optimiser's
    kp/kd candidates actually reach ``ExplicitPDController`` — without it the
    cached gains stay at URDF defaults and the kp/kd parameter dimensions
    don't influence the residual.

    Safe for the sysid sim twin (``make_sim_twin_config``) because it has ``episode_length_s=9999``
    (no time-based auto-reset) and no event-manager terms (no ``_RESET``
    domain-randomisation firing on the very parameters being identified).
    Copy-pasting this onto a task config with either of those would silently
    corrupt the rollout.
    """
    if trajectory.action is None:
        raise ValueError("Trajectory.action is required for replay.")

    entity = env.entities[entity_name]
    n_steps = len(trajectory)
    buffers: dict[str, torch.Tensor] = {}
    for s in signals:
        if s not in _SIGNAL_DIMS:
            raise ValueError(f"Unsupported rollout signal: {s!r}. Expected one of {tuple(_SIGNAL_DIMS)}.")
        dim = _SIGNAL_DIMS[s]
        dim_val = entity.num_dofs if dim == "num_dofs" else dim
        buffers[s] = torch.zeros(env.num_envs, n_steps, dim_val, device=env.device)

    env.reset()
    reset_to_initial_state(env, trajectory, entity_name)

    actions = torch.as_tensor(trajectory.action, dtype=torch.float32, device=env.device)
    for t in range(n_steps):
        action_batch = actions[t].unsqueeze(0).expand(env.num_envs, -1).contiguous()
        env.step(action_batch)
        if "dofs_pos" in buffers:
            buffers["dofs_pos"][:, t] = entity.get_dofs_pos()
        if "dofs_vel" in buffers:
            buffers["dofs_vel"][:, t] = entity.get_dofs_vel()
        if "dofs_torque" in buffers:
            buffers["dofs_torque"][:, t] = entity.get_dofs_control_force()
        if "base_quat" in buffers:
            buffers["base_quat"][:, t] = entity.get_quat()
        if "base_ang_vel" in buffers:
            buffers["base_ang_vel"][:, t] = entity.get_ang(frame="body")
    return buffers


def single_candidate_rollout(
    env: "RLEnvBase",
    params: ParameterSet,
    trajectory: Trajectory,
    entity_name: str = "robot",
    signals: Sequence[str] = ("dofs_pos",),
) -> dict[str, np.ndarray]:
    """Apply ``params`` to all envs, replay once, return env-0 numpy arrays."""
    apply_parameters(env, params)
    buffers = rollout(env, trajectory, entity_name, signals)
    return {k: v[0].detach().cpu().numpy() for k, v in buffers.items()}


def batched_candidate_rollout(
    env: "RLEnvBase",
    params: ParameterSet,
    candidates: np.ndarray,
    trajectory: Trajectory,
    entity_name: str = "robot",
    signals: Sequence[str] = ("dofs_pos",),
) -> dict[str, np.ndarray]:
    """Write each candidate row into its own env, replay once, return per-env numpy."""
    K, n_free = candidates.shape
    if n_free != params.size:
        raise ValueError(f"Candidates width {n_free} != ParameterSet.size {params.size}.")
    if K != env.num_envs:
        raise ValueError(f"Batched rollout requires env.num_envs == K (got K={K}, num_envs={env.num_envs}).")
    apply_candidates(env, params, candidates)
    buffers = rollout(env, trajectory, entity_name, signals)
    return {k: v.detach().cpu().numpy() for k, v in buffers.items()}
