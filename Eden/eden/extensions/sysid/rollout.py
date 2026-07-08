"""Rollout helpers for replaying and evaluating system-id candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from eden.extensions.sysid.modifier import apply_candidates, apply_parameters
from eden.extensions.sysid.parameter import ParameterSet
from eden.extensions.sysid.trajectory import Trajectory

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


_SUPPORTED_SIGNALS = ("dofs_pos", "dofs_vel", "dofs_torque", "base_quat", "base_ang_vel")


def reset_to_initial_state(env: "EnvBase", trajectory: Trajectory, entity_name: str) -> None:
    """Reset all envs to the measurement's initial state by writing entity state directly.

    Bypasses ``_reset_idx`` on purpose: firing ``_RESET`` events would re-run
    domain-randomisation terms that may mutate the very parameters being
    identified. The caller is responsible for first placing the sysid
    parameters via :func:`apply_parameters` / :func:`apply_candidates`.

    For floating-base entities, ``qpos`` from a sim-side recording already
    includes the 7-DOF base prefix (because :class:`SysIDRecorder` writes it
    using ``entity.qs_idx_local``). Deployment recordings, however, store
    DOF positions only — ``pos`` and ``quat`` come through as separate
    ``initial_state`` keys, and we apply them after ``set_qpos`` so the
    base actually starts at the measured pose.
    """
    entity = env.entities[entity_name]
    if "qpos" not in trajectory.initial_state:
        raise ValueError("Trajectory.initial_state missing required 'qpos' field.")

    qpos = torch.as_tensor(trajectory.initial_state["qpos"], dtype=torch.float32, device=env.device)
    qpos = qpos.unsqueeze(0).expand(env.num_envs, -1).contiguous()
    entity.set_qpos(qpos, zero_velocity=True)

    if "dofs_vel" in trajectory.initial_state:
        dofs_vel = torch.as_tensor(trajectory.initial_state["dofs_vel"], dtype=torch.float32, device=env.device)
        dofs_vel = dofs_vel.unsqueeze(0).expand(env.num_envs, -1).contiguous()
        entity.set_dofs_vel(dofs_vel)

    # Apply base pose if it was recorded separately. Fixed-base entities don't
    # accept set_pos/set_quat, so guard on is_fixed_base when available.
    is_fixed_base = bool(getattr(entity, "is_fixed_base", False))
    if not is_fixed_base:
        if "pos" in trajectory.initial_state:
            pos = torch.as_tensor(trajectory.initial_state["pos"], dtype=torch.float32, device=env.device)
            if pos.ndim == 1:
                pos = pos.unsqueeze(0).expand(env.num_envs, -1).contiguous()
            entity.set_pos(pos)
        if "quat" in trajectory.initial_state:
            quat = torch.as_tensor(trajectory.initial_state["quat"], dtype=torch.float32, device=env.device)
            if quat.ndim == 1:
                quat = quat.unsqueeze(0).expand(env.num_envs, -1).contiguous()
            entity.set_quat(quat)


def _physics_step(env: "EnvBase", action: torch.Tensor) -> None:
    """Narrow physics-only stepper: action_manager.compute once, then the decimation loop.

    Mirrors the test-harness pattern at ``tests/test_action_modifiers.py`` and
    the inner loop of ``EnvBase.step`` (``envs/base.py:671-676``) but omits
    termination, reward, reset, command, event, recorder, and observation
    manager side-effects. Safe to call on a fresh or mid-trajectory env.
    """
    env.action_manager.compute(action)
    for _ in range(env.env_options.decimation):
        env.action_manager.apply_actions()
        env.scene.step()


def replay_rollout(
    env: "EnvBase",
    trajectory: Trajectory,
    entity_name: str,
    signals: Sequence[str] = ("dofs_pos", "dofs_vel", "dofs_torque"),
) -> dict[str, torch.Tensor]:
    """Replay the measured action trajectory under the currently-applied parameters.

    Uses the narrow physics-only stepper so that termination, reward,
    reset, command, event, and recorder managers do not fire — a measured
    trajectory longer than ``max_episode_length`` would otherwise be
    corrupted by auto-reset.

    All ``env.num_envs`` envs step on every tick (Genesis steps the full
    batch). For K-candidate batched identification, build a dedicated env
    with ``env_options.num_envs = K`` and use :func:`batched_candidate_rollout`;
    otherwise all envs receive the same state (but may hold different
    parameters for Monte-Carlo variance estimation).

    Returns a dict ``{signal_name: tensor(num_envs, n_steps, dim)}``.
    """
    if trajectory.action is None:
        raise ValueError("Trajectory.action is required for replay.")
    for s in signals:
        if s not in _SUPPORTED_SIGNALS:
            raise ValueError(f"Unsupported rollout signal: {s!r}. Expected one of {_SUPPORTED_SIGNALS}.")

    entity = env.entities[entity_name]
    n_steps = len(trajectory)
    signal_dims = {
        "dofs_pos": entity.num_dofs,
        "dofs_vel": entity.num_dofs,
        "dofs_torque": entity.num_dofs,
        "base_quat": 4,
        "base_ang_vel": 3,
    }
    buffers: dict[str, torch.Tensor] = {
        s: torch.zeros(env.num_envs, n_steps, signal_dims[s], device=env.device) for s in signals
    }

    reset_to_initial_state(env, trajectory, entity_name)

    actions = torch.as_tensor(trajectory.action, dtype=torch.float32, device=env.device)
    for t in range(n_steps):
        action_batch = actions[t].unsqueeze(0).expand(env.num_envs, -1).contiguous()
        _physics_step(env, action_batch)
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
    env: "EnvBase",
    params: ParameterSet,
    trajectory: Trajectory,
    entity_name: str,
    signals: Sequence[str] = ("dofs_pos", "dofs_vel", "dofs_torque"),
) -> dict[str, np.ndarray]:
    """Apply ``params.value`` to all envs, replay once, return env-0 predictions.

    This is the serial primitive used by every optimiser that evaluates one
    parameter vector per step (scipy least-squares with FD Jacobian, and
    CMA-ES unless batched mode is opted into).
    """
    apply_parameters(env, params)
    pred = replay_rollout(env, trajectory, entity_name=entity_name, signals=signals)
    return {k: v[0].detach().cpu().numpy() for k, v in pred.items()}


def batched_candidate_rollout(
    env: "EnvBase",
    params: ParameterSet,
    candidates: np.ndarray,
    trajectory: Trajectory,
    entity_name: str,
    signals: Sequence[str] = ("dofs_pos", "dofs_vel", "dofs_torque"),
) -> dict[str, np.ndarray]:
    """Evaluate K candidates in one rollout on a K-sized env.

    Requires ``env.num_envs == K``. Each candidate row is written into the
    matching env via per-env DOF setters; a single replay then produces
    predictions for all K candidates simultaneously.

    Returns ``{signal_name: array(shape=(K, n_steps, dim))}``.
    """
    K = int(candidates.shape[0])
    if K != env.num_envs:
        raise ValueError(
            f"batched_candidate_rollout requires env.num_envs == K (got K={K}, "
            f"num_envs={env.num_envs}). Build the sysid env with "
            f"env_options.num_envs = K or use single_candidate_rollout."
        )
    apply_candidates(env, params, candidates)
    pred = replay_rollout(env, trajectory, entity_name=entity_name, signals=signals)
    return {k: v.detach().cpu().numpy() for k, v in pred.items()}
