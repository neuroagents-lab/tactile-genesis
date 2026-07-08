"""Common termination terms (timeout, success/failure, illegal contact, limits)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers import TERMINATION_TERM_REGISTRY
from eden.constants import MetricMode
from eden.utils.sample import apply_probability


if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.entities.base import Entity


@TERMINATION_TERM_REGISTRY.register()
def time_out(env: RLEnvBase) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


@TERMINATION_TERM_REGISTRY.register()
def on_success(env: RLEnvBase) -> torch.Tensor:
    return env.metric_manager.success_buf


@TERMINATION_TERM_REGISTRY.register()
def on_failure(env: RLEnvBase) -> torch.Tensor:
    return ~env.metric_manager.success_buf


@TERMINATION_TERM_REGISTRY.register()
def on_metric_success(env: RLEnvBase, metric_name: str) -> torch.Tensor:
    term = env.metric_manager._terms[metric_name]
    assert term.metric_mode == MetricMode.INTERVAL
    return term.is_success(env.metric_manager._step_metric[metric_name])


@TERMINATION_TERM_REGISTRY.register()
def illegal_contact(env: RLEnvBase, *, entity_name: str, illegal_contact_entity_names: list[str]) -> torch.Tensor:
    entity: Entity = env.entities[entity_name]
    contacts = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for illegal_contact_entity_name in illegal_contact_entity_names:
        illegal_contact_entity: Entity = env.entities[illegal_contact_entity_name]
        contacts |= entity.get_contacts(with_entity=illegal_contact_entity)["valid_mask"].any(dim=-1)
    return contacts


@TERMINATION_TERM_REGISTRY.register()
def dofs_pos_limit_exceeded(
    env: RLEnvBase,
    *,
    entity_name: str,
    probability: float = 1.0,
) -> torch.Tensor:
    """Terminate when any DOF position is outside the entity's soft position limits.

    ``probability`` gates the termination globally per call (one Bernoulli draw),
    leaving probabilistic slack for policies that occasionally bump the limits.
    Set to 1.0 for hard termination.
    """
    entity: Entity = env.entities[entity_name]
    soft = entity.soft_dofs_pos_limits
    if soft is None:
        # Non-articulated or unbuilt entity: nothing to violate.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dofs_pos = entity.get_dofs_pos()
    lower_violation = -(dofs_pos - soft[:, :, 0]).clip(max=0.0)
    upper_violation = (dofs_pos - soft[:, :, 1]).clip(min=0.0)
    violation = torch.sum(lower_violation + upper_violation, dim=1) > 0.0
    return apply_probability(violation, probability)


@TERMINATION_TERM_REGISTRY.register()
def dofs_vel_limit_exceeded(
    env: RLEnvBase,
    *,
    entity_name: str,
    max_vel: float,
    probability: float = 1.0,
) -> torch.Tensor:
    """Terminate when any DOF velocity exceeds ``max_vel`` in magnitude.

    ``probability`` gates the termination globally per call. Set to 1.0 for hard
    termination.
    """
    entity: Entity = env.entities[entity_name]
    dofs_vel = entity.get_dofs_vel()
    violation = (dofs_vel.abs() > max_vel).any(dim=1)
    return apply_probability(violation, probability)


@TERMINATION_TERM_REGISTRY.register()
def cumulative_reward_below_threshold(
    env: RLEnvBase,
    *,
    reward_term: str,
    per_step_threshold: float,
    grace_steps: int = 0,
    check_interval: int = 1,
) -> torch.Tensor:
    """Terminate envs whose running raw reward for ``reward_term`` falls below a threshold.

    The threshold is ``per_step_threshold * env.dt * completed_steps``.

    Mirrors dexmachina's early-reset curriculum trigger
    (``dexmachina/envs/base_env.py:712-750``). Reads the weight-free,
    dt-scaled accumulation via :meth:`RewardManager.get_episode_sum` so the
    threshold stays stable when reward curricula mutate ``term.weight`` at
    runtime.

    Mental model: average reward per completed step ≈
    ``cum / (dt * (episode_length_buf - 1))``. The term fires when that
    ratio drops below ``per_step_threshold``, optionally rounded down to
    multiples of ``check_interval`` steps.

    Parameters
    ----------
    env : RLEnvBase
        The environment instance.
    reward_term : str
        Name of a term registered with the reward manager. Raises
        ``KeyError`` (with the active term list) if missing.
    per_step_threshold : float
        Floor on raw per-step reward (unweighted ``compute()`` output).
        Compared as ``cum < per_step_threshold * env.dt * completed_steps``.
        Same units as dexmachina's ``--early_reset_threshold`` flag (paper
        default ``0.5``); no manual ``* dt`` rescale at config time.
    grace_steps : int
        Suppress the gate while ``episode_length_buf <= grace_steps``.
        Default ``0`` matches dexmachina (its ``interval=0`` rung is
        implicitly graceful — the natural ``cum == 0`` initial state
        satisfies ``cum < 0`` vacuously).
    check_interval : int
        ``1`` (default) → continuous per-completed-step check.
        ``> 1`` → dexmachina interval-ladder: round elapsed-steps down
        to ``check_interval`` multiples before applying the threshold.
        Pass ``5`` for ``early_reset_interval=5``, ``20`` for the
        auxiliary imitation/contact/BC triggers.
    """
    cum = env.reward_manager.get_episode_sum(reward_term, raw=True)
    # _episode_raw_sums covers steps 1..t-1 when termination evaluates at step t,
    # so the elapsed multiplier is (episode_length_buf - 1) clamped at zero.
    completed = (env.episode_length_buf - 1).clamp(min=0)
    if check_interval > 1:
        # Match dexmachina's strict `>` rung compare: largest rung strictly below N
        # equals ((N - 1) // ci) * ci. Naive (N // ci) * ci would round one rung
        # too high on exact multiples (step 20 vs rung 20 instead of 15).
        completed = (completed // check_interval) * check_interval
    floor = per_step_threshold * env.dt * completed.float()
    return (cum < floor) & (env.episode_length_buf > grace_steps)


@TERMINATION_TERM_REGISTRY.register()
def dofs_torque_limit_exceeded(
    env: RLEnvBase,
    *,
    entity_name: str,
    soft_ratio: float = 1.0,
    probability: float = 1.0,
) -> torch.Tensor:
    """Terminate when any applied DOF torque exceeds ``soft_ratio * default_dofs_force_limits``.

    Uses the entity's per-DOF effort limit (set from actuator-spec ``EFFORT_LIMIT``)
    and triggers when ``|tau| > soft_ratio * limit`` for any joint. ``probability``
    gates the termination globally per call.
    """
    entity: Entity = env.entities[entity_name]
    torques = entity.get_dofs_force()
    limit = entity.default_dofs_force_limits
    if limit is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # default_dofs_force_limits is shaped (num_envs, num_dofs) when batched, else (1, num_dofs).
    threshold = soft_ratio * limit
    violation = (torques.abs() > threshold).any(dim=1)
    return apply_probability(violation, probability)
