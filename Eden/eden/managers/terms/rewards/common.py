"""Common reward/penalty terms (action rate/magnitude, joint-limit penalties)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.utils.torch import compile_model

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@compile_model
def _compute_action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(action - prev_action), dim=1)


@REWARD_TERM_REGISTRY.register()
def action_rate_l2(
    env: EnvBase,
    *,
    action_term_name: str | None = None,
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if action_term_name is not None:
        action_term = env.action_manager.get_term(action_term_name)
        return _compute_action_rate_l2(action_term.action, action_term.prev_action)
    return _compute_action_rate_l2(env.action_manager.action, env.action_manager.prev_action)


@compile_model
def _compute_action_l2(action: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(action), dim=1)


@REWARD_TERM_REGISTRY.register()
def action_l2(
    env: EnvBase,
    *,
    action_term_name: str | None = None,
) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    if action_term_name is not None:
        action_term = env.action_manager.get_term(action_term_name)
        return _compute_action_l2(action_term.action)
    return _compute_action_l2(env.action_manager.action)


@compile_model
def _compute_dofs_pos_limits(dofs_pos: torch.Tensor, soft_limits: torch.Tensor) -> torch.Tensor:
    out_of_limits = -(dofs_pos - soft_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (dofs_pos - soft_limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


@REWARD_TERM_REGISTRY.register()
def dofs_pos_limits(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    entity: Entity = env.entities[entity_name]
    dofs_pos = entity.get_dofs_pos()
    return _compute_dofs_pos_limits(dofs_pos, entity.soft_dofs_pos_limits)


@compile_model
def _compute_dofs_vel_limits(dofs_vel: torch.Tensor, max_vel: float) -> torch.Tensor:
    excess = (dofs_vel.abs() - max_vel).clamp_min(0.0)
    return (excess**2).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def dofs_vel_limits(
    env: EnvBase,
    *,
    entity_name: str,
    max_vel: float | torch.Tensor,
) -> torch.Tensor:
    """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

    Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
    penalty, shaped as the negative squared L2 norm of the excess velocities.
    """
    entity: Entity = env.entities[entity_name]
    dofs_vel = entity.get_dofs_vel()
    return _compute_dofs_vel_limits(dofs_vel, max_vel)


@REWARD_TERM_REGISTRY.register()
def self_collision_cost(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize self collisions."""
    entity: Entity = env.entities[entity_name]
    contacts = entity.get_contacts(with_entity=entity, exclude_self_contact=False)
    return contacts["valid_mask"].sum(dim=1)


@REWARD_TERM_REGISTRY.register()
class UndesiredContacts(RewardTerm):
    """Penalty count of links whose net contact-force magnitude exceeds ``threshold``.

    ``links_name`` accepts the same regex / glob patterns as
    :func:`eden.utils.string.resolve_matching_names`. Use a negative reward
    weight to penalise (e.g. -0.1 per offending link).
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    threshold: float = 1.0

    def build(self) -> None:
        from eden.utils.string import resolve_matching_names

        entity = self._env.entities[self.entity_name]
        link_idx, _ = resolve_matching_names(self.links_name, entity.links_name, preserve_order=True)
        if not link_idx:
            raise ValueError(
                f"UndesiredContacts: links_name={self.links_name} matched no links on entity "
                f"'{self.entity_name}'. Available links: {entity.links_name}"
            )
        self._link_idx = torch.tensor(link_idx, dtype=torch.long, device=self.device)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        entity = self._env.entities[self.entity_name]
        forces = entity.get_links_net_contact_force()  # (N, n_links, 3)
        magnitudes = torch.norm(forces[:, self._link_idx], dim=-1)  # (N, K)
        return (magnitudes > self.threshold).float().sum(dim=-1)


@compile_model
def _compute_action_rate_l2_smooth(
    action: torch.Tensor, prev_action: torch.Tensor, prev_prev_action: torch.Tensor
) -> torch.Tensor:
    c1 = torch.sum(torch.square(action - prev_action), dim=1)
    c2 = torch.sum(torch.square(action - 2 * prev_action + prev_prev_action), dim=1)
    return c1 + c2


@REWARD_TERM_REGISTRY.register()
def action_rate_l2_smooth(
    env: EnvBase,
    *,
    action_term_name: str | None = None,
) -> torch.Tensor:
    """Penalize 1st and 2nd derivative of the actions using L2 squared kernel.

    Requires ``env._prev_prev_action`` to be set externally (e.g. by the
    env wrapper) before each step.  Falls back to zeros if the attribute
    does not exist, which reduces this to :func:`action_rate_l2` for the
    first two steps.
    """
    if action_term_name is not None:
        action_term = env.action_manager.get_term(action_term_name)
        action = action_term.action
        prev_action = action_term.prev_action
    else:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
    prev_prev_action = getattr(env, "_prev_prev_action", torch.zeros_like(action))
    return _compute_action_rate_l2_smooth(action, prev_action, prev_prev_action)


@compile_model
def _compute_dofs_torques(torques: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.square(torques), dim=1) + 1e-8) + torch.sum(torch.abs(torques), dim=1)


@REWARD_TERM_REGISTRY.register()
def dofs_torques(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize joint torques: sqrt(sum(tau^2)) + sum(|tau|)."""
    entity: Entity = env.entities[entity_name]
    torques = entity.get_dofs_force()
    return _compute_dofs_torques(torques)


@compile_model
def _compute_dofs_acc_l2(dvel: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(dvel), dim=1)


@REWARD_TERM_REGISTRY.register()
def dofs_acc_l2(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize joint accelerations approximated as velocity differences.

    Uses ``sum((vel - prev_vel)^2)`` where ``prev_vel`` is stored in
    ``env._prev_dofs_vel`` (must be set externally, e.g. by the env wrapper).
    The config weight should account for the ``1/dt^2`` factor compared to
    raw simulator accelerations.
    """
    entity: Entity = env.entities[entity_name]
    dofs_vel = entity.get_dofs_vel()
    prev_vel = getattr(env, "_prev_dofs_vel", torch.zeros_like(dofs_vel))
    return _compute_dofs_acc_l2(dofs_vel - prev_vel)


@REWARD_TERM_REGISTRY.register()
def similar_to_default(env, entity_name: str) -> torch.Tensor:
    # Penalize joint poses far away from default pose
    robot = env.entities[entity_name]
    return torch.sum(torch.abs(robot.get_dofs_pos() - robot.default_dofs_pos), dim=1)
