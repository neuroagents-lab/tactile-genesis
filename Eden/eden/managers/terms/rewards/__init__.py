"""Built-in reward terms."""

from eden.managers.terms.rewards.common import (
    dofs_pos_limits,
    dofs_vel_limits,
    action_rate_l2,
    action_rate_l2_smooth,
    action_l2,
    self_collision_cost,
    similar_to_default,
    dofs_torques,
    dofs_acc_l2,
    UndesiredContacts,
)


__all__ = [
    "dofs_pos_limits",
    "dofs_vel_limits",
    "action_rate_l2",
    "action_rate_l2_smooth",
    "action_l2",
    "self_collision_cost",
    "similar_to_default",
    "dofs_torques",
    "dofs_acc_l2",
    "UndesiredContacts",
]
