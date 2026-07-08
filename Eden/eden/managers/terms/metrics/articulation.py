"""Articulation metric terms (e.g. joint angle at target)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def joint_angle_at_target(
    env: EnvBase,
    *,
    entity_name: str,
    joint_name: str,
    target_angle: float,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Check if a single DOF is within *threshold* of *target_angle*.

    Returns 1.0 when the joint position satisfies
    ``|current - target_angle| < threshold``, 0.0 otherwise.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the articulated entity.
    joint_name: str
        Name of the DOF to query.
    target_angle: float
        Target joint position (radians or meters depending on joint type).
    threshold: float
        Maximum allowed deviation.
    """
    entity: Entity = env.entities[entity_name]
    _, dof_idx = entity.find_named_dofs_idx_local([joint_name], preserve_order=True)
    dof_pos = entity.get_dofs_pos(dofs_idx_local=dof_idx)  # (num_envs, 1)
    error = torch.abs(dof_pos[:, 0] - target_angle)
    return error < threshold
