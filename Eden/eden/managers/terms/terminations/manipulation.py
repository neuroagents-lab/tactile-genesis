"""Manipulation termination terms (object out of reach)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


@TERMINATION_TERM_REGISTRY.register()
def object_out_of_reach(
    env: EnvBase,
    *,
    entity_name: str,
    object_entity_name: str,
    object_link_name: str | None = None,
    max_dist: float,
):
    """Check the horizontal distance between the entity and the object is greater than the max_dist."""
    entity: Entity = env.entities[entity_name]  # robot
    object_entity: Entity = env.entities[object_entity_name]  # object
    if object_link_name is not None:
        object_link: RigidLink = object_entity.get_link(object_link_name)
        object_pos_w = object_link.get_pos()
    else:
        object_pos_w = object_entity.get_pos()
    dist = torch.norm(object_pos_w[..., :2] - entity.get_pos()[..., :2], dim=-1)
    return dist > max_dist
