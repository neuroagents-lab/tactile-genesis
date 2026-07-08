"""Containment metric terms (object above height / inside region)."""

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
def object_above_height(
    env: EnvBase,
    *,
    entity_name: str,
    height: float,
) -> torch.Tensor:
    """Check if an entity's root z-position is above *height*.

    Returns a float tensor (1.0 if above, 0.0 otherwise).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    height: float
        Minimum height threshold.
    """
    entity: Entity = env.entities[entity_name]
    return entity.get_pos()[:, 2] > height


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_in_region(
    env: EnvBase,
    *,
    entity_name: str,
    region_min: list[float],
    region_max: list[float],
) -> torch.Tensor:
    """Check if an entity's root position is inside an axis-aligned box.

    Returns a float tensor (1.0 if inside, 0.0 otherwise).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    region_min: list[float]
        Lower corner of the box ``[x_min, y_min, z_min]``.
    region_max: list[float]
        Upper corner of the box ``[x_max, y_max, z_max]``.
    """
    entity: Entity = env.entities[entity_name]
    pos = entity.get_pos()
    lo = torch.tensor(region_min, dtype=torch.float, device=env.device).unsqueeze(0)
    hi = torch.tensor(region_max, dtype=torch.float, device=env.device).unsqueeze(0)
    inside = (pos >= lo).all(dim=1) & (pos <= hi).all(dim=1)
    return inside
