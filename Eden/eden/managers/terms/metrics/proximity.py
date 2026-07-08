"""Proximity and relative-position metric terms between entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.typing import Vec3FType

from eden.constants import MetricDirection
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_a_is_on_b(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
    xy_threshold: float = 0.03,
    height_threshold: float = 0.04,
    height_diff: float = 0.0,
):
    """Check if object_a is on object_b by comparing height and xy distance of object origins.

    This function can be used as a termination term or a metric term.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        The name of the entity_a.
    entity_b_name: str
        The name of the entity_b.
    xy_threshold: float
        The threshold for the xy distance.
    height_threshold: float
        The threshold for the height distance.
    height_diff: float
        The expected height difference.

    Returns
    -------
    success: torch.Tensor
        A boolean tensor indicating if the object_a is on the object_b.
    """
    object_a: Entity = env.entities[entity_a_name]
    object_b: Entity = env.entities[entity_b_name]

    pos_diff = object_a.get_pos() - object_b.get_pos()
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    success = torch.logical_and(
        pos_diff[:, 2] > 0.0,  # object_a is above object_b
        torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold),
    )
    return success


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB)
def entity_distance(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
) -> torch.Tensor:
    """Distance between two entity root positions.

    Returns a float tensor of shape ``(num_envs,)`` with the Euclidean
    distance. Registered with ``direction=LIB`` (lower is better).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        Name of the first entity.
    entity_b_name: str
        Name of the second entity.
    """
    entity_a: Entity = env.entities[entity_a_name]
    entity_b: Entity = env.entities[entity_b_name]
    pos_diff = entity_a.get_pos() - entity_b.get_pos()
    return torch.linalg.vector_norm(pos_diff, dim=1)


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def entity_near_target(
    env: EnvBase,
    *,
    entity_name: str,
    target_pos: Vec3FType | torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Check if an entity is within *threshold* of a fixed world position.

    Returns a float tensor (1.0 if within threshold, 0.0 otherwise).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity to check.
    target_pos: list[float] | Tensor
        Target world position ``[x, y, z]``.  A plain list is converted to a
        tensor on first call; passing a pre-built tensor avoids repeated
        allocation on the hot path.
    threshold: float
        Maximum allowed distance.
    """
    entity: Entity = env.entities[entity_name]
    if not isinstance(target_pos, torch.Tensor):
        target_pos = torch.tensor(target_pos, dtype=torch.float, device=env.device).unsqueeze(0)
    dist = torch.linalg.vector_norm(entity.get_pos() - target_pos, dim=1)
    return dist < threshold


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB)
def entity_distance_xy(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
) -> torch.Tensor:
    """Horizontal (XY) distance between two entity root positions.

    Like :func:`entity_distance` but ignores the Z axis.  Useful for
    placement tasks where objects sit at different heights (e.g. an apple
    on a plate).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        Name of the first entity.
    entity_b_name: str
        Name of the second entity.
    """
    entity_a: Entity = env.entities[entity_a_name]
    entity_b: Entity = env.entities[entity_b_name]
    pos_diff = entity_a.get_pos()[:, :2] - entity_b.get_pos()[:, :2]
    return torch.linalg.vector_norm(pos_diff, dim=1)


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def entity_near_entity_xy(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Check if two entities are within *threshold* horizontal (XY) distance.

    Returns 1.0 when the XY distance between root positions is below
    *threshold*, 0.0 otherwise.  Useful for placement tasks where objects
    naturally sit at different heights.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        Name of the first entity.
    entity_b_name: str
        Name of the second entity.
    threshold: float
        Maximum allowed horizontal distance.
    """
    entity_a: Entity = env.entities[entity_a_name]
    entity_b: Entity = env.entities[entity_b_name]
    pos_diff = entity_a.get_pos()[:, :2] - entity_b.get_pos()[:, :2]
    dist = torch.linalg.vector_norm(pos_diff, dim=1)
    return dist < threshold


@METRIC_TERM_REGISTRY.register()
class EeNearEntity(MetricTerm):
    """Check if the mean end-effector position is within *threshold* of a target entity.

    This is a class-based metric that caches link indices at ``build()`` time
    for efficiency.

    Parameters
    ----------
    robot_name: str
        Name of the robot entity.
    ee_link_names: list[str]
        Link names whose mean position is treated as the EE position.
    entity_name: str
        Name of the target entity.
    threshold: float
        Maximum allowed distance.
    """

    robot_name: str = "robot"
    ee_link_names: list[str] = []
    entity_name: str = "obj"
    threshold: float = 0.05

    def build(self):
        self._robot = self._env.entities[self.robot_name]
        self._target = self._env.entities[self.entity_name]
        _, ee_idx = self._robot.find_named_links_idx_local(self.ee_link_names, preserve_order=True)
        self._ee_idx_local = ee_idx

    def compute(self) -> torch.Tensor:
        ee_pos = self._robot.get_links_pos(ls_idx_local=self._ee_idx_local).mean(dim=1)
        obj_pos = self._target.get_pos()
        dist = torch.linalg.vector_norm(ee_pos - obj_pos, dim=1)
        return dist < self.threshold
