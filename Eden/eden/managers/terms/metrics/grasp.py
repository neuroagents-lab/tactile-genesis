"""Grasp metric terms (object lifted, is-grasping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity
    from eden.options.managers.metrics import MetricTermOptions


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_lifted(
    env: EnvBase,
    *,
    entity_name: str,
    min_height: float,
) -> torch.Tensor:
    """Check if an entity has been lifted above *min_height*.

    Returns 1.0 when the entity's root z-position exceeds *min_height*,
    0.0 otherwise.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    min_height: float
        Height threshold.
    """
    entity: Entity = env.entities[entity_name]
    return entity.get_pos()[:, 2] > min_height


@METRIC_TERM_REGISTRY.register()
class IsGrasping(MetricTerm):
    """Detect a parallel-jaw grasp via contact force alignment.

    Returns 1.0 per environment when **both** gripper pads have contact
    forces aligned with their closing direction above *force_threshold*.
    This correctly detects a grasp even when the fingers cannot fully
    close (e.g. when an object is between them).

    The algorithm mirrors the grasp detection logic used by
    :class:`~eden.managers.terms.actions.welding_actions.ParallelJawWelding`.

    Parameters
    ----------
    robot_name : str
        Name of the robot entity.
    left_gripper_link_name : str
        Link name of the left gripper pad (e.g. ``"left_gripper_pad"``).
    right_gripper_link_name : str
        Link name of the right gripper pad (e.g. ``"right_gripper_pad"``).
    force_threshold : float
        Minimum projected contact force magnitude on **each** pad to
        count as grasping.  Lower values are more sensitive.
    """

    robot_name: str = "robot"
    left_gripper_link_name: str = "left_gripper_pad"
    right_gripper_link_name: str = "right_gripper_pad"
    force_threshold: float = 1.0

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env=env, options=options)
        self._left_link_idx: int = -1
        self._right_link_idx: int = -1
        self._gripper_idx_local: list[int] = []

    def build(self):
        entity = self._env.entities[self.robot_name]
        self._entity = entity
        self._left_link_idx = entity.get_link(self.left_gripper_link_name).idx
        self._right_link_idx = entity.get_link(self.right_gripper_link_name).idx
        self._gripper_idx_local = [
            entity.get_link(self.left_gripper_link_name).idx_local,
            entity.get_link(self.right_gripper_link_name).idx_local,
        ]

    def compute(self) -> torch.Tensor:
        contact_info = self._entity.get_contacts(exclude_self_contact=True)
        link_a = contact_info["link_a"]
        link_b = contact_info["link_b"]
        force_a = contact_info["force_a"]
        force_b = contact_info["force_b"]
        valid_mask = contact_info["valid_mask"]

        # Sum contact forces acting on each gripper pad
        left_forces = torch.sum(
            force_a * ((link_a == self._left_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        left_forces += torch.sum(
            force_b * ((link_b == self._left_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        right_forces = torch.sum(
            force_a * ((link_a == self._right_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        right_forces += torch.sum(
            force_b * ((link_b == self._right_link_idx) & valid_mask)[..., None],
            dim=1,
        )

        # Compute gripper close direction (left pad → right pad)
        gripper_pos = self._entity.get_links_pos(ls_idx_local=self._gripper_idx_local)
        close_dir = gripper_pos[:, 1] - gripper_pos[:, 0]  # left → right
        close_dir = close_dir / close_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Project forces onto close direction → shape (num_envs,)
        left_proj = torch.bmm(
            left_forces.unsqueeze(1),
            close_dir.unsqueeze(2),
        ).reshape(-1)
        right_proj = torch.bmm(
            right_forces.unsqueeze(1),
            (-close_dir).unsqueeze(2),
        ).reshape(-1)

        left_ok = left_proj.abs() > self.force_threshold
        right_ok = right_proj.abs() > self.force_threshold
        return left_ok & right_ok
