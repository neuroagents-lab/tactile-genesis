"""Welding action terms for suction-cup and parallel-jaw attach/detach."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.base import ManagerTermBase
from eden.utils.common import ConfigurableMixin

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


class _WeldingBase(ActionTerm):
    """A special ActionTerm that does not require any dofs for control."""

    entity_name: str = ""
    ee_link_name: str = ""
    ignore_entities_name: list[str] = []
    ignore_links_idx: list[int] = []

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

        assert not self.dofs_name, "This ActionTerm does not require any dofs for control."

        self._entity = None
        self.dofs_name = []
        self.dofs_name_map = {}
        self.dofs_idx_map = {}
        self.dofs_idx_local = torch.tensor([], dtype=gs.tc_int, device=self.device)
        self._n_dofs = 0

        self._raw_action: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None
        self._processed_action: torch.Tensor | None = None

        self._ee_link_idx = None
        self._welded = None
        self._ignore_links_idx = set()

    def build(self) -> None:
        self._entity = self._env.entities[self.entity_name]

        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_action = torch.zeros(self.num_envs, device=self.device)

        ee_link = self._entity.get_link(self.ee_link_name)
        assert ee_link.n_geoms > 0, (
            "End-effector link must have at least one geom, but got 0; make sure the link is not a site."
        )
        self._ee_link_idx = ee_link.idx
        self._welded = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
        ignore_entities = [self._env.entities[name] for name in self.ignore_entities_name]
        self._ignore_links_idx = set(
            self.ignore_links_idx + [link.idx for ignore_entity in ignore_entities for link in ignore_entity.links]
        )

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        if actions.dtype == torch.bool:
            # true: suction, false: release
            self._processed_action = actions.squeeze()
        else:
            # true: suction, false: release
            self._processed_action = actions.squeeze() > 0

        # NOTE: remove weld constraints if the action is release
        weld_const_info = self._env.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[link_b == self._ee_link_idx],
                    link_b[link_a == self._ee_link_idx],
                ]
            )
        )
        unweld_mask = ~self._processed_action & self._welded
        if not unweld_mask.any():
            return
        for link_idx in objects_link_idx:
            # NOTE: ensure the object is welded before deleting the weld constraint
            envs_unweld = unweld_mask & (
                ((link_a == link_idx) & (link_b == self._ee_link_idx)).any(dim=1)
                | ((link_b == link_idx) & (link_a == self._ee_link_idx)).any(dim=1)
            )
            self._env.rigid_solver.delete_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_unweld)
            self._welded[envs_unweld] = False

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = torch.ones(self.num_envs, dtype=gs.tc_bool, device=self.device)
        if isinstance(envs_idx, slice):
            envs_mask = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
            envs_mask[envs_idx] = True
            envs_idx = envs_mask

        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

        weld_const_info = self._env.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[link_b == self._ee_link_idx],
                    link_b[link_a == self._ee_link_idx],
                ]
            )
        )
        for link_idx in objects_link_idx:
            envs_unweld = envs_idx & (
                ((link_a == link_idx) & (link_b == self._ee_link_idx)).any(dim=1)
                | ((link_b == link_idx) & (link_a == self._ee_link_idx)).any(dim=1)
            )
            if not envs_unweld.any():
                continue
            self._env.rigid_solver.delete_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_unweld)
            self._welded[envs_unweld] = False


@ACTION_TERM_REGISTRY.register()
class SuctionCupWelding(_WeldingBase):
    """Perform collision-aware welding between the end-effector link (suction cup) and an object link.

    When the action is applied and the end-effector link is in contact with an object link,
    the end-effector link and the object link are welded together.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    ee_link_name: str
        The name of the end-effector link.
    ignore_entities_name: list[str]
        The names of the entities to exclude from welding (e.g., "table", "floor").
    ignore_links_idx: list[int]
        The link indices to exclude from welding.

    Note
    ----
    action is a boolean tensor, True: suction, False: release
    if float is used, >0: suction, <=0: release
    """

    @property
    def action_dim(self) -> int:
        return 1

    def apply_actions(self) -> None:
        contact_info = self._entity.get_contacts(exclude_self_contact=True)
        link_a = contact_info["link_a"]
        link_b = contact_info["link_b"]
        valid_mask = contact_info["valid_mask"]
        # NOTE: get object links in contact
        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[(link_b == self._ee_link_idx) & valid_mask],
                    link_b[(link_a == self._ee_link_idx) & valid_mask],
                ]
            )
        )
        suction_mask = self._processed_action & ~self._welded
        if not suction_mask.any():
            return
        for link_idx in objects_link_idx:
            if link_idx in self._ignore_links_idx:
                continue
            # NOTE: only add weld constraint if the object is not welded and the action is true
            envs_weld = suction_mask & (
                ((link_a == link_idx) & (link_b == self._ee_link_idx) & valid_mask).any(dim=1)
                | ((link_b == link_idx) & (link_a == self._ee_link_idx) & valid_mask).any(dim=1)
            )
            if not envs_weld.any():
                continue
            self._env.rigid_solver.add_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_weld)
            self._welded[envs_weld] = True


@ACTION_TERM_REGISTRY.register()
class ParallelJawWelding(_WeldingBase):
    """Perform collision-aware welding between the end-effector link and an object link.

    When the action is applied and the end-effector link is in contact with an object link,
    the end-effector link and the object link are welded together.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control. (e.g., "robot")
    ee_link_name: str
        The name of the end-effector link. (e.g., "hand")
    ignore_entities_name: list[str]
        The names of the entities to exclude from welding (e.g., "table", "floor").
    ignore_links_idx: list[int]
        The link indices to exclude from welding.
    left_gripper_link_name: str
        The name of the left gripper link. (e.g., "left_gripper_pad")
    right_gripper_link_name: str
        The name of the right gripper link. (e.g., "right_gripper_pad")
    grasp_force_threshold: float
        The threshold for the gripper to grasp the object.
    """

    left_gripper_link_name: str = ""
    right_gripper_link_name: str = ""
    grasp_force_threshold: float = 1.0

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._left_gripper_link_idx: int = -1
        self._right_gripper_link_idx: int = -1
        self._gripper_idx_local: list[int] = [-1, -1]

    def build(self) -> None:
        super().build()

        self._left_gripper_link_idx = self._entity.get_link(self.left_gripper_link_name).idx
        self._right_gripper_link_idx = self._entity.get_link(self.right_gripper_link_name).idx
        self._gripper_idx_local = [
            self._entity.get_link(self.left_gripper_link_name).idx_local,
            self._entity.get_link(self.right_gripper_link_name).idx_local,
        ]

    @property
    def action_dim(self) -> int:
        return 1

    def apply_actions(self) -> None:
        contact_info = self._entity.get_contacts(exclude_self_contact=True)
        link_a = contact_info["link_a"]
        link_b = contact_info["link_b"]
        force_a = contact_info["force_a"]  # geom a force
        force_b = contact_info["force_b"]  # geom b force
        valid_mask = contact_info["valid_mask"]

        # NOTE: check if the object is grasped by the gripper or not
        # NOTE: we check if the net force on gripper is aligned with the gripper close direction
        left_forces = torch.sum(
            force_a * ((link_a == self._left_gripper_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        left_forces += torch.sum(
            force_b * ((link_b == self._left_gripper_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        right_forces = torch.sum(
            force_a * ((link_a == self._right_gripper_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        right_forces += torch.sum(
            force_b * ((link_b == self._right_gripper_link_idx) & valid_mask)[..., None],
            dim=1,
        )
        gripper_pos = self._entity.get_links_pos(ls_idx_local=self._gripper_idx_local)
        left_close_direction = gripper_pos[:, 1] - gripper_pos[:, 0]
        left_close_direction = left_close_direction / left_close_direction.norm(dim=1, keepdim=True)
        right_close_direction = -left_close_direction

        # project the forces to the close direction
        left_forces_proj = (
            torch.bmm(left_forces.unsqueeze(1), left_close_direction.unsqueeze(2)).squeeze() * left_close_direction
        )
        right_forces_proj = (
            torch.bmm(right_forces.unsqueeze(1), right_close_direction.unsqueeze(2)).squeeze() * right_close_direction
        )

        # check if the forces are aligned with the close direction
        left_aligned_mask = left_forces_proj.norm(dim=1) > self.grasp_force_threshold
        right_aligned_mask = right_forces_proj.norm(dim=1) > self.grasp_force_threshold

        is_grasping = left_aligned_mask & right_aligned_mask

        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[
                        is_grasping
                        & ((link_b == self._left_gripper_link_idx) | (link_b == self._right_gripper_link_idx))
                        & valid_mask
                    ],
                    link_b[
                        is_grasping
                        & ((link_a == self._left_gripper_link_idx) | (link_a == self._right_gripper_link_idx))
                        & valid_mask
                    ],
                ]
            )
        )
        for link_idx in objects_link_idx:
            if link_idx in self._ignore_links_idx:
                continue
            # NOTE: only add weld constraint if the object is not welded and the action is true
            suction_mask = self._processed_action & ~self._welded & is_grasping
            envs_weld = suction_mask & (
                (
                    (link_a == link_idx)
                    & ((link_b == self._left_gripper_link_idx) | (link_b == self._right_gripper_link_idx))
                    & valid_mask
                ).any(dim=1)
                | (
                    (link_b == link_idx)
                    & ((link_a == self._left_gripper_link_idx) | (link_a == self._right_gripper_link_idx))
                    & valid_mask
                ).any(dim=1)
            )
            if not envs_weld.any():
                continue

            self._env.rigid_solver.add_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_weld)
            self._welded[envs_weld] = True
