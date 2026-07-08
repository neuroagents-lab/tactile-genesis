"""Manipulation observation terms (end-effector / object distances)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.options.managers.observations import ObservationTermOptions
from eden.utils.isaac_math import quat_apply, quat_inv

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase
    from eden.managers.terms.commands.oracle import UniformSE3Command, UniformT3Command
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


@OBSERVATION_TERM_REGISTRY.register()
class EndEffectorToObjectDistance(ObservationTerm):
    """Observation term for the distance between the end effector and the object."""

    robot_name: str = "robot"
    left_ee_link_name: str = "left_gripper"
    right_ee_link_name: str = "right_gripper"
    entity_name: str = "obj"
    entity_link_name: str | None = None
    frame: Literal["world", "robot"] = "robot"

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self.target: Entity | None = None
        self.robot: Entity | None = None
        self.ee_idx_local: torch.Tensor | None = None

    def build(self) -> None:
        target_entity: Entity = self._env.entities[self.entity_name]
        if self.entity_link_name is not None:
            self.target = target_entity.get_link(self.entity_link_name)
        else:
            self.target = target_entity

        self.robot = self._env.entities[self.robot_name]
        _, ee_idx_local = self.robot.find_named_links_idx_local(
            [self.left_ee_link_name, self.right_ee_link_name],
            preserve_order=True,
        )
        self.ee_idx_local = torch.tensor(ee_idx_local, device=self._env.device)

    def compute(self, *args, **kwargs):
        ee_pos_w = self.robot.get_links_pos(ls_idx_local=self.ee_idx_local).mean(dim=1)
        obj_pos_w = self.target.get_pos()
        distance_vec_w = obj_pos_w - ee_pos_w

        if self.frame == "world":
            return distance_vec_w
        elif self.frame == "robot":
            base_quat_w = self.robot.get_quat()
            distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
            return distance_vec_b
        else:
            raise ValueError(f"Invalid frame '{self.frame}'. Expected 'world' or 'robot'.")


@OBSERVATION_TERM_REGISTRY.register()
class ObjectToOracleDistance(ObservationTerm):
    """Observation term for the distance between the object and the oracle."""

    entity_name: str = "obj"
    entity_link_name: str | None = None
    command_name: str = "command"
    robot_name: str = "robot"
    frame: Literal["world", "robot"] = "robot"

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self.robot: Entity | None = None
        self.target: RigidLink | None = None
        self.command: UniformSE3Command | UniformT3Command | None = None

    def build(self) -> None:
        self.robot = self._env.entities[self.robot_name]
        target_entity: Entity = self._env.entities[self.entity_name]
        if self.entity_link_name is not None:
            self.target = target_entity.get_link(self.entity_link_name)
        else:
            self.target = target_entity
        self.command = self._env.command_manager.get_term(self.command_name)

    def compute(self, *args, **kwargs):
        oracle_pos_b = self.command.command[..., :3]
        base_pos_w = self.robot.get_pos()
        base_quat_w = self.robot.get_quat()
        oracle_pos_w = quat_apply(base_quat_w, oracle_pos_b) + base_pos_w

        obj_pos_w = self.target.get_pos()
        distance_vec_w = oracle_pos_w - obj_pos_w

        if self.frame == "world":
            return distance_vec_w
        elif self.frame == "robot":
            distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
            return distance_vec_b
        else:
            raise ValueError(f"Invalid frame '{self.frame}'. Expected 'world' or 'robot'.")
