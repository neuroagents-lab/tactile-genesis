"""Franka Emika Panda and Franka Research 3 arm configurations."""

from __future__ import annotations
from typing import ClassVar

from genesis.typing import Vec3FType
from eden.options.entities import RobotOptions, MetadataOptions


class FrankaResearch3(RobotOptions):
    file: str = "fr3.xml"

    is_fixed_base: bool = True

    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)  # x,y,z [m]
    links_to_keep: list[str] = ["attachment"]
    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -2.27,
        "joint5": 0.0,
        "joint6": 2.27,
        "joint7": 0.785398,
    }
    default_dofs_kp: dict[str, float] = {
        "joint1": 4500,
        "joint2": 4500,
        "joint3": 3500,
        "joint4": 3500,
        "joint5": 2000,
        "joint6": 2000,
        "joint7": 2000,
    }
    default_dofs_kd: dict[str, float] = {
        "joint1": 450,
        "joint2": 450,
        "joint3": 350,
        "joint4": 350,
        "joint5": 200,
        "joint6": 200,
        "joint7": 200,
    }


class FrankaEmikaPanda(RobotOptions):
    file: str = "panda.xml"

    is_fixed_base: bool = True
    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)  # x,y,z [m]
    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -2.27,
        "joint5": 0.0,
        "joint6": 2.27,
        "joint7": 0.785398,
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    links_to_keep: list[str] = ["left_gripper_pad", "right_gripper_pad"]

    # dofs_vel_limit: dict[str, float] = {
    #     "joint1": 2.61799,  # rad/s
    #     "joint2": 2.61799,
    #     "joint3": 2.61799,
    #     "joint4": 2.61799,
    #     "joint5": 3.14159,
    #     "joint6": 3.14159,
    #     "joint7": 3.14159,
    #     "finger_joint1": 0.05,  # 50 mm/s per finger
    #     "finger_joint2": 0.05,
    # }
    default_dofs_kp: dict[str, float] = {
        "joint1": 4500,  # * 0.8,
        "joint2": 4500,  # * 0.8,
        "joint3": 3500,  # * 0.8,
        "joint4": 3500,  # * 0.8,
        "joint5": 2000,  # * 0.8,
        "joint6": 2000,  # * 0.8,
        "joint7": 2000,  # * 0.8,
        "finger_joint1": 100,
        "finger_joint2": 100,
    }
    default_dofs_kd: dict[str, float] = {
        "joint1": 450,
        "joint2": 450,
        "joint3": 350,
        "joint4": 350,
        "joint5": 200,
        "joint6": 200,
        "joint7": 200,
        "finger_joint1": 10,
        "finger_joint2": 10,
    }

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        ee_link_names=[
            "hand",
        ]
    )

    ee_links_name: ClassVar[list[str]] = ["ee_link"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.0,
        "finger_joint2": 0.0,
    }
