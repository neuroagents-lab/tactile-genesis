"""Franka Hand parallel-jaw gripper configuration."""

from __future__ import annotations
from typing import ClassVar

from eden.options.entities import RobotOptions


class FrankaHand(RobotOptions):
    file: str = "franka_hand.xml"

    is_fixed_base: bool = True
    links_to_keep: list[str] = ["hand", "left_gripper_pad", "right_gripper_pad"]

    dofs_name: list[str] = [
        "finger_joint1",
        "finger_joint2",
    ]

    default_dofs_pos: dict[str, float] = {
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    default_dofs_kp: dict[str, float] = {
        "finger_joint1": 100.0,
        "finger_joint2": 100.0,
    }
    default_dofs_kd: dict[str, float] = {
        "finger_joint1": 10.0,
        "finger_joint2": 10.0,
    }

    ee_links_name: ClassVar[list[str]] = ["ee_link"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.0,
        "finger_joint2": 0.0,
    }
