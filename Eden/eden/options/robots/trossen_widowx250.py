"""Trossen WidowX-250 arm configuration."""

from typing import ClassVar

from eden.options.entities import RobotOptions


class WidowX250(RobotOptions):
    file: str = "wx250s.xml"
    is_fixed_base: bool = True

    dofs_name: list[str] = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "left_finger",
        "right_finger",
    ]

    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "waist": 0.0,
        "shoulder": 0.0,
        "elbow": 0.0,
        "forearm_roll": 0.0,
        "wrist_angle": 0.0,
        "wrist_rotate": 0.0,
        "left_finger": 0.037,
        "right_finger": -0.037,
    }

    links_to_keep: list[str] = ["gripper_link"]
    default_dofs_kp: dict[str, float] = {
        "waist": 4500 * 0.8,
        "shoulder": 4500 * 0.8,
        "elbow": 3500 * 0.8,
        "forearm_roll": 3500 * 0.8,
        "wrist_angle": 2000 * 0.8,
        "wrist_rotate": 2000 * 0.8,
        "left_finger": 100,
        "right_finger": 100,
    }

    default_dofs_kd: dict[str, float] = {
        "waist": 450,
        "shoulder": 450,
        "elbow": 350,
        "forearm_roll": 350,
        "wrist_angle": 200,
        "wrist_rotate": 200,
        "left_finger": 10,
        "right_finger": 10,
    }

    ee_links_name: ClassVar[list[str]] = ["gripper_link"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "left_finger": 0.037,
        "right_finger": -0.037,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "left_finger": 0.0,
        "right_finger": 0.0,
    }
