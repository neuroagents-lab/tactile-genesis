"""Universal Robots UR5e arm configuration."""

from __future__ import annotations
from typing import ClassVar

import numpy as np

from eden.options.entities import RobotOptions


class UR5e(RobotOptions):
    file: str = "ur5e.urdf"

    is_fixed_base: bool = True
    # Keep URDF fixed end-effector link so it can be referenced by controllers
    # and attachment hooks (e.g., gripper mount point).
    links_to_keep: list[str] = ["ee_link"]

    dofs_name: list[str] = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "shoulder_pan": 0 * np.pi / 180,
        "shoulder_lift": -90 * np.pi / 180,
        "elbow": 90 * np.pi / 180,
        "wrist_1": -90 * np.pi / 180,
        "wrist_2": -90 * np.pi / 180,
        "wrist_3": 0 * np.pi / 180,
    }
    default_dofs_kp: dict[str, float] = {
        "shoulder_pan": 4500,
        "shoulder_lift": 4500,
        "elbow": 3500,
        "wrist_1": 3500,
        "wrist_2": 1000,
        "wrist_3": 1000,
    }
    default_dofs_kd: dict[str, float] = {
        "shoulder_pan": 450,
        "shoulder_lift": 450,
        "elbow": 350,
        "wrist_1": 350,
        "wrist_2": 200,
        "wrist_3": 200,
    }

    ee_links_name: ClassVar[list[str]] = ["ee_link"]
