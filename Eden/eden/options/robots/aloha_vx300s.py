"""ALOHA (ViperX 300 S) bimanual robot configuration."""

from typing import ClassVar

from genesis.typing import UnitVec4FType

from eden.options.entities import RobotOptions


class Aloha(RobotOptions):
    file: str = "aloha.urdf"
    is_fixed_base: bool = True
    default_root_quat: UnitVec4FType = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)  # w,x,y,z

    dofs_name: list[str] = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate",
        "left/left_finger",
        "left/right_finger",
        "right/waist",
        "right/shoulder",
        "right/elbow",
        "right/forearm_roll",
        "right/wrist_angle",
        "right/wrist_rotate",
        "right/left_finger",
        "right/right_finger",
    ]

    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "*/waist": 0.0,
        "*/shoulder": -1.85,
        "*/elbow": 1.6,
        "*/forearm_roll": 0.0,
        "*/wrist_angle": 0.0,
        "*/wrist_rotate": 0.0,
        "*/left_finger": 0.04,
        "*/right_finger": 0.04,
    }

    links_to_keep: list[str] = ["left/gripper_base", "right/gripper_base"]
    default_dofs_kp: dict[str, float] = {
        "*/waist": 4500 * 0.8,
        "*/shoulder": 4500 * 0.8,
        "*/elbow": 3500 * 0.8,
        "*/forearm_roll": 3500 * 0.8,
        "*/wrist_angle": 2000 * 0.8,
        "*/wrist_rotate": 2000 * 0.8,
        "*/left_finger": 100,
        "*/right_finger": 100,
    }

    default_dofs_kd: dict[str, float] = {
        "*/waist": 450,
        "*/shoulder": 450,
        "*/elbow": 350,
        "*/forearm_roll": 350,
        "*/wrist_angle": 200,
        "*/wrist_rotate": 200,
        "*/left_finger": 10,
        "*/right_finger": 10,
    }

    ee_links_name: ClassVar[list[str]] = ["left/gripper_base", "right/gripper_base"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "left/left_finger": 0.04,
        "left/right_finger": 0.04,
        "right/left_finger": 0.04,
        "right/right_finger": 0.04,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "left/left_finger": 0.005,
        "left/right_finger": 0.005,
        "right/left_finger": 0.005,
        "right/right_finger": 0.005,
    }
