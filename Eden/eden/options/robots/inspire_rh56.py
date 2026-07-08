"""Inspire RH56 dexterous hand configurations (left/right)."""

from __future__ import annotations
from typing import ClassVar

from eden.options.entities import RobotOptions, MetadataOptions


# WARNING: THIS IS A PLACEHOLDER FOR THE INSPIRE ROBOT.
# TODO: the inspire setup is unstable, do not use it for now
# We need the physics implementation for non-backdrivable / high joint friction robots.
class _InspireRH56(RobotOptions):
    dofs_name: list[str] = [
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "thumb_intermediate_joint",
        "thumb_distal_joint",
        "index_proximal_joint",
        "index_intermediate_joint",
        "middle_proximal_joint",
        "middle_intermediate_joint",
        "ring_proximal_joint",
        "ring_intermediate_joint",
        "pinky_proximal_joint",
        "pinky_intermediate_joint",
    ]

    default_dofs_pos: dict[str, float] = {
        "thumb_proximal_yaw_joint": 0.0,
        "thumb_proximal_pitch_joint": 0.0,
        "thumb_intermediate_joint": 0.0,
        "thumb_distal_joint": 0.0,
        "index_proximal_joint": 0.0,
        "index_intermediate_joint": 0.0,
        "middle_proximal_joint": 0.0,
        "middle_intermediate_joint": 0.0,
        "ring_proximal_joint": 0.0,
        "ring_intermediate_joint": 0.0,
        "pinky_proximal_joint": 0.0,
        "pinky_intermediate_joint": 0.0,
    }
    default_dofs_kp: dict[str, float] = {
        "*": 500.0,  # 100.0,
        # "*proximal*": 1000.0,#500.0,
        # "*intermediate*": 600.0,#500.0,
        # "*distal*": 600.0,#500.0,
    }
    default_dofs_kd: dict[str, float] = {
        "*": 50.0,  # 100.0,
        # "*proximal*": 10.0,#100.0,
        # "*intermediate*": 6.0,#100.0,
        # "*distal*": 6.0,#100.0,
    }
    actuated_dofs_name: ClassVar[list[str]] = [
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint",
    ]


class InspireRH56_R(_InspireRH56):
    file: str = "inspire_hand_right.urdf"

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        active_dofs_name=[
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
        ],
        mimic_joint_map={
            "thumb_distal_joint": ("thumb_proximal_pitch_joint", 0.667),
            "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.334),
            "index_intermediate_joint": ("index_proximal_joint", 1.06399),
            "middle_intermediate_joint": ("middle_proximal_joint", 1.06399),
            "ring_intermediate_joint": ("ring_proximal_joint", 1.06399),
            "pinky_intermediate_joint": ("pinky_proximal_joint", 1.06399),
        },
    )


class InspireRH56_L(_InspireRH56):
    file: str = "inspire_hand_left.urdf"

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        active_dofs_name=[
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
        ],
        mimic_joint_map={
            "thumb_distal_joint": ("thumb_proximal_pitch_joint", 0.667),
            "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.334),
            "index_intermediate_joint": ("index_proximal_joint", 1.06399),
            "middle_intermediate_joint": ("middle_proximal_joint", 1.06399),
            "ring_intermediate_joint": ("ring_proximal_joint", 1.06399),
            "pinky_intermediate_joint": ("pinky_proximal_joint", 1.06399),
        },
    )
