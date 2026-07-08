"""i2RT YAM arm configuration."""

from __future__ import annotations
from typing import ClassVar

from eden.options.entities import RobotOptions
from eden.options.actuators import ActuatorSpecOptions, DamiaoDM4340, DamiaoDM4310


__all__ = ["Yam", "YAM_ACTION_SCALE"]


_Yam_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    "joint1": DamiaoDM4340,
    "joint2": DamiaoDM4340,
    "joint3": DamiaoDM4340,
    "joint4": DamiaoDM4310,
    "joint5": DamiaoDM4310,
    "joint6": DamiaoDM4310,
    "left_finger": DamiaoDM4310,
    "right_finger": DamiaoDM4310,
}


class Yam(RobotOptions):
    """YAM Robot Configuration."""

    file: str = "i2rt_yam/yam.xml"

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = _Yam_DOFS_SPEC

    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "left_finger",
        "right_finger",
    ]

    links_to_keep: list[str] = [
        "link_left_finger",
        "link_right_finger",
    ]

    default_dofs_pos: dict[str, float] = {
        "joint1": 0.0,
        "joint2": 1.047,
        "joint3": 1.05,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
        "left_finger": 0.037524,
        "right_finger": -0.037524,
    }

    # Raibert helpers below are evaluated at class-definition time using the
    # default ``natural_freq=NATURAL_FREQ`` (10 Hz). If you tune NATURAL_FREQ
    # globally these dicts won't follow — rebuild them on the subclass.
    # Gripper fingers run at 5x the heuristic kp for stiff grasping.
    default_dofs_kp: dict[str, float] = {
        **{p: s.RAIBERT_HEURISTIC_KP() for p, s in _Yam_DOFS_SPEC.items()},
        "left_finger": 5 * DamiaoDM4310.RAIBERT_HEURISTIC_KP(),
        "right_finger": 5 * DamiaoDM4310.RAIBERT_HEURISTIC_KP(),
    }
    default_dofs_kd: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KD() for p, s in _Yam_DOFS_SPEC.items()}
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in _Yam_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in _Yam_DOFS_SPEC.items()}

    ee_links_name: ClassVar[list[str]] = ["link_6"]
    open_dofs_pos: ClassVar[dict[str, float]] = {
        "left_finger": 0.037524,
        "right_finger": -0.037524,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "left_finger": 0.0,
        "right_finger": 0.0,
    }


# Per-joint action scale = ``0.25 * EFFORT_LIMIT / kp``. Fingers are excluded:
# the gripper is driven by BinaryJointController which doesn't consume a
# per-joint scale, and including them here would also be incorrect because
# ``default_dofs_kp`` overrides finger kp to 5x the heuristic value (so the
# Raibert action_scale would be 5x too large for the actual kp).
YAM_ACTION_SCALE: dict[str, float] = {
    p: s.RAIBERT_HEURISTIC_ACTION_SCALE() for p, s in _Yam_DOFS_SPEC.items() if not p.endswith("_finger")
}
