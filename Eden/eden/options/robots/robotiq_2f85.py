"""Robotiq 2F-85 parallel-jaw gripper configuration."""

from __future__ import annotations
from typing import ClassVar

from genesis.typing import UnitVec4FType
from eden.options.entities import RobotOptions


class Robotiq2f85(RobotOptions):
    """Robotiq 2F85 parallel-jaw gripper, tuned for stable grasping.

    The silicone finger pads use a flat **box** collision geom with higher
    friction (rather than a cluster of tiny spheres): a box pad makes a flat-face
    contact that resists rotation, so a parallel-jaw pinch actually holds instead
    of letting the object pivot/slip. Pair with ``collision_link_patterns=[".*pad"]``
    so only the pads collide and the object isn't wedged against the inner linkage.

    The driver joints are **force-limited** (``default_dofs_force_limits``): the
    kp=100 position controller commanded to the closed target otherwise develops
    ~40 N·m and crushes straight through a small cube (pads sink ~1.5 cm into each
    face and "grip" by overlap, not a surface pinch). MuJoCo's original
    force-limits the tendon actuator (``forcerange -5 5``); a 3 N·m per-driver cap
    reproduces that, so the cube stops the pads at its surface for a real friction
    grip.
    """

    file: str = "grippers/robotiq_2f85/2f85.xml"

    is_fixed_base: bool = True
    links_to_keep: list[str] = ["gripper_base_link"]
    default_root_quat: UnitVec4FType = (
        0.7071067811865476,
        0.0,
        0.0,
        -0.7071067811865476,
    )

    dofs_name: list[str] = [
        "left_driver_joint",
        "left_spring_link_joint",
        "left_follower",
        "right_driver_joint",
        "right_spring_link_joint",
        "right_follower_joint",
    ]
    default_dofs_pos: dict[str, float] = {
        "left_driver_joint": 0.0,
        "left_spring_link_joint": 0.0,
        "left_follower": 0.0,
        "right_driver_joint": 0.0,
        "right_spring_link_joint": 0.0,
        "right_follower_joint": 0.0,
    }
    default_dofs_kp: dict[str, float] = {
        "left_driver_joint": 100.0,  # 11.25,
        "right_driver_joint": 100.0,  # 11.25,
    }
    default_dofs_kd: dict[str, float] = {
        "left_driver_joint": 10.0,  # 0.1,
        "right_driver_joint": 10.0,  # 0.1,
    }
    # Force-limit the drivers so closing grips the object's surface instead of
    # crushing through it (see the class docstring).
    default_dofs_force_limits: dict[str, float] = {
        "left_driver_joint": 3.0,
        "right_driver_joint": 3.0,
    }

    actuated_dofs_name: ClassVar[list[str]] = ["left_driver_joint", "right_driver_joint"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "left_driver_joint": 0.0,
        "right_driver_joint": 0.0,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "left_driver_joint": 0.8,
        "right_driver_joint": 0.8,
    }
