from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions, RobotOptions
from eden.options.materials import MaterialLike, RigidMaterialOptions

from registry import ROBOT_REGISTRY
from utils import get_asset_path


@ROBOT_REGISTRY.register(name="xhand1")
class XHand1(RobotOptions):
    file: str = get_asset_path("hands/xhand1/xhand_right_updated_effort.urdf")
    material: MaterialLike = RigidMaterialOptions(gravity_compensation=0.0)

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    dofs_name: list[str] = [
        "right_hand_thumb_bend_joint",
        "right_hand_index_bend_joint",
        "right_hand_mid_joint1",
        "right_hand_ring_joint1",
        "right_hand_pinky_joint1",
        "right_hand_thumb_rota_joint1",
        "right_hand_index_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint2",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_joint2",
    ]

    default_dofs_pos: dict[str, float] = dict.fromkeys(dofs_name, 0.0)

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        fingertip_links=[
            "right_hand_thumb_rota_link2",
            "right_hand_index_rota_link2",
            "right_hand_mid_link2",
            "right_hand_ring_link2",
            "right_hand_pinky_link2",
        ],
        finger_links=[
            "right_hand_thumb_bend_link",
            "right_hand_index_bend_link",
            "right_hand_mid_link1",
            "right_hand_ring_link1",
            "right_hand_pinky_link1",
            "right_hand_thumb_rota_link1",
            "right_hand_index_rota_link1",
            "right_hand_mid_link2",
            "right_hand_ring_link2",
            "right_hand_pinky_link2",
            "right_hand_thumb_rota_link2",
            "right_hand_index_rota_link2",
        ],
        palm_link="right_hand_link",
        calibration_params="data/xhand_sysid/manual_params.yaml",
        grasp_center=(0.06, -0.01, 0.07),
        tactile_probe_cfgs={
            "low": get_asset_path("sensors/xhand1/low/probes_90_hand_xhand1.json"),
            "med": get_asset_path("sensors/xhand1/med/probes_199_hand_xhand1.json"),
            "high": get_asset_path("sensors/xhand1/high/probes_668_hand_xhand1.json"),
        },
        priv_sensor_cfgs={
            "fingertips": {
                "right_hand_thumb_rota_link2": (0.0, 0.042, 0.007),
                "right_hand_index_rota_link2": (0.008, 0.0, 0.035),
                "right_hand_mid_link2": (0.008, 0.0, 0.035),
                "right_hand_ring_link2": (0.008, 0.0, 0.035),
                "right_hand_pinky_link2": (0.008, 0.0, 0.035),
            },
            "tips+palm": {
                "right_hand_thumb_rota_link2": (0.0, 0.042, 0.007),
                "right_hand_index_rota_link2": (0.008, 0.0, 0.035),
                "right_hand_mid_link2": (0.008, 0.0, 0.035),
                "right_hand_ring_link2": (0.008, 0.0, 0.035),
                "right_hand_pinky_link2": (0.008, 0.0, 0.035),
                "right_hand_link": (0.020, -0.002, 0.075),
            },
        },
    )
