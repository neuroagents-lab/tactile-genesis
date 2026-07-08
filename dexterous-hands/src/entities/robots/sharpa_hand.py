from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions, RobotOptions
from eden.options.materials import MaterialLike, RigidMaterialOptions

from registry import ROBOT_REGISTRY
from utils import get_asset_path


@ROBOT_REGISTRY.register(name="sharpa")
class SharpaHand(RobotOptions):
    file: str = get_asset_path("hands/sharpa/sharpa_right_updated_effort.urdf")
    material: MaterialLike = RigidMaterialOptions(gravity_compensation=0.0)

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    dofs_name: list[str] = [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ]

    default_dofs_pos: dict[str, float] = dict.fromkeys(dofs_name, 0.0)

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        fingertip_links=[
            "right_thumb_DP",
            "right_index_DP",
            "right_middle_DP",
            "right_ring_DP",
            "right_pinky_DP",
        ],
        finger_links=[
            "right_thumb_MC",
            "right_thumb_PP",
            "right_thumb_DP",
            "right_index_PP",
            "right_index_MP",
            "right_index_DP",
            "right_middle_PP",
            "right_middle_MP",
            "right_middle_DP",
            "right_ring_PP",
            "right_ring_MP",
            "right_ring_DP",
            "right_pinky_MC",
            "right_pinky_PP",
            "right_pinky_MP",
            "right_pinky_DP",
        ],
        palm_link="right_hand_C_MC",
        calibration_params="data/sharpa/manual_params.yaml",
        tactile_probe_cfgs={
            "low": get_asset_path("sensors/sharpa/low/probes_98_sharpa.json"),
            "med": get_asset_path("sensors/sharpa/med/probes_207_hand_sharpa.json"),
            "high": get_asset_path("sensors/sharpa/high/probes_782_hand_sharpa.json"),
        },
        priv_sensor_cfgs={
            "fingertips": {
                "right_index_DP": (0.018, 0.005, 0.0001),
                "right_middle_DP": (0.018, 0.005, 0.0001),
                "right_ring_DP": (0.018, 0.005, 0.0001),
                "right_pinky_DP": (0.018, 0.005, 0.0001),
                "right_thumb_DP": (0.020, 0.005, 0.0005),
            }
        },
    )

    def model_post_init(self, context):
        super().model_post_init(context)
        # Deferred so `gs.surfaces.Metal()` is not built at module import,
        # which would violate the Eden-before-genesis import ordering.
        import genesis as gs

        self.surface = gs.surfaces.Metal(
            color=(0.82, 0.84, 0.86, 1.0),
            roughness=0.12,
            metallic=1.0,
        )
