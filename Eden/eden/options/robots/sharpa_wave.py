"""Sharpa Wave dexterous hand configurations (left/right)."""

from typing import ClassVar

from eden.options.entities import RobotOptions, MetadataOptions


class _SharpaWave(RobotOptions):
    default_dofs_pos: dict[str, float] = {}

    default_dofs_kp: dict[str, float] = {
        "*": 50.0,
    }

    default_dofs_kd: dict[str, float] = {
        "*": 25.0,
    }


class SharpaWave_R(_SharpaWave):
    file: str = "sharpa_hand/right_sharpa_ha4_v2_1.urdf"

    # Hand-axis convention via Eden's standard ``up`` / ``front`` fields:
    #   ``up``    = URDF axis the fingers extend along
    #   ``front`` = URDF axis the palm normal points along (= the URDF
    #               axis aligned with world +X after load)
    # Sharpa Wave's right-hand URDF places all finger MCP origins at
    # positive URDF +Z (fingers extend along URDF +Z = ``up``) and the
    # palm normal along URDF +X (= ``front``), matching Eden defaults.
    # The MANO-driven retargeter reads these declarations (and a matching
    # MANO-side declaration in ``dex_retargeter.calibrate``) and derives
    # the URDF↔MANO frame conversion via ``align_up_and_front``.

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

    links_to_keep: list[str] = [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # NOTE: the order of ``target_hand_links`` is the same as the order of ``mano_hand_idx``
        target_hand_links=[
            "right_thumb_fingertip",
            "right_index_fingertip",
            "right_middle_fingertip",
            "right_ring_fingertip",
            "right_pinky_fingertip",
        ],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


class SharpaWave_L(_SharpaWave):
    file: str = "sharpa_hand/left_sharpa_ha4_v2_1.urdf"

    # Same Eden-default ``up`` / ``front`` as SharpaWave_R: fingers along
    # URDF +Z, palm normal along URDF +X. The URDF is mirrored about
    # the XZ plane vs the right hand (thumb on -Y), but the chirality
    # flip leaves both the finger axis and the palm-normal axis on the
    # same URDF signs — so no override needed here either.

    dofs_name: list[str] = [
        "left_thumb_CMC_FE",
        "left_thumb_CMC_AA",
        "left_thumb_MCP_FE",
        "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_index_MCP_FE",
        "left_index_MCP_AA",
        "left_index_PIP",
        "left_index_DIP",
        "left_middle_MCP_FE",
        "left_middle_MCP_AA",
        "left_middle_PIP",
        "left_middle_DIP",
        "left_ring_MCP_FE",
        "left_ring_MCP_AA",
        "left_ring_PIP",
        "left_ring_DIP",
        "left_pinky_CMC",
        "left_pinky_MCP_FE",
        "left_pinky_MCP_AA",
        "left_pinky_PIP",
        "left_pinky_DIP",
    ]

    links_to_keep: list[str] = [
        "left_thumb_fingertip",
        "left_index_fingertip",
        "left_middle_fingertip",
        "left_ring_fingertip",
        "left_pinky_fingertip",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        # NOTE: the order of ``target_hand_links`` is the same as the order of ``mano_hand_idx``
        target_hand_links=[
            "left_thumb_fingertip",
            "left_index_fingertip",
            "left_middle_fingertip",
            "left_ring_fingertip",
            "left_pinky_fingertip",
        ],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )
