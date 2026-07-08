"""RobotEra XHand1 dexterous hand configurations (left/right)."""

from typing import ClassVar

from eden.options.entities import RobotOptions, MetadataOptions


class _XHand1(RobotOptions):
    default_dofs_pos: dict[str, float] = {}

    default_dofs_kp: dict[str, float] = {
        "*": 50.0,
    }

    default_dofs_kd: dict[str, float] = {
        "*": 25.0,
    }


class XHand1_R(_XHand1):
    file: str = "robotera_xhand/xhand1_right.urdf"

    # Hand-axis convention via Eden's standard ``up`` / ``front`` fields:
    #   ``up``    = URDF axis the fingers extend along
    #   ``front`` = URDF axis the palm normal points along (= the URDF
    #               axis aligned with world +X after load)
    # XHand1_R already satisfies the Eden defaults (fingers along URDF
    # +Z = ``up``; palm normal along URDF +X = ``front``), so no override
    # is needed — the robot loads with R_load = identity. The
    # MANO-driven retargeter reads these declarations (and a matching
    # MANO-side declaration in ``dex_retargeter.calibrate``) and derives
    # the URDF↔MANO frame conversion via ``align_up_and_front``.

    dofs_name: list[str] = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]

    links_to_keep: list[str] = [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # NOTE: the order of ``target_hand_links`` is the same as the order of ``mano_hand_idx``
        target_hand_links=[
            "right_hand_thumb_rota_tip",
            "right_hand_index_rota_tip",
            "right_hand_mid_tip",
            "right_hand_ring_tip",
            "right_hand_pinky_tip",
        ],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


class XHand1_L(_XHand1):
    file: str = "robotera_xhand/xhand1_left.urdf"

    # Same Eden-default ``up`` / ``front`` as XHand1_R: fingers along
    # URDF +Z, palm normal along URDF +X. The URDF is mirrored about
    # the YZ plane vs the right hand, but the chirality flip leaves
    # both the finger axis and the palm-normal axis on the same URDF
    # signs — so no override needed here either.

    dofs_name: list[str] = [
        "left_hand_thumb_bend_joint",
        "left_hand_thumb_rota_joint1",
        "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint",
        "left_hand_index_joint1",
        "left_hand_index_joint2",
        "left_hand_mid_joint1",
        "left_hand_mid_joint2",
        "left_hand_ring_joint1",
        "left_hand_ring_joint2",
        "left_hand_pinky_joint1",
        "left_hand_pinky_joint2",
    ]

    links_to_keep: list[str] = [
        "left_hand_thumb_rota_tip",
        "left_hand_index_rota_tip",
        "left_hand_mid_tip",
        "left_hand_ring_tip",
        "left_hand_pinky_tip",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        # NOTE: the order of ``target_hand_links`` is the same as the order of ``mano_hand_idx``
        target_hand_links=[
            "left_hand_thumb_rota_tip",
            "left_hand_index_rota_tip",
            "left_hand_mid_tip",
            "left_hand_ring_tip",
            "left_hand_pinky_tip",
        ],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )
