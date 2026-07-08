"""LEAP hand configurations (left/right)."""

from typing import ClassVar

from eden.options.entities import RobotOptions, MetadataOptions


class _LeapHand(RobotOptions):
    default_dofs_pos: dict[str, float] = {}

    default_dofs_kp: dict[str, float] = {
        "*": 50.0,
    }

    default_dofs_kd: dict[str, float] = {
        "*": 25.0,
    }


# LEAP joint-name → physical-joint mapping (URDF keeps the upstream LEAP
# convention where joint names equal motor IDs 0..15 — same as the LEAP
# hardware driver and the dex-retargeting reference configs):
#   0  index  MCP side (adduction/abduction)
#   1  index  MCP forward (flexion)
#   2  index  PIP
#   3  index  DIP
#   4  middle MCP side
#   5  middle MCP forward
#   6  middle PIP
#   7  middle DIP
#   8  ring   MCP side
#   9  ring   MCP forward
#   10 ring   PIP
#   11 ring   DIP
#   12 thumb  CMC
#   13 thumb  MCP
#   14 thumb  DIP
#   15 thumb  IP (tip)


class LeapHand_R(_LeapHand):
    file: str = "leap_hand/leap_hand_right.urdf"

    # Hand-axis convention via Eden's standard ``up`` / ``front`` fields:
    #   ``up``    = URDF axis the fingers extend along
    #   ``front`` = URDF axis the palm normal points along (= the URDF
    #               axis aligned with world +X after load)
    # LEAP_R already satisfies the Eden defaults (in the ``base`` link
    # frame, FK at zero pose places index/middle/ring tips at z≈+0.23
    # — fingers along URDF +Z = ``up`` — and all tips at x≈+0.02 — palm
    # normal along URDF +X = ``front``), so the robot loads with
    # R_load = identity. The MANO-driven retargeter reads these
    # declarations (and a matching MANO-side declaration in
    # ``dex_retargeter.calibrate``) and derives the URDF↔MANO frame
    # conversion via ``align_up_and_front``.

    dofs_name: list[str] = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]

    links_to_keep: list[str] = [
        "thumb_tip_head",
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # NOTE: the order of ``target_hand_links`` is the same as the order of ``mano_hand_idx``.
        # LEAP has 4 fingers (no pinky), so we omit MANO index 20.
        target_hand_links=[
            "thumb_tip_head",
            "index_tip_head",
            "middle_tip_head",
            "ring_tip_head",
        ],
        mano_hand_idx=[16, 17, 18, 19],
    )


class LeapHand_L(_LeapHand):
    file: str = "leap_hand/leap_hand_left.urdf"

    # Same Eden-default ``up`` / ``front`` as LeapHand_R: fingers along
    # URDF +Z, palm normal along URDF +X (verified by FK at zero pose —
    # index/middle/ring tips at z≈+0.225, all tips at x≈+0.008). The
    # URDF mirrors the right hand about the XZ plane (thumb on -Y) but
    # the chirality flip leaves both the finger axis and the palm-normal
    # axis on the same URDF signs — so no override needed here either.

    dofs_name: list[str] = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]

    links_to_keep: list[str] = [
        "thumb_tip_head",
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        # NOTE: the order of ``target_hand_links`` is the same as the order of ``mano_hand_idx``.
        # LEAP has 4 fingers (no pinky), so we omit MANO index 20.
        target_hand_links=[
            "thumb_tip_head",
            "index_tip_head",
            "middle_tip_head",
            "ring_tip_head",
        ],
        mano_hand_idx=[16, 17, 18, 19],
    )
