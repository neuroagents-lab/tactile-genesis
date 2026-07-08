"""Configuration options for system-identification extensions."""

from __future__ import annotations

from typing import Literal, Sequence

from eden.options.options import ConfigurableOptions


DofProperty = Literal["damping", "armature", "stiffness", "frictionloss", "kp", "kd"]

SignalName = Literal[
    "dofs_pos",
    "dofs_vel",
    "dofs_torque",
    "base_quat",
    "base_ang_vel",
]
# Note: ``base_lin_acc`` is intentionally absent. The Trajectory dataclass
# can carry it (deployment recordings include IMU lin_acc), but the sysid
# rollout can't reproduce it from the rigid-body sim — there's no
# corresponding entity getter — so accepting it as a residual signal would
# only crash at runtime.


class SystemIdentificationOptions(ConfigurableOptions):
    """
    Options for sysid extensions.

    Parameters
    ----------
    entity_name: str
        Name of the entity being identified (must exist in the scene).
    signals: Sequence[SignalName]
        Subset of signals included in the residual. Defaults to proprioception.
    signal_weights: dict[str, float]
        Per-signal multiplicative weights. Missing signals default to 1.0.
    normalize: bool
        Divide each signal block by its measured RMS before weighting.
    max_iters: int
        Maximum optimizer iterations / evaluations.
    verbose: bool
        Log per-iteration progress.
    """

    entity_name: str = "robot"
    signals: Sequence[SignalName] = (
        "dofs_pos",
        "dofs_vel",
        "dofs_torque",
    )
    signal_weights: dict[str, float] = {}
    normalize: bool = True
    max_iters: int = 200
    verbose: bool = True
