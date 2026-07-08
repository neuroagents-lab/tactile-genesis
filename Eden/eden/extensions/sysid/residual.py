"""Signal-residual functions for system-identification objectives."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from eden.extensions.sysid.trajectory import Trajectory


def signal_residual(
    predicted: Mapping[str, np.ndarray],
    measured: Trajectory,
    signals: Sequence[str],
    weights: Mapping[str, float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Flatten per-signal residuals into a single 1-D vector.

    ``predicted[s]`` has shape ``(n_steps, dim)``. The measured value is read
    from ``measured.signal(s)`` with the same shape. Missing signals are
    skipped. When ``normalize=True``, each signal block is divided by its
    measured RMS so heterogeneous units contribute comparably.
    """
    weights = weights or {}
    pieces: list[np.ndarray] = []
    for s in signals:
        meas = measured.signal(s)
        if meas is None:
            continue
        pred = predicted.get(s)
        if pred is None:
            continue
        n = min(pred.shape[0], meas.shape[0])
        diff = pred[:n] - meas[:n]
        w = float(weights.get(s, 1.0))
        if w != 1.0:
            diff = diff * w
        if normalize:
            rms = float(np.sqrt(np.mean(meas[:n] ** 2)))
            if rms > 1e-8:
                diff = diff / rms
        pieces.append(diff.ravel())
    if not pieces:
        raise ValueError("No overlapping signals between predicted and measured.")
    return np.concatenate(pieces)


def multi_trajectory_residual(
    predicted_trajectories: Sequence[Mapping[str, np.ndarray]],
    measured_trajectories: Sequence[Trajectory],
    signals: Sequence[str],
    weights: Mapping[str, float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Concatenate residuals from multiple (predicted, measured) pairs."""
    if len(predicted_trajectories) != len(measured_trajectories):
        raise ValueError("predicted and measured trajectory lists have different lengths.")
    blocks = [
        signal_residual(p, m, signals=signals, weights=weights, normalize=normalize)
        for p, m in zip(predicted_trajectories, measured_trajectories)
    ]
    return np.concatenate(blocks)
