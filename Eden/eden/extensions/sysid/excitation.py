"""Excitation signal generators (chirp, PRBS, playback)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Excitation(ABC):
    """Generator of per-DOF position commands used to excite the robot for sysid.

    Implementations return a length-``num_dofs`` array at each call. The
    driver (``DeploymentRecorder``) adds this to the robot's default pose
    before sending to hardware.
    """

    num_dofs: int

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """Return the commanded position offset at time ``t`` (seconds)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def duration(self) -> float:
        """Total excitation duration in seconds."""
        raise NotImplementedError


class ChirpExcitation(Excitation):
    """Linear-swept sine — the canonical sysid input for linear-system ID.

    Produces ``amplitude * sin(2*pi * phase(t))`` where the instantaneous
    frequency ramps from ``f_start`` to ``f_end`` over ``duration``. When
    ``dof_indices`` lists multiple DOFs, each is phase-shifted by
    ``stagger_phase`` * (idx / num_active) so simultaneous excitation
    doesn't create degenerate rank-1 inputs.
    """

    def __init__(
        self,
        num_dofs: int,
        dof_indices: Sequence[int],
        f_start: float,
        f_end: float,
        duration: float,
        amplitude: float | Sequence[float],
        stagger_phase: float = 0.0,
    ) -> None:
        self.num_dofs = num_dofs
        self.dof_indices = np.asarray(list(dof_indices), dtype=np.int64)
        self.f_start = float(f_start)
        self.f_end = float(f_end)
        self._duration = float(duration)
        self.stagger_phase = float(stagger_phase)
        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.full(self.dof_indices.size, float(amp[0]))
        if amp.size != self.dof_indices.size:
            raise ValueError("amplitude size must be 1 or len(dof_indices).")
        self.amplitude = amp

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> np.ndarray:
        t = max(0.0, min(t, self._duration))
        # Linear frequency sweep: phase(t) = f0*t + 0.5*k*t^2, k = (f1-f0)/T.
        if self._duration > 0:
            k = (self.f_end - self.f_start) / self._duration
        else:
            k = 0.0
        phase = self.f_start * t + 0.5 * k * t * t
        n = self.dof_indices.size
        shifts = self.stagger_phase * (np.arange(n) / max(n, 1))
        offsets = np.zeros(self.num_dofs, dtype=np.float64)
        offsets[self.dof_indices] = self.amplitude * np.sin(2 * np.pi * phase + shifts)
        return offsets


class PRBSExcitation(Excitation):
    """Pseudo-random binary sequence — good for stiction / friction characterisation.

    At each boundary of length ``period`` the command flips between
    ``+amplitude`` and ``-amplitude``. A different seed produces an
    independent sequence per DOF, which avoids correlated excitation.
    """

    def __init__(
        self,
        num_dofs: int,
        dof_indices: Sequence[int],
        period: float,
        duration: float,
        amplitude: float | Sequence[float],
        seed: int = 0,
    ) -> None:
        self.num_dofs = num_dofs
        self.dof_indices = np.asarray(list(dof_indices), dtype=np.int64)
        self.period = float(period)
        self._duration = float(duration)
        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.full(self.dof_indices.size, float(amp[0]))
        if amp.size != self.dof_indices.size:
            raise ValueError("amplitude size must be 1 or len(dof_indices).")
        self.amplitude = amp
        n_flips = max(int(np.ceil(self._duration / max(self.period, 1e-6))) + 1, 1)
        rng = np.random.default_rng(seed)
        self._flips = rng.choice([-1.0, 1.0], size=(n_flips, self.dof_indices.size))

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> np.ndarray:
        idx = int(t / max(self.period, 1e-6))
        idx = max(0, min(idx, self._flips.shape[0] - 1))
        offsets = np.zeros(self.num_dofs, dtype=np.float64)
        offsets[self.dof_indices] = self.amplitude * self._flips[idx]
        return offsets


class PlaybackExcitation(Excitation):
    """Replay a pre-recorded trajectory (e.g. captured earlier in sim).

    ``offsets`` is ``(n_steps, num_dofs)`` of **offsets from default pose**
    sampled at ``dt``. At arbitrary ``t`` the nearest earlier sample is
    returned (zero-order hold) — robust to clock skew between generator
    and deployment loop.
    """

    def __init__(self, offsets: np.ndarray, dt: float) -> None:
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.ndim != 2:
            raise ValueError("offsets must be 2-D (n_steps, num_dofs).")
        self.num_dofs = int(offsets.shape[1])
        self._offsets = offsets
        self._dt = float(dt)

    @property
    def duration(self) -> float:
        return self._offsets.shape[0] * self._dt

    def __call__(self, t: float) -> np.ndarray:
        idx = int(t / max(self._dt, 1e-9))
        idx = max(0, min(idx, self._offsets.shape[0] - 1))
        return self._offsets[idx]
