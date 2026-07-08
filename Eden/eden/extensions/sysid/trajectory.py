"""Trajectory container for system identification."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import numpy as np


_TIME_KEY = "sysid/stamp"
_ACTION_KEY = "sysid/action"
_SIGNAL_KEYS: dict[str, tuple[str, ...]] = {
    "dofs_pos": ("sysid/dofs_pos",),
    "dofs_vel": ("sysid/dofs_vel",),
    "dofs_torque": ("sysid/dofs_control_force", "sysid/dofs_force"),
    "base_quat": ("sysid/base_quat",),
    "base_ang_vel": ("sysid/base_ang_vel",),
    "base_lin_acc": ("sysid/base_lin_acc",),
}
_INITIAL_STATE_KEYS: tuple[str, ...] = ("qpos", "dofs_vel", "pos", "quat")


@dataclass
class Trajectory:
    """Measured-or-simulated trajectory for sysid.

    All arrays are 2-D ``(n_steps, dim)`` (``times`` is 1-D). Fields absent
    from the measurement are stored as None and skipped by the residual.

    Attributes
    ----------
    times: np.ndarray
        Time stamps, shape ``(n_steps,)``.
    action: np.ndarray
        Applied action / command, shape ``(n_steps, n_action_dim)``. This is
        the **pre-decimation** action that the action manager receives; the
        rollout re-injects this vector into ``env.step(action)``.
    dofs_pos, dofs_vel, dofs_torque:
        Per-DOF measurements, shape ``(n_steps, n_dofs)``. In dofs_name order.
    base_quat, base_ang_vel, base_lin_acc:
        Floating-base signals in wxyz / body-frame IMU convention. Each
        shape ``(n_steps, 3)`` (4 for quat). May be None for fixed-base robots.
    dof_names: tuple[str, ...]
        Names in matching column order with dofs_*.
    initial_state: dict[str, np.ndarray]
        Snapshot of the entity state at t=0 for rollout reset. Keys:
        ``qpos``, ``dofs_vel``, optionally ``pos``, ``quat``.
    extra: dict[str, Any]
        Free-form attachments (e.g. fingertip tactile, metadata).
    """

    times: np.ndarray
    action: np.ndarray | None = None
    dofs_pos: np.ndarray | None = None
    dofs_vel: np.ndarray | None = None
    dofs_torque: np.ndarray | None = None
    base_quat: np.ndarray | None = None
    base_ang_vel: np.ndarray | None = None
    base_lin_acc: np.ndarray | None = None
    dof_names: tuple[str, ...] = ()
    initial_state: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        n = self.times.shape[0]
        for key in ("action", "dofs_pos", "dofs_vel", "dofs_torque", "base_quat", "base_ang_vel", "base_lin_acc"):
            val = getattr(self, key)
            if val is None:
                continue
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[0] != n:
                raise ValueError(f"Trajectory.{key}: got {arr.shape[0]} rows, expected {n}.")
            setattr(self, key, arr)

    def __len__(self) -> int:
        return int(self.times.shape[0])

    @property
    def dt(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(self.times[1] - self.times[0])

    def signal(self, name: str) -> np.ndarray | None:
        return getattr(self, name, None)

    def save(self, path: str | pathlib.Path) -> None:
        payload: dict[str, np.ndarray] = {"times": self.times}
        for f in fields(self):
            if f.name in ("times", "dof_names", "initial_state", "extra"):
                continue
            val = getattr(self, f.name)
            if val is not None:
                payload[f.name] = val
        payload["dof_names"] = np.asarray(self.dof_names)
        for k, v in self.initial_state.items():
            payload[f"initial_state/{k}"] = np.asarray(v)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Trajectory":
        with np.load(path, allow_pickle=False) as npz:
            keys = set(npz.files)
            times = npz["times"]
            kwargs: dict[str, Any] = {"times": times}
            for name in ("action", "dofs_pos", "dofs_vel", "dofs_torque", "base_quat", "base_ang_vel", "base_lin_acc"):
                if name in keys:
                    kwargs[name] = npz[name]
            if "dof_names" in keys:
                kwargs["dof_names"] = tuple(str(s) for s in npz["dof_names"])
            initial_state = {k.split("/", 1)[1]: npz[k] for k in keys if k.startswith("initial_state/")}
            kwargs["initial_state"] = initial_state
        return cls(**kwargs)

    @classmethod
    def from_recorder_episode(
        cls,
        path: str | pathlib.Path,
        dof_names: Sequence[str] | None = None,
        demo: str | None = None,
    ) -> "Trajectory":
        """Load a Trajectory from a ``RecorderManager`` npz produced by ``SysIDRecorder``.

        Reads only the ``sysid/*`` key schema. ``NPZFileHandler`` writes one or
        more episodes under ``demo_{n}/...`` keys; pass ``demo="demo_0"`` to
        pick a specific episode, or leave it to auto-select the first one.
        A flat npz whose keys are already under ``sysid/...`` (no ``demo_``
        prefix) is also supported.
        """
        with np.load(path, allow_pickle=False) as npz:
            keys = set(npz.files)

            prefix = ""
            flat_stamp = _TIME_KEY
            if flat_stamp not in keys:
                # Recorder-manager output stores episodes under ``demo_N/``.
                demo_names = sorted({k.split("/", 1)[0] for k in keys if k.startswith("demo_")})
                if not demo_names:
                    raise ValueError(f"Recorder episode {path} is missing '{_TIME_KEY}'.")
                chosen = demo if demo is not None else demo_names[0]
                if chosen not in demo_names:
                    raise ValueError(f"{path} has no episode '{chosen}' (available: {demo_names}).")
                prefix = f"{chosen}/"
                if f"{prefix}{_TIME_KEY}" not in keys:
                    raise ValueError(f"Recorder episode {path}:{chosen} is missing '{_TIME_KEY}'.")

            # ``times`` may arrive as (n_steps,), (n_steps, 1), (n_steps, num_envs),
            # or (n_steps, num_envs, 1) depending on which recorder layout produced
            # the npz. Squeeze trailing singletons and pick env 0 explicitly so a
            # blind reshape(-1) can't interleave envs and steps for multi-env runs.
            raw_times = np.asarray(npz[f"{prefix}{_TIME_KEY}"])
            env_axis_size = raw_times.shape[1] if raw_times.ndim >= 2 else 1
            if raw_times.ndim == 1:
                times = raw_times
            elif raw_times.ndim == 2:
                times = raw_times[:, 0] if raw_times.shape[1] >= 1 else raw_times.reshape(-1)
            else:  # ndim >= 3
                times = raw_times[:, 0, 0] if raw_times.shape[1] >= 1 else raw_times.reshape(-1)
            n_steps = int(times.shape[0])

            def _take(key: str) -> np.ndarray | None:
                full = f"{prefix}{key}"
                if full not in keys:
                    return None
                arr = np.asarray(npz[full])
                # (n_steps, num_envs, dim) → take env 0; (n_steps, num_envs) where
                # num_envs > 1 → take env 0 column. Single-env (n_steps, 1) layouts
                # fall through unchanged and are 2-D-OK for the Trajectory dataclass.
                if arr.ndim == 3 and arr.shape[0] == n_steps:
                    arr = arr[:, 0, :]
                elif arr.ndim == 2 and env_axis_size > 1 and arr.shape == (n_steps, env_axis_size):
                    arr = arr[:, 0:1]
                return arr

            kwargs: dict[str, Any] = {"times": times}
            for name, candidates in _SIGNAL_KEYS.items():
                for candidate in candidates:
                    arr = _take(candidate)
                    if arr is not None:
                        kwargs[name] = arr
                        break
            action = _take(_ACTION_KEY)
            if action is not None:
                kwargs["action"] = action

            initial_state: dict[str, np.ndarray] = {}
            for key in _INITIAL_STATE_KEYS:
                full_key = f"{prefix}sysid/initial_state/{key}"
                if full_key not in keys:
                    continue
                arr = np.asarray(npz[full_key])
                if arr.ndim == 2:
                    arr = arr[0]
                initial_state[key] = arr
            kwargs["initial_state"] = initial_state

        if dof_names is not None:
            kwargs["dof_names"] = tuple(dof_names)
        return cls(**kwargs)
