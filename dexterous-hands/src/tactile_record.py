"""Record per-step tactile sensor readings during a play episode and save them to an .npz.

``main.py --mode=play --save_tactile`` builds a :class:`TactileEpisodeRecorder`,
calls :meth:`TactileEpisodeRecorder.record` once per simulation step, and finally
:meth:`TactileEpisodeRecorder.save`. The resulting .npz holds the per-step tactile
field for offline analysis.

For every ``tactile_*`` sensor on the env this captures, per step and for one env:
  - the world-frame probe positions (probes move with the hand), and
  - the per-probe reading -- a 3D vector for the vector sensors (elastomer /
    force_torque / force / proximity) or a scalar for the contact sensors
    (bool / depth / agg_force / agg_bool / link_bool / link_force).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import genesis.utils.geom as gu
import numpy as np
import torch

from tactile_sensors import spec_for_sensor_name

# Sensor types whose per-probe reading is a 3D vector (the rest read a scalar).
# Mirrors the vector-field branch of ``scripts/sensors_viewer.py``.
_VECTOR_SENSOR_TYPES: frozenset[str] = frozenset({"elastomer", "force_torque", "force", "proximity"})


def _discover_tactile_sensors(env: Any) -> dict[str, str]:
    """Map ``tactile_*`` sensor name -> sensor type, for every tactile sensor on the env."""
    sensors = getattr(env, "sensors", {}) or {}
    out: dict[str, str] = {}
    for name in sorted(sensors):
        if not name.startswith("tactile_"):
            continue
        spec = spec_for_sensor_name(name)
        if spec is not None:
            out[name] = spec.name
    return out


def _env_slice(tensor: torch.Tensor, env_idx: int, *, is_vector: bool) -> torch.Tensor:
    """Pick env ``env_idx`` from a sensor reading, tolerating a missing env axis.

    Vector readings are ``([n_envs,] n_probes, 3)``; scalars are ``([n_envs,] n_probes)``.
    """
    min_ndim = 2 if is_vector else 1
    if tensor.ndim > min_ndim:
        tensor = tensor[env_idx]
    return tensor


class TactileEpisodeRecorder:
    """Buffers per-step tactile readings + probe positions and writes them to an .npz."""

    def __init__(self, env: Any, *, env_idx: int = 0, robot: str = "", task: str = "") -> None:
        self.env = env
        self.env_idx = int(env_idx)
        self.robot = str(robot)
        self.task = str(task)
        self._sensor_types: dict[str, str] = _discover_tactile_sensors(env)
        self.t: list[float] = []
        self._data: dict[str, list[np.ndarray]] = {n: [] for n in self._sensor_types}
        self._pos: dict[str, list[np.ndarray]] = {n: [] for n in self._sensor_types}
        # True once a sensor yields probe world positions; positions are then
        # expected every frame so the saved arrays stack cleanly.
        self._has_pos: dict[str, bool] = {}

    @property
    def sensor_names(self) -> list[str]:
        return list(self._sensor_types)

    # -- recording ---------------------------------------------------------

    def _link_pose(self, gs_sensor: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        """World ``(pos (3,), quat (4,))`` of the sensor's attached link, for env ``env_idx``."""
        link = getattr(gs_sensor, "_link", None)
        if link is None:
            return None
        link_pos = link.get_pos()
        link_quat = link.get_quat()
        if link_pos.ndim > 1:
            link_pos = link_pos[self.env_idx]
        if link_quat.ndim > 1:
            link_quat = link_quat[self.env_idx]
        return link_pos.reshape(3), link_quat.reshape(4)

    def _probe_world_pos(
        self, gs_sensor: Any, link_pose: tuple[torch.Tensor, torch.Tensor] | None
    ) -> np.ndarray | None:
        """World-frame probe positions ``(n_probes, 3)``, or ``None`` for link-attached sensors."""
        local = getattr(gs_sensor, "probe_local_pos", None)
        if local is None or link_pose is None:
            return None
        link_pos, link_quat = link_pose
        world = gu.transform_by_trans_quat(local.reshape(-1, 3), link_pos, link_quat)
        return world.detach().cpu().numpy().astype(np.float32)

    def record(self, t: float) -> None:
        """Read every tactile sensor once and append this step's sample."""
        self.t.append(float(t))
        for name, stype in self._sensor_types.items():
            sensor = self.env.sensors[name]
            gs_sensor = getattr(sensor, "_sensor", sensor)
            link_pose = self._link_pose(gs_sensor)
            is_vector = stype in _VECTOR_SENSOR_TYPES

            data = sensor.read()
            if is_vector:
                # ElastomerTaxel.read() returns the displacement tensor directly;
                # the others return a NamedTuple whose `.force` field holds it.
                raw = data if stype == "elastomer" else data.force
                vec = _env_slice(raw, self.env_idx, is_vector=True).reshape(-1, 3).float()
                if link_pose is not None:
                    # Rotate link-local vectors into the world frame to match probe positions.
                    vec = gu.transform_by_quat(vec, link_pose[1])
                sample = vec.detach().cpu().numpy().astype(np.float32)
            else:
                raw = data[0] if isinstance(data, tuple) else data
                scalar = _env_slice(raw, self.env_idx, is_vector=False).reshape(-1).float()
                sample = scalar.detach().cpu().numpy().astype(np.float32)
            self._data[name].append(sample)

            pos = self._probe_world_pos(gs_sensor, link_pose)
            if pos is not None:
                self._pos[name].append(pos)
                self._has_pos[name] = True

    # -- saving ------------------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Write all buffered samples to a compressed .npz and return the path.

        Layout: ``t`` ``(T,)``, parallel string arrays ``sensor_names`` /
        ``sensor_types`` / ``sensor_kinds``, plus per sensor ``<name>__data``
        ``(T, n_probes[, 3])`` and (when available) ``<name>__pos`` ``(T, n_probes, 3)``.
        """
        arrays: dict[str, np.ndarray] = {
            "t": np.asarray(self.t, dtype=np.float32),
            "robot": np.array(self.robot),
            "task": np.array(self.task),
        }
        names = [n for n in self.sensor_names if self._data[n]]
        arrays["sensor_names"] = np.array(names)
        arrays["sensor_types"] = np.array([self._sensor_types[n] for n in names])
        arrays["sensor_kinds"] = np.array(
            ["vector" if self._sensor_types[n] in _VECTOR_SENSOR_TYPES else "scalar" for n in names]
        )
        for name in names:
            arrays[f"{name}__data"] = np.stack(self._data[name], axis=0)
            if self._has_pos.get(name):
                arrays[f"{name}__pos"] = np.stack(self._pos[name], axis=0)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return str(path)
