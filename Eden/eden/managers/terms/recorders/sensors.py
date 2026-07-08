"""Recorder term capturing sensor readings."""

from __future__ import annotations

import torch

from eden.managers import RECORDER_TERM_REGISTRY, RecorderTerm


def _to_recordable_dict(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _to_recordable_dict(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return {f"item_{idx}": _to_recordable_dict(val) for idx, val in enumerate(value)}
    try:
        return torch.as_tensor(value).clone()
    except Exception:
        return None


@RECORDER_TERM_REGISTRY.register()
class SensorRecorder(RecorderTerm):
    """Record sensor outputs during simulation."""

    use_ground_truth: bool = False
    record_post_reset_snapshot: bool = True

    def _read_sensors(self, envs_idx=None) -> dict[str, torch.Tensor | dict]:
        if not self._env.sensors:
            return {}

        sensor_records: dict[str, torch.Tensor | dict] = {}
        for sensor_name, sensor in self._env.sensors.items():
            sensor_value = (
                sensor.read_ground_truth(envs_idx=envs_idx) if self.use_ground_truth else sensor.read(envs_idx=envs_idx)
            )
            recordable = _to_recordable_dict(sensor_value)
            if recordable is not None:
                sensor_records[sensor_name] = recordable

        if not sensor_records:
            return {}
        return {"sensor": sensor_records}

    def record_post_reset(self, envs_idx=None) -> dict[str, torch.Tensor | dict]:
        if not self.record_post_reset_snapshot:
            return {}
        return self._read_sensors(envs_idx=envs_idx)

    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        return self._read_sensors()
