"""Observation term that concatenates configured sensor readings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from eden.managers import OBSERVATION_TERM_REGISTRY
from eden.managers.observation_manager import ObservationTerm
from eden.utils.string import resolve_matching_names

if TYPE_CHECKING:
    pass


@OBSERVATION_TERM_REGISTRY.register()
class SensorRead(ObservationTerm):
    """Concatenate selected sensor readings into a single observation tensor.

    Sensors return a tensor or NamedTuple of tensors. The tensors are concatenated if needed.

    Parameters
    ----------
    sensor_names : list[str], optional
        A list of sensor names to include in the observation. If None, all sensors are included.
    read_ground_truth : bool, optional
        If True, read from the sensors' ground truth values instead of their noisy readings.
    post_process_func : Callable, optional
        An optional function to apply to each sensor reading before concatenation.
    """

    sensor_names: list[str] | None = None
    read_ground_truth: bool = False
    post_process_func: Callable | None = None

    def build(self) -> None:
        all_sensor_names = list(self._env.sensors.keys())
        if self.sensor_names is None:
            matched_sensor_names = all_sensor_names
        else:
            _, matched_sensor_names = resolve_matching_names(
                self.sensor_names,
                all_sensor_names,
                preserve_order=True,
            )
        self._sensors = [self._env.sensors[sensor_name] for sensor_name in matched_sensor_names]

    def compute(self, *args, **kwargs) -> torch.Tensor:
        sensor_readings = []
        for sensor in self._sensors:
            data = sensor.read_ground_truth() if self.read_ground_truth else sensor.read()
            if self.post_process_func is not None:
                data = self.post_process_func(data)
            if not isinstance(data, tuple):
                data = (data,)

            for tensor in data:
                if tensor.ndim > 2:
                    tensor = tensor.flatten(start_dim=1)
                sensor_readings.append(tensor.float())

        self._cached = torch.cat(sensor_readings, dim=-1)
        return self._cached
