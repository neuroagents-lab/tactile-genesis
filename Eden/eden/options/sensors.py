"""Sensor and multi-sensor configuration options."""

import genesis as gs
from pydantic import Field, model_validator

from eden.options.options import ConfigurableOptions


class SensorOptions(ConfigurableOptions):
    """Configuration for a single sensor.

    Parameters
    ----------
    sensor: gs.sensors.SensorOptions
        The Genesis sensor options.
    attach_entity_name : str
        Name of the entity to attach the sensor to. Empty for static sensors.
    attach_link_name : str
        Name of the link to attach the sensor to.
    track_link_names : list[str]
        Resolved at env build into the nested sensor's ``track_link_idx`` (global link indices)
        when that sensor model defines ``track_link_idx``. Each entry is a scene ``entity_name`` (all
        links on that entity) or ``entity_name/link_name`` (one link). See ``Sensor`` for details.

    Examples
    --------
    IMU sensor attached to a robot link
    >>> sensor_options = SensorOptions(
    ...     sensor=gs.sensors.IMU(
    ...         acc_noise=(0.01, 0.01, 0.01),
    ...     ),
    ...     attach_entity_name="robot",
    ...     attach_link_name="base_link",
    ... )
    """

    sensor: gs.sensors.SensorOptions | None = None
    attach_entity_name: str = ""
    attach_link_name: str = ""
    track_link_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_track_link_idx_placeholder(self):
        """Create build-time placeholders so Genesis validators accept named-link sensors."""
        names = list(self.track_link_names)
        if not names or self.sensor is None:
            return self
        sensor_cls = type(self.sensor)
        if "track_link_idx" not in getattr(sensor_cls, "model_fields", {}):
            raise ValueError(f"{sensor_cls.__name__} has no track_link_idx field; remove track_link_names.")
        n = len(names)
        cur = tuple(self.sensor.track_link_idx)
        # ``track_link_idx`` is resolved from ``track_link_names`` during env build. Preserve any
        # non-placeholder values if this model is revalidated after build or after an in-process reload.
        if len(cur) != n and not any(idx != 0 for idx in cur):
            self.sensor = self.sensor.model_copy(update={"track_link_idx": tuple(0 for _ in range(n))})
        return self


class SensorsOptions(ConfigurableOptions):
    """
    Container for multiple named sensors in an environment.

    Parameters
    ----------
    <sensor_name> : SensorOptions
        Named sensor configurations. Each sensor is accessible via its name
        in the environment's sensors dictionary.

    Examples
    --------
    >>> sensors_options = SensorsOptions(
    ...     contact_sensor=SensorOptions(
    ...         sensor=gs.sensors.Contact(),
    ...         attach_entity_name="robot",
    ...         attach_link_name="end_effector",
    ...     ),
    ...     imu_sensor=SensorOptions(
    ...         sensor=gs.sensors.IMU(acc_noise=(0.01, 0.01, 0.01)),
    ...         attach_entity_name="robot",
    ...     ),
    ... )
    """

    def model_post_init(self, context):
        super().model_post_init(context)

        for key in self.keys():
            val = getattr(self, key)
            assert isinstance(val, SensorOptions), (
                f"Invalid property `{key}` in SensorsOptions. All attributes should be an instance of SensorOptions."
            )
