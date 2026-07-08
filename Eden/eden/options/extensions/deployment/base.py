"""Base configuration options for real-robot deployment backends."""

from __future__ import annotations
from eden.options.options import ConfigurableOptions


class DeploymentOptions(ConfigurableOptions):
    """
    Options for deployment extensions.

    Parameters
    ----------
    entity_name: str
        The name of the entity to deploy
    control_dt: float | None
        The control time step in seconds.
        Defaults to None, in which case it is determined by the control frequency, fallback to sim dt * decimation.
    control_freq: float | None
        The control frequency in Hz. Defaults to None, in which case it is determined by 1/control_dt.
    decimation: int | None
        The decimation factor. Defaults to None, in which case sim decimation is used.
    sync: bool
        Whether to synchronize the control loop. Defaults to True.
    connect_timeout_s: float
        The timeout for connecting to the robot. Defaults to 5.0 seconds.
    auto_yaw_align: bool
        Capture the robot's IMU yaw at the end of ``reset`` and remove it from
        subsequent world-frame observations so the deployment world frame
        matches the (yaw-aligned) world frame the policy was trained in.
        Defaults to True. Disable when an external estimator already provides
        a yaw-consistent base orientation (e.g. mocap).
    """

    entity_name: str = "robot"
    control_dt: float | None = None
    control_freq: float | None = None
    decimation: int | None = None
    sync: bool = True
    connect_timeout_s: float = 5.0
    auto_yaw_align: bool = True
