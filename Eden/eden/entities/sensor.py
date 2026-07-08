"""Sensor entity wrapper over Genesis sensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.utils.geom import euler_to_quat

from eden.options.sensors import SensorOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.isaac_math import quat_apply, quat_mul

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class Sensor(ConfigurableMixin[SensorOptions]):
    """Wrapper around a Genesis sensor, configured from :class:`SensorOptions`.

    Parameters
    ----------
    sensor: gs.sensors.SensorOptions
        The Genesis sensor options.
    attach_entity_name : str
        Name of the entity to attach the sensor to. Empty for static sensors.
    attach_link_name : str
        Name of the link to attach the sensor to.
    track_link_names : list[str]
        Resolved into the nested sensor's ``track_link_idx`` (global link indices) if applicable to the sensor.
        Each entry is either:
        - ``entity_name`` - every link on that scene entity (Genesis link order) is tracked.
        - ``entity_name/link_name`` - a single link on that entity. ``link_name`` is the Genesis link
          name (with morph-prefix fallback if a short suffix is used, e.g. ``baselink`` → ``sphere_baselink``).
    """

    sensor_options: gs.sensors.SensorOptions
    attach_entity_name: str = ""
    attach_link_name: str = ""
    track_link_names: list[str] = []

    def __init__(self, env: EnvBase, options: SensorOptions):
        self._uid = gs.UID()
        self._options = options
        self._env = env
        self._sensor = None  # Will be set in pre_build()

        for name in self.get_parameter_names():
            if name in self._options.model_dump():
                setattr(self, name, getattr(self._options, name))
            else:
                setattr(self, name, getattr(self, name))

    def pre_build(self):
        # If sensor is attached to an entity, inject entity_idx and link_idx_local
        if self.attach_entity_name:
            if self.attach_entity_name not in self._env.entities:
                raise ValueError(
                    f"Sensor references non-existent entity: {self.attach_entity_name}. "
                    f"Available entities: {self._env.entities.keys()}"
                )
            entity = self._env.entities[self.attach_entity_name]
            self._options.sensor.entity_idx = entity.idx

            if self.attach_link_name:
                link = entity.get_link(self.attach_link_name)
                self._options.sensor.link_idx_local = link.idx_local
        elif self.attach_link_name:
            raise ValueError("Should provide attach_entity_name if attach_link_name is provided.")

        self._apply_track_link_indices_from_names()

        self._sensor = self._env.scene.add_sensor(self._options.sensor)

    def _apply_track_link_indices_from_names(self) -> None:
        """Set ``sensor.track_link_idx`` from link names when the Genesis sensor model defines that field."""
        so = self._options.sensor
        if so is None:
            return
        sensor_cls = type(so)
        if "track_link_idx" not in getattr(sensor_cls, "model_fields", {}):
            return

        names = list(self.track_link_names)
        if not names:
            return

        if not self.attach_entity_name:
            raise ValueError("track_link_names requires attach_entity_name to resolve link names.")

        attach_entity = self._env.entities[self.attach_entity_name]
        if not hasattr(attach_entity, "get_link"):
            raise TypeError(
                f"Entity '{self.attach_entity_name}' does not support link name lookup; cannot resolve track_link names."
            )

        idxs: list[int] = []
        for token in names:
            idxs.extend(self._expand_track_token(token))
        so.track_link_idx = tuple(idxs)

    def _expand_track_token(self, token: str) -> list[int]:
        """Map one ``track_link_names`` entry to solver-global link indices (``RigidLink.idx``)."""
        entities = self._env.entities
        if "/" in token:
            entity_name, _, link_part = token.partition("/")
            if entity_name not in entities:
                raise ValueError(
                    f"track_link_names entry {token!r}: unknown entity {entity_name!r}. "
                    f"Available: {list(entities.keys())}"
                )
            ent = entities[entity_name]
            return [self._get_rigid_link(ent, link_part).idx]

        if token in entities:
            ent = entities[token]
            links = getattr(ent._entity, "links", None) or []
            if not links:
                raise ValueError(f"track_link_names entry {token!r}: entity has no links.")
            return [link.idx for link in links]

        raise ValueError(
            f"track_link_names entry {token!r}: expected a scene entity name or entity_name/link_name. "
            f"Entities: {list(entities.keys())}"
        )

    def _get_rigid_link(self, entity, link_name: str):
        """Resolve ``link_name`` on ``entity``, with morph-prefix fallback for primitive root links."""
        if not hasattr(entity, "get_link"):
            raise TypeError(f"Entity {entity.name!r} does not support get_link.")
        try:
            return entity.get_link(link_name)
        except gs.GenesisException:
            links = getattr(entity._entity, "links", None) or []
            if not links:
                raise
            suffix = f"_{link_name}"
            matches = [link for link in links if link.name.endswith(suffix)]
            if len(matches) == 1:
                return matches[0]
            raise

    @property
    def is_attached(self) -> bool:
        """Whether this sensor is attached to an entity link."""
        return bool(self.attach_entity_name)

    def get_pos(self, envs_idx=None) -> torch.Tensor:
        """Get the world position of the sensor.

        For attached sensors, this computes `link_pos + quat_apply(link_quat, offset_pos)`.
        For static sensors, returns the initial position from sensor options.

        Returns
        -------
        pos : torch.Tensor
            World position of shape (num_envs, 3).
        """
        if not self.is_attached:
            pos = torch.tensor(self._options.sensor.pos_offset, dtype=torch.float32, device=self._env.device)
            return pos.unsqueeze(0).expand(self._env.num_envs, -1)

        link = self._sensor._link
        link_pos = link.get_pos(envs_idx)  # (num_envs, 3)
        link_quat = link.get_quat(envs_idx)  # (num_envs, 4) wxyz

        offset_pos = self._sensor._shared_metadata.offsets_pos[:, self._sensor._idx]  # (B, 3)
        if envs_idx is not None:
            offset_pos = offset_pos[envs_idx]

        return link_pos + quat_apply(link_quat, offset_pos)

    def get_quat(self, envs_idx=None) -> torch.Tensor:
        """Get the world orientation of the sensor as a quaternion (wxyz).

        For attached sensors, this computes `quat_mul(link_quat, offset_quat)`.
        For static sensors, returns the initial orientation from sensor options.

        Returns
        -------
        quat : torch.Tensor
            World orientation quaternion (wxyz) of shape (num_envs, 4).
        """
        if not self.is_attached:
            quat_np = euler_to_quat([self._options.sensor.euler_offset])  # (1, 4)
            quat = torch.tensor(quat_np, dtype=torch.float32, device=self._env.device).reshape(4)
            return quat.unsqueeze(0).expand(self._env.num_envs, -1)

        link = self._sensor._link
        link_quat = link.get_quat(envs_idx)  # (num_envs, 4) wxyz

        offset_quat = self._sensor._shared_metadata.offsets_quat[:, self._sensor._idx]  # (B, 4)
        if envs_idx is not None:
            offset_quat = offset_quat[envs_idx]

        return quat_mul(link_quat, offset_quat)

    def set_pos_offset(self, pos_offset, envs_idx=None):
        """Set the positional offset of the sensor relative to the attached link.

        Parameters
        ----------
        pos_offset : array-like
            Position offset (x, y, z) in link-local frame.
        envs_idx : array-like, optional
            Environment indices. If None, applies to all environments.
        """
        if not self.is_attached:
            raise RuntimeError("Cannot set position offset on a static (unattached) sensor.")
        self._sensor.set_pos_offset(pos_offset, envs_idx=envs_idx)

    def set_quat_offset(self, quat_offset, envs_idx=None):
        """Set the rotational offset of the sensor relative to the attached link.

        Parameters
        ----------
        quat_offset : array-like
            Quaternion offset (wxyz) in link-local frame.
        envs_idx : array-like, optional
            Environment indices. If None, applies to all environments.
        """
        if not self.is_attached:
            raise RuntimeError("Cannot set quaternion offset on a static (unattached) sensor.")
        self._sensor.set_quat_offset(quat_offset, envs_idx=envs_idx)

    def read(self, envs_idx=None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Read current sensor data (with noise, delay, and other effects if configured).

        Parameters
        ----------
        envs_idx : array-like, optional
            Environment indices to read from. If None, reads from all environments.

        Returns
        -------
        sensor_data: torch.Tensor | tuple[torch.Tensor, ...]
            The sensor data with noise if configured. The return format depends on the sensor type.
        """
        return self._sensor.read(envs_idx=envs_idx)

    def read_ground_truth(self, envs_idx=None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Read ground truth sensor data (without noise, delay, or other effects).

        Parameters
        ----------
        envs_idx : array-like, optional
            Environment indices to read from. If None, reads from all environments.

        Returns
        -------
        sensor_data: torch.Tensor | tuple[torch.Tensor, ...]
            Ground truth sensor data. The return format depends on the sensor type.
        """
        return self._sensor.read_ground_truth(envs_idx=envs_idx)

    @property
    def sensor(self):
        """Access the underlying Genesis sensor object."""
        return self._sensor
