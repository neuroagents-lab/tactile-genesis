"""Recorder term capturing data for system identification."""

from __future__ import annotations

import torch

from eden.managers.recorder_manager import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class SysIDRecorder(RecorderTerm):
    """Record the proprioceptive trace needed to rebuild a sysid Trajectory.

    Captures the signals under the ``sysid/`` namespace so that
    ``eden.extensions.sysid.Trajectory.from_recorder_episode`` can reload
    an episode without any extra metadata.

    Parameters
    ----------
    entity_name: str
        Entity to record. Must exist in ``env.entities``.
    include_torque: bool
        Record ``get_dofs_control_force()`` under ``sysid/dofs_control_force``.
    include_base: bool
        Record floating-base quat / ang_vel. Set False for fixed-base arms.
    """

    entity_name: str = "robot"
    include_torque: bool = True
    include_base: bool = True

    def record_post_reset(self, envs_idx=None) -> dict[str, torch.Tensor | dict]:
        entity = self._env.entities[self.entity_name]
        initial_state: dict[str, torch.Tensor] = {
            "qpos": entity.get_qpos(qs_idx_local=entity.qs_idx_local, envs_idx=envs_idx).clone(),
            "dofs_vel": entity.get_dofs_vel(envs_idx=envs_idx).clone(),
        }
        if self.include_base:
            initial_state["pos"] = entity.get_pos(envs_idx=envs_idx).clone()
            initial_state["quat"] = entity.get_quat(envs_idx=envs_idx).clone()
        return {"sysid": {"initial_state": initial_state}}

    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        action = self._env.action_manager.action.clone()
        stamp = (self._env.episode_length_buf.float() * self._env.dt).unsqueeze(-1)
        return {"sysid": {"action": action, "stamp": stamp}}

    def record_post_step(self) -> dict[str, torch.Tensor | dict]:
        entity = self._env.entities[self.entity_name]
        out: dict[str, torch.Tensor] = {
            "dofs_pos": entity.get_dofs_pos().clone(),
            "dofs_vel": entity.get_dofs_vel().clone(),
        }
        if self.include_torque:
            out["dofs_control_force"] = entity.get_dofs_control_force().clone()
        if self.include_base:
            out["base_quat"] = entity.get_quat().clone()
            out["base_ang_vel"] = entity.get_ang(frame="body").clone()
        return {"sysid": out}
