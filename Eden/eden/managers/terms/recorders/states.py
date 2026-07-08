"""Recorder terms capturing initial and per-step entity states."""

from __future__ import annotations

import torch

from eden.managers import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class InitialStateRecorder(RecorderTerm):
    """Record per-episode initial entity states right after reset."""

    include_qpos: bool = True
    include_dofs_vel: bool = True
    include_pos_quat: bool = True

    def record_post_reset(self, envs_idx=None) -> dict[str, torch.Tensor | dict]:
        if not self._env.entities:
            return {}

        entities_state: dict[str, dict[str, torch.Tensor]] = {}
        for entity_name, entity in self._env.entities.items():
            entity_state: dict[str, torch.Tensor] = {}

            if self.include_qpos:
                entity_state["qpos"] = entity.get_qpos(qs_idx_local=entity.qs_idx_local, envs_idx=envs_idx).clone()
            if self.include_dofs_vel:
                entity_state["dofs_vel"] = entity.get_dofs_vel(envs_idx=envs_idx).clone()
            if self.include_pos_quat:
                entity_state["pos"] = entity.get_pos(envs_idx=envs_idx).clone()
                entity_state["quat"] = entity.get_quat(envs_idx=envs_idx).clone()

            if entity_state:
                entities_state[entity_name] = entity_state

        if not entities_state:
            return {}

        return {"initial_state": {"entities": entities_state}}


@RECORDER_TERM_REGISTRY.register()
class StateRecorder(RecorderTerm):
    """Record the state of the environment at the beginning of each step."""

    entity_name: str = "robot"
    include_qpos: bool = True
    include_dofs_vel: bool = True
    include_pos_quat: bool = True

    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        entity = self._env.entities[self.entity_name]
        entity_state: dict[str, torch.Tensor] = {}
        if self.include_qpos:
            entity_state["qpos"] = entity.get_qpos(qs_idx_local=entity.qs_idx_local).clone()
        if self.include_dofs_vel:
            entity_state["dofs_vel"] = entity.get_dofs_vel().clone()
        if self.include_pos_quat:
            entity_state["pos"] = entity.get_pos().clone()
            entity_state["quat"] = entity.get_quat().clone()
        return {"state": entity_state}
