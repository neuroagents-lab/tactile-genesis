"""Recorder term capturing per-step actions."""

from __future__ import annotations

import torch

from eden.managers.recorder_manager import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class ActionRecorder(RecorderTerm):
    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        return {"action": self._env.action_manager.action.clone()}
