"""Recorder term capturing RL step transitions."""

from __future__ import annotations

import torch

from eden.managers.recorder_manager import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class RLStepRecorder(RecorderTerm):
    """Record per-step RL buffers (reward, done, terminated, timeout, timestamp).

    Reads ``RLEnvBase`` step buffers after rewards and terminations are computed.
    Opt in by registering on ``recorder_options`` for RL tasks; not applicable to
    non-RL ``EnvBase`` flows (the buffers don't exist there).
    """

    include_reward: bool = True
    include_done: bool = True
    include_terminated: bool = True
    include_timeout: bool = True
    include_timestamp: bool = True

    def record_post_step(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.include_reward and hasattr(self._env, "reward_buf"):
            out["reward"] = self._env.reward_buf.unsqueeze(-1)
        if self.include_done and hasattr(self._env, "reset_buf"):
            out["done"] = self._env.reset_buf.to(dtype=torch.bool).unsqueeze(-1)
        if self.include_terminated and hasattr(self._env, "reset_terminated"):
            out["terminated"] = self._env.reset_terminated.to(dtype=torch.bool).unsqueeze(-1)
        if self.include_timeout and hasattr(self._env, "reset_timeouts"):
            out["timeout"] = self._env.reset_timeouts.to(dtype=torch.bool).unsqueeze(-1)
        if self.include_timestamp and hasattr(self._env, "episode_length_buf") and hasattr(self._env, "dt"):
            out["timestamp"] = self._env.episode_length_buf.to(dtype=torch.float32).unsqueeze(-1) * float(self._env.dt)
        return out
