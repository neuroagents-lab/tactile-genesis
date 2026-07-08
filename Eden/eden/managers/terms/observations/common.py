"""Common observation terms (commands, last action, episode phase)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers import OBSERVATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.action_manager import ActionTerm


@OBSERVATION_TERM_REGISTRY.register()
def generated_commands(
    env: RLEnvBase,
    *,
    command_name: str,
) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


@OBSERVATION_TERM_REGISTRY.register()
def last_action(
    env: RLEnvBase,
    *,
    action_name: str | None = None,
) -> torch.Tensor:
    if action_name is not None:
        action_term: ActionTerm = env.action_manager.get_term(action_name)
        return action_term.action
    return env.action_manager.action


@OBSERVATION_TERM_REGISTRY.register()
def episode_phase(
    env: RLEnvBase,
    *,
    centered: bool = False,
) -> torch.Tensor:
    """Return normalized episode time as a ``(num_envs, 1)`` tensor.

    ``centered=False`` (default): ``t / max_episode_length`` in ``[0, 1]``.
    ``centered=True``: ``2 * t / max_episode_length - 1`` in ``[-1, 1]``.
    """
    phase = env.episode_length_buf.float() / env.max_episode_length
    if centered:
        phase = 2.0 * phase - 1.0
    return phase.unsqueeze(-1)
