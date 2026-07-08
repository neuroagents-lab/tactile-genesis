"""Command manager for generating and updating commands."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from eden.managers.base import ManagerBase, ManagerTermBase
from eden.options.managers.commands import CommandManagerOptions, CommandTermOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import sanitize_envs_idx
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class CommandTerm(ManagerTermBase, ConfigurableMixin[CommandTermOptions]):
    """
    Base class for command terms.

    Parameters
    ----------
    resampling_time_range: tuple[float, float]
        The range of time before commands are changed [s].
    debug_vis: bool
        Whether to visualize debug information. Defaults to False.
    """

    resampling_time_range: tuple[float, float] = None
    debug_vis: bool = False

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

        self.stats = dict()
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    @property
    @abstractmethod
    def command(self):
        raise NotImplementedError

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, float]:
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs)
        extras = {}
        for stat_name, stat_value in self.stats.items():
            selected = stat_value[envs_idx]
            extras[stat_name] = (selected.sum() / max(selected.numel(), 1)).item()
            stat_value[envs_idx] = 0.0
        self.command_counter[envs_idx] = 0
        self._resample(envs_idx)
        return extras

    def compute(self, dt: float) -> None:
        self.time_left -= dt
        resample_envs_mask = self.time_left <= 0.0
        # NOTE: always call _resample unconditionally to avoid a blocking
        # GPU→CPU sync from .any(). Masked ops are no-ops for all-false masks.
        self._resample(resample_envs_mask)
        self._update_command()

    def _resample(self, envs_idx: slice | torch.Tensor) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return

        self.time_left[envs_idx] = self.time_left[envs_idx].uniform_(*self.resampling_time_range)
        self._resample_command(envs_idx)
        self.command_counter[envs_idx] += 1

    @abstractmethod
    def _resample_command(self, envs_idx: slice | torch.Tensor) -> None:
        """Resample the command for the specified environments."""
        raise NotImplementedError

    @abstractmethod
    def _update_command(self) -> None:
        """Update the command based on the current state."""
        raise NotImplementedError

    def draw_vis(self) -> None:
        pass


COMMAND_TERM_REGISTRY = Registry("COMMAND_TERM")


class CommandManager(ManagerBase[CommandManagerOptions]):
    def __init__(self, env: EnvBase, options: CommandManagerOptions):
        super().__init__(env=env, options=options)

        self._commands = dict()

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Command Terms",
            field_names=["Index", "Name", "Type"],
            rows=([index, name, term.__class__.__name__] for index, (name, term) in enumerate(self._terms.items())),
            align={"Name": "l"},
        )

    def draw_vis(self) -> None:
        for term in self._terms.values():
            if term.debug_vis:
                term.draw_vis()

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, torch.Tensor]:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        extras = {}
        for name, term in self._terms.items():
            stats = term.reset(envs_idx=envs_idx)
            for stat_name, stat_value in stats.items():
                extras[f"Commands/{name}/{stat_name}"] = stat_value
        return extras

    def compute(self, dt: float) -> None:
        for term in self._terms.values():
            term.compute(dt)

    def get_command(self, name: str) -> torch.Tensor:
        """Return the command for the specified command term."""
        return self.get_term(name).command

    def _prepare_terms(self):
        self._terms: dict[str, CommandTerm] = dict()

        for term_name in self._options.term_keys():
            term: CommandTerm = self._build_term(term_name, COMMAND_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
