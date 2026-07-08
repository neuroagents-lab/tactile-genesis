"""Base class and registry for dataset file handlers."""

from __future__ import annotations

import re
from abc import abstractmethod

from eden.options.file_handler import FileHandlerOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.file_handler.episode_data import EpisodeData
from eden.utils.registry import Registry


FILE_HANDLER_REGISTRY = Registry("FILE_HANDLER")

# Per-episode top-level attributes shared by all file-handler backends.
EPISODE_ATTRS: tuple[str, ...] = ("seed", "success", "env_id")
# Demo-group naming convention (``demo_<int>``) shared by all file-handler backends.
DEMO_NAME_RE = re.compile(r"^demo_(\d+)$")


class FileHandlerBase(ConfigurableMixin[FileHandlerOptions]):
    """Abstract class for handling dataset files."""

    @abstractmethod
    def resolve_path(self, file_path: str) -> str:
        """Return the concrete on-disk path used by this handler."""
        ...

    @abstractmethod
    def open(self, file_path: str, mode: str = "r", env_cfg: dict | None = None):
        """Open a file. If `env_cfg` is provided, implementations may warn on mismatch."""
        ...

    @abstractmethod
    def create(self, file_path: str, env_cfg: dict):
        """Create a new file."""
        ...

    @abstractmethod
    def write_episode(self, episode: EpisodeData):
        """Write episode data to the file."""
        ...

    @abstractmethod
    def flush(self):
        """Flush the file."""
        ...

    @abstractmethod
    def close(self):
        """Close the file."""
        ...

    @abstractmethod
    def load_episode(self, episode_name: str, device: str = "cpu") -> EpisodeData | None:
        """Load episode data from the file."""
        ...

    @abstractmethod
    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        ...
