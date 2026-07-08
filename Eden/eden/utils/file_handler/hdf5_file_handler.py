"""HDF5 dataset file handler."""

import json
import os
import warnings
from collections.abc import Iterable

import numpy as np
import torch

from eden.utils.file_handler.base import (
    DEMO_NAME_RE as _DEMO_NAME_RE,
    EPISODE_ATTRS as _EPISODE_ATTRS,
    FILE_HANDLER_REGISTRY,
    FileHandlerBase,
)
from eden.utils.file_handler.episode_data import EpisodeData

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


@FILE_HANDLER_REGISTRY.register()
class HDF5FileHandler(FileHandlerBase):
    """HDF5 dataset file handler for storing and loading episode data.

    The format of the stored structure
    dataset_file.hdf5
    └── data/                     # _hdf5_data_group
        ├── demo_0/               # Episode group
        │   ├── actions           # Dataset (Tensor saved as NumPy array)
        │   ├── states            # Nested dict -> datasets
        │   │   └── obs           # Dataset
        │   └── ...               # Any other keys
        │   ├── attrs:
        │   │   seed: int
        │   │   success: bool
        │   │   env_id: int
        ├── demo_1/
        │   └── ...
        └── ...                   # Additional episodes
    attrs of data group:
        env_cfg: str (JSON)        # Serialized environment configuration
    """

    def __init__(self):
        """Initialize the HDF5 dataset file handler."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is not installed, please install it with `pip install h5py`")
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        # _next_demo_id is the next writable demo index; _demo_count is the actual
        # episode count. They diverge when a resumed dataset has gaps in numbering
        # (e.g. demo_0, demo_2 → next_id=3, count=2).
        self._next_demo_id = 0
        self._demo_count = 0

    def resolve_path(self, file_path: str) -> str:
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        return file_path

    def open(self, file_path: str, mode: str = "r", env_cfg: dict | None = None):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        file_path = self.resolve_path(file_path)
        self._hdf5_file_stream = h5py.File(file_path, mode)

        if "data" not in self._hdf5_file_stream:
            raise RuntimeError("Invalid dataset file: missing 'data' group")

        self._hdf5_data_group = self._hdf5_file_stream["data"]
        # Track the next writable demo id separately from the episode count so
        # gaps in numbering (e.g. demo_0, demo_2 after a crash) neither collide
        # on the next write nor over-report `get_num_episodes()`.
        existing_indices = [
            int(m.group(1)) for m in (_DEMO_NAME_RE.fullmatch(name) for name in self._hdf5_data_group.keys()) if m
        ]
        self._next_demo_id = max(existing_indices) + 1 if existing_indices else 0
        self._demo_count = len(existing_indices)

        if env_cfg is not None and "env_cfg" in self._hdf5_data_group.attrs:
            try:
                stored_cfg = json.loads(self._hdf5_data_group.attrs["env_cfg"])
            except (ValueError, json.JSONDecodeError):
                stored_cfg = None
            if stored_cfg is not None and stored_cfg != env_cfg:
                warnings.warn(
                    f"Resumed HDF5 dataset '{file_path}' was recorded with a different env_cfg; "
                    "mixing episodes from different configs may break downstream consumers.",
                    stacklevel=2,
                )

    def create(self, file_path: str, env_cfg: dict):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        file_path = self.resolve_path(file_path)
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        self._hdf5_file_stream = h5py.File(file_path, "w")

        # set up a data group in the file
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["env_cfg"] = json.dumps(env_cfg)
        self._hdf5_data_group.attrs["total"] = 0
        self._next_demo_id = 0
        self._demo_count = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- Properties ---

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return list(self._hdf5_data_group.keys())

    def get_num_episodes(self) -> int:
        self._raise_if_not_initialized()
        return self._demo_count

    @property
    def demo_count(self) -> int:
        self._raise_if_not_initialized()
        return self._demo_count

    # --- Operations ---

    def load_episode(self, episode_name: str, device: str = "cpu") -> EpisodeData | None:
        """Load episode data from the file."""
        self._raise_if_not_initialized()
        if episode_name not in self._hdf5_data_group:
            return None
        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]

        def load_dataset_helper(group):
            """Load a dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                if isinstance(group[key], h5py.Group):
                    data[key] = load_dataset_helper(group[key])
                else:
                    # Converting group[key] to numpy array greatly improves the performance
                    # when converting to torch tensor
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = load_dataset_helper(h5_episode_group)

        for attr in _EPISODE_ATTRS:
            if attr in h5_episode_group.attrs:
                value = h5_episode_group.attrs[attr]
                if attr == "success":
                    value = bool(value)
                elif attr in ("seed", "env_id"):
                    value = int(value)
                setattr(episode, attr, value)

        return episode

    def write_episode(self, episode: EpisodeData):
        """Add an episode to the dataset."""
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        episode_group_name = f"demo_{self._next_demo_id}"

        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the dataset")

        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        for attr in _EPISODE_ATTRS:
            val = getattr(episode, attr)
            if val is not None:
                h5_episode_group.attrs[attr] = val

        def create_dataset_helper(group, key, value):
            """Create a dataset that contains recursive dict objects."""
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                data = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                group.create_dataset(key, data=data, compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        self._next_demo_id += 1
        self._demo_count += 1
        self._hdf5_data_group.attrs["total"] = self._demo_count

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()
        self._hdf5_file_stream.flush()

    def close(self):
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None
            self._hdf5_data_group = None
            self._next_demo_id = 0
            self._demo_count = 0

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")
