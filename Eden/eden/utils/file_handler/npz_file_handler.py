"""NPZ dataset file handler."""

import json
import os
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

# Keys with these prefixes store metadata, not episode array data.
_META_PREFIX = "__meta__"
_ATTRS_SUFFIX = "/__attrs__"


def _encode_json_to_bytes(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)


def _decode_bytes_to_json(arr: np.ndarray):
    # Handle both new uint8 byte arrays and legacy np.void-scalar encodings.
    if arr.dtype.kind == "V":
        return json.loads(bytes(arr))
    return json.loads(bytes(np.asarray(arr, dtype=np.uint8)))


def _flatten_dict(d: dict, parent_key: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested dict of tensors/arrays into slash-separated keys."""
    items: dict[str, np.ndarray] = {}
    for key, value in d.items():
        new_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key))
        else:
            data = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            items[new_key] = data
    return items


def _unflatten_dict(flat: dict[str, np.ndarray], device: str = "cpu") -> dict:
    """Reconstruct a nested dict of tensors from slash-separated keys."""
    result: dict = {}
    for key, value in flat.items():
        parts = key.split("/")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = torch.tensor(value, device=device)
    return result


@FILE_HANDLER_REGISTRY.register()
class NPZFileHandler(FileHandlerBase):
    """NPZ dataset file handler for storing and loading episode data.

    Since NPZ files are flat key-value stores of numpy arrays, we use a
    naming convention to represent the hierarchical episode structure:

        dataset_file.npz
        ├── __meta__env_cfg              # JSON string stored as bytes array
        ├── __meta__total                # scalar: number of episodes
        ├── demo_0/actions               # episode data (slash-separated hierarchy)
        ├── demo_0/states/obs            # nested episode data
        ├── demo_0/__attrs__             # episode attrs as structured array
        ├── demo_1/actions
        └── ...
    """

    def __init__(self):
        self._file_path: str | None = None
        self._arrays: dict[str, np.ndarray] = {}
        # _next_demo_id is the next writable demo index; _demo_count is the actual
        # episode count. They diverge when a resumed dataset has gaps in numbering.
        self._next_demo_id: int = 0
        self._demo_count: int = 0
        self._writable: bool = False
        self._dirty: bool = False

    def resolve_path(self, file_path: str) -> str:
        if not file_path.endswith(".npz"):
            file_path += ".npz"
        return file_path

    def open(self, file_path: str, mode: str = "r", env_cfg: dict | None = None):
        if self._file_path is not None:
            raise RuntimeError("NPZ dataset file handler is already in use")

        file_path = self.resolve_path(file_path)
        with np.load(file_path, allow_pickle=False) as data:
            self._arrays = {k: np.array(v) for k, v in data.items()}
        self._file_path = file_path
        self._writable = mode != "r"
        self._dirty = False

        if f"{_META_PREFIX}total" not in self._arrays:
            raise RuntimeError("Invalid dataset file: missing metadata")

        # Track next writable demo id separately from episode count; gaps in
        # numbering shouldn't over-report `get_num_episodes()`.
        existing_indices = [
            int(m.group(1)) for m in (_DEMO_NAME_RE.fullmatch(name) for name in self._collect_episode_names()) if m
        ]
        self._next_demo_id = max(existing_indices) + 1 if existing_indices else 0
        self._demo_count = len(existing_indices)

        if env_cfg is not None and f"{_META_PREFIX}env_cfg" in self._arrays:
            try:
                stored_cfg = _decode_bytes_to_json(self._arrays[f"{_META_PREFIX}env_cfg"])
            except (ValueError, json.JSONDecodeError):
                stored_cfg = None
            if stored_cfg is not None and stored_cfg != env_cfg:
                import warnings

                warnings.warn(
                    f"Resumed NPZ dataset '{file_path}' was recorded with a different env_cfg; "
                    "mixing episodes from different configs may break downstream consumers.",
                    stacklevel=2,
                )

    def create(self, file_path: str, env_cfg: dict):
        if self._file_path is not None:
            raise RuntimeError("NPZ dataset file handler is already in use")
        file_path = self.resolve_path(file_path)
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        self._file_path = file_path
        self._writable = True
        self._arrays = {}
        self._dirty = True

        self._arrays[f"{_META_PREFIX}total"] = np.array(0)
        self._arrays[f"{_META_PREFIX}env_cfg"] = _encode_json_to_bytes(env_cfg)
        self._next_demo_id = 0
        self._demo_count = 0

    def _collect_episode_names(self) -> set[str]:
        return {key.split("/", 1)[0] for key in self._arrays if not key.startswith(_META_PREFIX)}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- Properties ---

    def get_episode_names(self) -> Iterable[str]:
        self._raise_if_not_initialized()
        return sorted(self._collect_episode_names())

    def get_num_episodes(self) -> int:
        self._raise_if_not_initialized()
        return self._demo_count

    @property
    def demo_count(self) -> int:
        self._raise_if_not_initialized()
        return self._demo_count

    # --- Operations ---

    def load_episode(self, episode_name: str, device: str = "cpu") -> EpisodeData | None:
        self._raise_if_not_initialized()

        prefix = f"{episode_name}/"
        attrs_key = f"{episode_name}{_ATTRS_SUFFIX}"

        # Collect all data arrays for this episode
        flat_data: dict[str, np.ndarray] = {}
        has_any = False
        for key, value in self._arrays.items():
            if not key.startswith(prefix):
                continue
            has_any = True
            # Skip the attrs key
            if key == attrs_key:
                continue
            # Strip episode name prefix
            sub_key = key[len(prefix) :]
            flat_data[sub_key] = value

        if not has_any:
            return None

        episode = EpisodeData()
        episode.data = _unflatten_dict(flat_data, device=device)

        # Load attributes
        if attrs_key in self._arrays:
            attrs = _decode_bytes_to_json(self._arrays[attrs_key])
            for attr in _EPISODE_ATTRS:
                if attr in attrs:
                    value = attrs[attr]
                    if attr == "success":
                        value = bool(value)
                    elif attr in ("seed", "env_id"):
                        value = int(value)
                    setattr(episode, attr, value)

        return episode

    def write_episode(self, episode: EpisodeData):
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        episode_name = f"demo_{self._next_demo_id}"

        # Check for duplicates
        prefix = f"{episode_name}/"
        if any(k.startswith(prefix) for k in self._arrays):
            raise ValueError(f"Episode '{episode_name}' already exists in the dataset")

        # Flatten and store episode data
        flat = _flatten_dict(episode.data, episode_name)
        self._arrays.update(flat)

        # Store episode attributes
        attrs = {}
        for attr in _EPISODE_ATTRS:
            val = getattr(episode, attr)
            if val is not None:
                attrs[attr] = val
        if attrs:
            self._arrays[f"{episode_name}{_ATTRS_SUFFIX}"] = _encode_json_to_bytes(attrs)

        self._next_demo_id += 1
        self._demo_count += 1
        self._arrays[f"{_META_PREFIX}total"] = np.array(self._demo_count)
        self._dirty = True

    def flush(self):
        # NPZ has no incremental write format, so flush rewrites the full archive.
        # We do this atomically (tmp file + rename) so a kill mid-write cannot leave
        # a partially-written dataset. Callers should prefer close() over flush()
        # per episode for large datasets (cost is O(total dataset size)).
        self._raise_if_not_initialized()
        if not self._writable or not self._dirty:
            return
        # np.savez_compressed auto-appends ".npz" when the path doesn't end with it,
        # so we keep the suffix and swap via os.replace.
        tmp_path = f"{self._file_path}.tmp.npz"
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **self._arrays)
        os.replace(tmp_path, self._file_path)
        self._dirty = False

    def close(self):
        if self._file_path is not None:
            if self._writable:
                self.flush()
            self._file_path = None
            self._arrays = {}
            self._next_demo_id = 0
            self._demo_count = 0
            self._writable = False
            self._dirty = False

    def _raise_if_not_initialized(self):
        if self._file_path is None:
            raise RuntimeError("NPZ dataset file handler is not initialized")
