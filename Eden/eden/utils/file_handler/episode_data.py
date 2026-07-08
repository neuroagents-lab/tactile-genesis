"""EpisodeData container for recorded trajectories."""

from __future__ import annotations

import torch


def _stack_recursive(data: dict, in_place: bool = True) -> dict:
    """Recursively stack lists into tensors."""
    target = data if in_place else {}
    for key, value in data.items():
        if isinstance(value, list):
            target[key] = torch.stack(value)
        elif isinstance(value, dict):
            target[key] = _stack_recursive(value, in_place)
        elif not in_place:
            target[key] = value
    return target


class EpisodeData:
    """Class to store episode data."""

    def __init__(self) -> None:
        self.data: dict = {}
        self.seed: int | None = None
        self.env_id: int | None = None
        self.success: bool | None = None
        self._next_index_dict: dict[str, int] = {}

    def is_empty(self):
        """Check if the episode data is empty."""
        return not bool(self.data)

    def _resolve_key(self, key: str):
        """Traverse nested dict by slash-separated key. Returns None if not found."""
        node = self.data
        for k in key.split("/"):
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node

    def add(self, key: str, value: torch.Tensor | dict):
        """Add a key-value pair to the dataset.

        The key can be nested by using the "/" character.
        For example: "obs/joint_pos".

        Parameters
        ----------
        key : str
            The key name.
        value : torch.Tensor or dict
            The corresponding value, either a tensor or a (possibly nested) dict.
        """
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add(f"{key}/{sub_key}", sub_value)
            return

        self._append_owned(key, value.clone())

    def _append_owned(self, key: str, value: torch.Tensor | dict):
        """Append a value the caller has already cloned, skipping a redundant copy.

        Used by the recorder commit path where per-env slices were already cloned
        into a pending buffer; calling `add` would clone them a second time.
        """
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._append_owned(f"{key}/{sub_key}", sub_value)
            return

        *parents, leaf = key.split("/")
        node = self.data
        for k in parents:
            node = node.setdefault(k, {})

        if leaf not in node:
            node[leaf] = [value]
        else:
            node[leaf].append(value)

    def get(self, key: str, index: int | None = None, sequential: bool = False):
        """General getter for any key in the episode data.

        Parameters
        ----------
        key : str
            Hierarchical key, e.g. ``'states/obs'``, ``'actions'`` or ``'rewards'``.
        index : int, optional
            Specific timestep to retrieve.
        sequential : bool, optional
            If True, use and increment a per-key pointer.

        Returns
        -------
        torch.Tensor or dict or None
            The tensor (or dict of tensors) at the requested timestep, or None if
            the key is missing or the index is out of range.

        Examples
        --------
        >>> # Random access
        >>> state5 = episode.get("states/obs/joint_pos", index=5)
        >>> action3 = episode.get("actions", index=3)
        >>> # Sequential access
        >>> next_state = episode.get("states/obs", sequential=True)
        >>> next_action = episode.get("actions", sequential=True)
        """
        data_pointer = self._resolve_key(key)
        if data_pointer is None:
            return None

        # Determine the index
        if sequential:
            idx = self._next_index_dict.get(key, 0)
            self._next_index_dict[key] = idx + 1
        elif index is not None:
            idx = index
        else:
            idx = 0

        # Access the data
        if isinstance(data_pointer, (list, torch.Tensor)):
            if idx >= len(data_pointer):
                return None
            return data_pointer[idx] if isinstance(data_pointer, list) else data_pointer[idx, ...]
        elif isinstance(data_pointer, dict):
            return {subkey: self.get(f"{key}/{subkey}", index=idx) for subkey in data_pointer}
        return None

    def size(self, key: str):
        """Return the size of the data at the given key.

        Parameters
        ----------
        key : str
            Hierarchical key, e.g. ``'states/obs'``, ``'actions'`` or ``'rewards'``.

        Returns
        -------
        int or dict or None
            Size of the stored value, a dict of sizes for nested keys, or None if
            the key is missing.
        """
        data_pointer = self._resolve_key(key)
        if data_pointer is None:
            return None

        if isinstance(data_pointer, (list, torch.Tensor)):
            return len(data_pointer)
        elif isinstance(data_pointer, dict):
            return {subkey: self.size(f"{key}/{subkey}") for subkey in data_pointer}
        return 0

    def reset(self, clear_data: bool = False):
        self._next_index_dict = {}
        if clear_data:
            self.data = {}

    def pre_export(self):
        """Stack lists into tensors in-place. Destructive — do not call add() after this."""
        _stack_recursive(self.data, in_place=True)

    def pre_export_copy(self):
        """Return a new EpisodeData with lists stacked into tensors."""
        exported = EpisodeData()
        exported.seed = self.seed
        exported.env_id = self.env_id
        exported.success = self.success
        exported.data = _stack_recursive(self.data, in_place=False)
        return exported
