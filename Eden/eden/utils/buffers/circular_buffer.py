# Adapted from https://github.com/mujocolab/mjlab/blob/main/src/mjlab/utils/buffers/circular_buffer.py

"""Fixed-length circular buffer for batched tensor history.

Stores ``max_len`` frames per environment with shape
``(max_len, batch_size, ...)`` internally (time-first for pointer arithmetic)
and exposes a ``(batch_size, max_len, ...)`` view via the :attr:`buffer`
property (batch-first, chronological oldest-to-newest).

When backfill is enabled (default), the first append after construction or
reset copies the new value into every history slot so that downstream
consumers always see valid data instead of zeros.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class CircularBuffer:
    """Fixed-length circular buffer for batched tensor history.

    Stores history with shape ``(max_len, batch_size, ...)`` internally.
    The :attr:`buffer` property returns chronologically ordered data
    (oldest to newest) with shape ``(batch_size, max_len, ...)``.

    LIFO retrieval via ``__getitem__``::

        buffer[0]  # most recent frame
        buffer[1]  # one step back
        buffer[2]  # two steps back (oldest if max_len=3)

    Per-batch :meth:`reset` zeroes the specified rows and marks them
    for backfill on the next :meth:`append`.

    Use :meth:`peek_buffer` to obtain the chronological view that *would*
    result from appending a frame, without mutating internal state — useful
    when a caller needs a hypothetical post-append snapshot but the
    canonical advance is owned by another code path (e.g. the observation
    manager's ``update_history=False`` snapshot pass during
    ``record_final_observations``).

    Parameters
    ----------
    max_len : int
        Maximum number of historical frames to retain.
    batch_size : int
        Size of the batch dimension (number of environments).
    device : str
        Torch device for storage.
    backfill : bool, optional
        If True (default), the first append after construction or reset
        copies the new value into every history slot.
    """

    def __init__(self, max_len: int, batch_size: int, device: str, backfill: bool = True) -> None:
        if max_len < 1:
            raise ValueError(f"Buffer size must be >= 1, got {max_len}")

        self._max_len = max_len
        self._batch_size = batch_size
        self._device = device
        self._pointer: int = -1
        self._buffer: torch.Tensor | None = None
        # Reused scratch for the chronological reorder in ``buffer``/``peek_buffer`` so those don't allocate
        # a fresh tensor every call. Returned views alias this storage and are valid until the next call.
        self._chrono_buf: torch.Tensor | None = None
        self._peek_buf: torch.Tensor | None = None
        self._all_indices = torch.arange(batch_size, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._max_len_tensor = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
        self._backfill = backfill
        self._has_unfilled_rows = True  # CPU-side flag to avoid GPU sync in append()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def current_length(self) -> torch.Tensor:
        """Per-batch count of valid frames. Shape: (batch_size,)."""
        return torch.minimum(self._num_pushes, self._max_len_tensor)

    @property
    def is_initialized(self) -> bool:
        """Check if the buffer has been initialized with at least one append."""
        return self._buffer is not None

    @property
    def buffer(self) -> torch.Tensor:
        """History in chronological order (oldest to newest).

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, max_len, ...)``, index 0 is oldest.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        start = (self._pointer + 1) % self._max_len
        idx = (torch.arange(self._max_len, device=self._device) + start) % self._max_len
        if self._chrono_buf is None or self._chrono_buf.shape != self._buffer.shape:
            self._chrono_buf = torch.empty_like(self._buffer)
        torch.index_select(self._buffer, 0, idx, out=self._chrono_buf)  # (max_len, batch, ...)
        return self._chrono_buf.transpose(0, 1)  # (batch, max_len, ...)

    def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Zero out values and counters for specified batch rows.

        Parameters
        ----------
        batch_ids : Sequence[int] | torch.Tensor | None, optional
            Batch indices to reset, or ``None`` to reset all.
        """
        ids: Sequence[int] | torch.Tensor | slice = slice(None) if batch_ids is None else batch_ids
        self._num_pushes[ids] = 0
        if self._buffer is not None:
            self._buffer[:, ids] = 0.0
        if self._backfill:
            self._has_unfilled_rows = True

    def append(self, data: torch.Tensor) -> None:
        """Append a new frame for all batch elements.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape ``(batch_size, ...)``.
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

        data = data.to(self._device)

        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.zeros((self._max_len, *data.shape), dtype=data.dtype, device=self._device)

        self._pointer = (self._pointer + 1) % self._max_len
        self._buffer[self._pointer] = data

        # Backfill entire history with first frame for newly initialized batches.
        # Use CPU-side flag to avoid torch.any() GPU→CPU sync every step.
        if self._backfill and self._has_unfilled_rows:
            is_first_push = self._num_pushes == 0
            self._buffer[:, is_first_push] = data[is_first_push]
            self._has_unfilled_rows = False

        self._num_pushes += 1

    def peek_buffer(self, data: torch.Tensor) -> torch.Tensor:
        """Return the buffer that *would* result from ``append(data)``, without mutating state.

        Used by the observation manager when snapshotting a history-bearing
        group with ``update_history=False``: the freshly-computed frame must
        appear in the most-recent slot of the returned view, but the underlying
        buffer state is owned by the once-per-step canonical advance elsewhere.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape ``(batch_size, ...)``, same shape as ``append``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, max_len, ...)``, oldest to newest, with
            ``data`` occupying the newest slot. Does not mutate the ring; backed
            by reused scratch, so the view is valid only until the next
            ``buffer()``/``peek_buffer()`` call (copy to retain).
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

        data = data.to(self._device)

        if self._buffer is None:
            # Hypothetical first append on an uninitialized buffer. A reorder of zero storage is just zeros,
            # so allocate a fresh zero buffer (rare path) and let the per-row backfill below reproduce
            # append()'s ``backfill`` semantics.
            buf = torch.zeros((self._max_len, *data.shape), dtype=data.dtype, device=self._device)
        else:
            next_pointer = (self._pointer + 1) % self._max_len
            start = (next_pointer + 1) % self._max_len
            idx = (torch.arange(self._max_len, device=self._device) + start) % self._max_len
            if self._peek_buf is None or self._peek_buf.shape != self._buffer.shape:
                self._peek_buf = torch.empty_like(self._buffer)
            torch.index_select(self._buffer, 0, idx, out=self._peek_buf)  # reorder into reused scratch
            buf = self._peek_buf
        buf[-1] = data  # most-recent slot

        # Mirror append()'s per-row backfill for batches still on their first push.
        if self._backfill and self._has_unfilled_rows:
            is_first_push = self._num_pushes == 0
            buf[:, is_first_push] = data[is_first_push].unsqueeze(0)

        return buf.transpose(0, 1)  # (batch, max_len, ...)

    def __getitem__(self, key: torch.Tensor | int) -> torch.Tensor:
        """Retrieve lagged frames per batch (LIFO).

        Parameters
        ----------
        key : torch.Tensor | int
            Per-batch lags of shape ``(batch_size,)`` or a shared scalar lag.

        Returns
        -------
        torch.Tensor
            Lagged frames with shape ``(batch_size, ...)``.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        if isinstance(key, int):
            key = torch.full((self._batch_size,), key, dtype=torch.long, device=self._device)
        else:
            if key.ndim == 0:
                key = key.expand(self._batch_size)
            key = key.to(device=self._device, dtype=torch.long)

        if key.numel() != self._batch_size:
            raise ValueError(f"Expected {self._batch_size} lags, got {key.numel()}")

        pushes = self._num_pushes.clamp_min(1)
        valid = torch.minimum(key, pushes - 1).clamp_min(0)

        if torch.all(valid == 0):
            return self._buffer[self._pointer]

        idx = torch.remainder(self._pointer - valid, self._max_len)
        return self._buffer[idx, self._all_indices]
