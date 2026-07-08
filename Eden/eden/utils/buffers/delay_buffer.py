# Adapted from https://github.com/mujocolab/mjlab/blob/main/src/mjlab/utils/buffers/delay_buffer.py

"""Delay buffer for stochastically delayed observations.

Wraps a :class:`~eden.utils.buffers.circular_buffer.CircularBuffer` to
simulate observation (or action) delays by returning the frame from
``T - lag`` timesteps ago, where lag is sampled from ``[min_lag, max_lag]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from eden.utils.buffers.circular_buffer import CircularBuffer


class DelayBuffer:
    """Serve stochastically delayed observations from a rolling history.

    At each timestep the caller appends a new frame via :meth:`append`,
    then calls :meth:`compute` to retrieve the delayed frame.  The lag
    per environment is sampled from ``[min_lag, max_lag]`` and can be
    refreshed every step or on a configurable period.

    Parameters
    ----------
    min_lag : int
        Minimum lag in steps (inclusive, >= 0).
    max_lag : int
        Maximum lag in steps (inclusive, >= ``min_lag``).
    batch_size : int
        Number of parallel environments.
    device : str
        Torch device for storage and RNG.
    per_env : bool, optional
        If True (default), sample independent lags per environment;
        otherwise share one lag across all environments.
    hold_prob : float, optional
        Probability in ``[0, 1]`` to keep the previous lag when an
        update would occur.  Default is ``0.0``.
    update_period : int, optional
        If > 0, refresh lags every *N* steps per environment; if 0
        (default), consider updating every step.
    per_env_phase : bool, optional
        If True (default) and ``update_period > 0``, each environment
        uses a random phase offset in ``[0, update_period)`` so lag
        refreshes are staggered across the batch.
    generator : torch.Generator | None, optional
        Optional RNG for reproducible lag sampling.
    """

    def __init__(
        self,
        min_lag: int = 0,
        max_lag: int = 3,
        batch_size: int = 1,
        device: str = "cpu",
        per_env: bool = True,
        hold_prob: float = 0.0,
        update_period: int = 0,
        per_env_phase: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if min_lag < 0:
            raise ValueError(f"min_lag must be >= 0, got {min_lag}")
        if max_lag < min_lag:
            raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
        if not 0.0 <= hold_prob <= 1.0:
            raise ValueError(f"hold_prob must be in [0, 1], got {hold_prob}")
        if update_period < 0:
            raise ValueError(f"update_period must be >= 0, got {update_period}")

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.batch_size = batch_size
        self.device = device
        self.per_env = per_env
        self.hold_prob = hold_prob
        self.update_period = update_period
        self.per_env_phase = per_env_phase
        self.generator = generator

        buffer_size = max_lag + 1 if max_lag > 0 else 1
        self._buffer = CircularBuffer(max_len=buffer_size, batch_size=batch_size, device=device)
        self._current_lags = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._step_count = torch.zeros(batch_size, dtype=torch.long, device=device)

        if update_period > 0 and per_env_phase:
            self._phase_offsets = torch.randint(
                0,
                update_period,
                (batch_size,),
                dtype=torch.long,
                device=device,
                generator=generator,
            )
        else:
            self._phase_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

    @property
    def is_initialized(self) -> bool:
        """Check if buffer has been initialized with at least one append."""
        return self._buffer.is_initialized

    @property
    def current_lags(self) -> torch.Tensor:
        """Current lag per environment. Shape: (batch_size,)."""
        return self._current_lags

    def set_lags(
        self,
        lags: int | torch.Tensor,
        batch_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set lag values for specified environments.

        Parameters
        ----------
        lags : int | torch.Tensor
            Lag values of shape ``(num_batch_ids,)`` or scalar.
        batch_ids : Sequence[int] | torch.Tensor | slice | None, optional
            Batch indices to set, or ``None`` to set all.
        """
        if isinstance(lags, int):
            lags = torch.tensor(lags, dtype=torch.long, device=self.device)
        idx = slice(None) if batch_ids is None else batch_ids
        self._current_lags[idx] = lags.clamp(self.min_lag, self.max_lag)

    def reset(self, batch_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Reset specified environments to initial state.

        Parameters
        ----------
        batch_ids : Sequence[int] | torch.Tensor | slice | None, optional
            Batch indices to reset, or ``None`` to reset all.
        """
        if isinstance(batch_ids, slice):
            indices = range(*batch_ids.indices(self.batch_size))
            batch_ids = list(indices)

        self._buffer.reset(batch_ids=batch_ids)
        idx = slice(None) if batch_ids is None else batch_ids
        self._current_lags[idx] = 0
        self._step_count[idx] = 0
        if self.update_period > 0 and self.per_env_phase:
            new_phases = torch.randint(
                0,
                self.update_period,
                (self.batch_size,),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
            self._phase_offsets[idx] = new_phases[idx]

    def append(self, data: torch.Tensor) -> None:
        """Append a new frame to the buffer.

        Parameters
        ----------
        data : torch.Tensor
            Observation tensor of shape ``(batch_size, ...)``.
        """
        self._buffer.append(data)

    def compute(self) -> torch.Tensor:
        """Return the delayed frame for the current step.

        Returns
        -------
        torch.Tensor
            Delayed observation with shape ``(batch_size, ...)``.
        """
        if not self.is_initialized:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        self._update_lags()

        # Clamp lags to valid range [0, buffer_length - 1].
        # Buffer may not be full yet (e.g., only 2 frames but sampled lag=3).
        valid_lags = torch.minimum(self._current_lags, self._buffer.current_length - 1)
        valid_lags = valid_lags.clamp_min(0)

        return self._buffer[valid_lags]

    def _update_lags(self) -> None:
        """Update current lags according to configured policy."""
        if self.update_period > 0:
            phase_adjusted_count = (self._step_count + self._phase_offsets) % (self.update_period)
            should_update = phase_adjusted_count == 0
        else:
            should_update = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        new_lags = self._sample_lags(should_update)
        self._current_lags = torch.where(should_update, new_lags, self._current_lags)
        self._step_count += 1

    def _sample_lags(self, mask: torch.Tensor) -> torch.Tensor:
        """Sample new lags for specified environments.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape ``(batch_size,)`` indicating which
            environments to sample.

        Returns
        -------
        torch.Tensor
            New lags with shape ``(batch_size,)``.
        """
        if self.per_env:
            candidate_lags = torch.randint(
                self.min_lag,
                self.max_lag + 1,
                (self.batch_size,),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
        else:
            shared_lag = torch.randint(
                self.min_lag,
                self.max_lag + 1,
                (1,),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
            candidate_lags = shared_lag.expand(self.batch_size)

        if self.hold_prob > 0.0:
            should_sample = (
                torch.rand(
                    self.batch_size,
                    dtype=torch.float32,
                    device=self.device,
                    generator=self.generator,
                )
                >= self.hold_prob
            )
            update_mask = mask & should_sample
        else:
            update_mask = mask

        return torch.where(update_mask, candidate_lags, self._current_lags)
