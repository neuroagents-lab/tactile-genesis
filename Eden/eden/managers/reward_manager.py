"""Reward manager for computing reward signals."""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, TypeAlias

import torch

import eden as en
from eden.managers.base import ManagerBase, ManagerTermBase, ManagerTermFuncWrapperBase
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


class RewardTerm(ManagerTermBase, ConfigurableMixin[RewardTermOptions]):
    """Base class for reward terms."""

    range_s: list[tuple[float, float]] | None = None
    weight: float = 1.0
    tags: list[str] = []

    def __init__(self, env: RLEnvBase, options: RewardTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)
        self.is_temporal_dependent = options.range_s is not None

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        # NOTE: resets the term state if needed.
        pass


class RewardTermFuncWrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[RewardTermOptions]):
    """Base class for reward terms defined as a function."""

    range_s: list[tuple[float, float]] | None = None
    weight: float = 1.0
    tags: list[str] = []

    def __init__(self, func: FunctionType, env: RLEnvBase, options: RewardTermOptions):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)
        self.is_temporal_dependent = options.range_s is not None


RewardTermLike: TypeAlias = RewardTerm | RewardTermFuncWrapper
"""Union of a class-based reward term and its function-wrapper form."""


REWARD_TERM_REGISTRY = TermRegistry("REWARD_TERM", RewardTerm, RewardTermFuncWrapper)


class RewardManager(ManagerBase[RewardManagerOptions]):
    def __init__(self, env: RLEnvBase, options: RewardManagerOptions):
        super().__init__(env=env, options=options)
        num_terms = len(self._term_names)

        # 2D episode sums for vectorized accumulation (replaces per-term dict).
        # ``_episode_sums`` is the weighted+masked accumulation; ``_episode_raw_sums``
        # mirrors the raw (unweighted) term outputs for diagnostic logging.
        self._episode_sums = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self._episode_raw_sums = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._step_reward = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)

        # Pre-compute weights vector for vectorized multiply
        self._weights = torch.tensor(
            [term.weight for term in self._terms.values()],
            dtype=torch.float,
            device=self.device,
        )

        # Pre-allocate temporal mask (1.0 for non-temporal terms, updated per-step for temporal)
        self._temporal_mask = torch.ones((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self._temporal_term_info: list[tuple[int, list]] = []
        for idx, term in enumerate(self._terms.values()):
            # Hand each term its column of the step-reward buffer as its output cache. Cache-aware terms
            # write straight into it (no per-term alloc + copy); others return a tensor we copy in. See
            # ManagerTermBase for the cache protocol.
            term._cache = self._step_reward[:, idx]
            if term.is_temporal_dependent:
                self._temporal_term_info.append((idx, term.range_s))

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Reward Terms",
            field_names=["Index", "Name", "Weight"],
            rows=([index, name, term.weight] for index, (name, term) in enumerate(self._terms.items())),
            align={"Name": "l", "Weight": "r"},
        )

    def iter_terms(self):
        """Yield ``(name, term)`` pairs for every active reward term.

        Public accessor for sibling managers (e.g. curricula) that need to scan terms
        without reaching into the private ``_terms`` dict.
        """
        return self._terms.items()

    def get_episode_sum(self, name: str, raw: bool = True) -> torch.Tensor:
        """Return the per-env running reward sum for term ``name`` as a ``(num_envs,)`` view.

        ``raw=True`` (default) returns the unweighted, mask-applied accumulation;
        ``raw=False`` returns the weighted post-curriculum sum. Both buffers are
        reset to zero per env in :meth:`reset`. Raises ``KeyError`` (with the
        active term list) if ``name`` is not registered. See :meth:`iter_terms`
        for full-buffer scanning.
        """
        idx = self._term_name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"reward term '{name}' not active. Active terms: {self._term_names}")
        buf = self._episode_raw_sums if raw else self._episode_sums
        return buf[:, idx]

    @staticmethod
    def is_timestep_in_range(timestep: torch.Tensor, range_s: list[tuple[float, float]]) -> torch.Tensor:
        """Check whether a batch of timesteps falls within any of the given intervals.

        Parameters
        ----------
        timestep : torch.Tensor
            ``(batch_size,)`` tensor of timesteps.
        range_s : list of tuple of float
            List of ``(lower, upper)`` tuples defining valid intervals.

        Returns
        -------
        torch.Tensor
            ``(batch_size,)`` boolean tensor indicating membership.
        """
        is_valid = torch.zeros_like(timestep, dtype=torch.bool, device=timestep.device)
        for lower, upper in range_s:
            is_valid |= (timestep >= lower) & (timestep <= upper)
        return is_valid

    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if envs_idx is None:
            envs_idx = slice(None)
        extras = {}
        for term_idx, key in enumerate(self._term_names):
            selected = self._episode_sums[envs_idx, term_idx]
            denom = max(selected.numel(), 1)
            episodic_sum_avg = selected.sum() / denom
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            self._episode_sums[envs_idx, term_idx] = 0.0

            raw_selected = self._episode_raw_sums[envs_idx, term_idx]
            raw_avg = raw_selected.sum() / denom
            extras["Episode_Reward_Raw/" + key] = raw_avg / self._env.max_episode_length_s
            self._episode_raw_sums[envs_idx, term_idx] = 0.0
        for terms in self._reset_terms:
            terms.reset(envs_idx=envs_idx)
        return extras

    @torch.inference_mode()
    def compute(self, dt: float) -> torch.Tensor:
        # _step_reward is mutated in-place through the phases below.
        # After compute() returns it holds weighted, masked values
        # (i.e. compute() * mask * weight), NOT raw term outputs.

        # Phase 1: Fill raw term outputs and sync weights (curriculum may modify them).
        # Each term's ``_cache`` is its column of _step_reward (set in __init__); compute_cached() writes
        # the result into that column (zero-copy for cache-aware terms, one copy otherwise). See
        # ManagerTermBase.compute_cached for the protocol.
        for term_idx, term in enumerate(self._terms.values()):
            term.compute_cached()
            self._weights[term_idx] = term.weight

        # Phase 2a: Apply temporal mask in-place. Raw episode sums respect this
        # mask too — accumulating before masking would log term outputs from
        # timesteps where range_s makes the term inactive.
        if self._temporal_term_info:
            for idx, range_s in self._temporal_term_info:
                self._temporal_mask[:, idx] = self.is_timestep_in_range(self._env.episode_length_buf, range_s).float()
            # _step_reward *= mask  →  now holds compute() * mask
            self._step_reward.mul_(self._temporal_mask)

        # Phase 2b: Accumulate raw episode sums (compute() * mask, no weights).
        self._episode_raw_sums.add_(self._step_reward, alpha=dt)

        # Phase 2c: Apply per-term weights in-place.
        # _step_reward *= weights  →  now holds compute() * mask * weight
        self._step_reward.mul_(self._weights)

        # Phase 3: Reduce to scalar reward and accumulate episode sums.
        torch.sum(self._step_reward, dim=1, out=self._reward_buf)
        self._reward_buf.mul_(dt)
        self._episode_sums.add_(self._step_reward, alpha=dt)
        return self._reward_buf

    def _prepare_terms(self):
        self._terms: dict[str, RewardTermLike] = dict()
        self._reset_terms: list[RewardTerm] = []

        for term_name in self._options.term_keys():
            term_option: RewardTermOptions = getattr(self._options, term_name)
            if term_option.weight == 0.0:
                en.logger.warning(f"Reward term {term_name} has weight 0.0, it will be ignored.")
                continue
            term: RewardTermLike = self._build_term(term_name, REWARD_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
            if isinstance(term, RewardTerm):
                self._reset_terms.append(term)

        # name → column-index map; used by get_episode_sum() so per-step termination
        # terms don't pay an O(num_terms) list scan on every call.
        self._term_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self._term_names)}
