"""Reward-weight curricula (staged ramp-up, penalty scheduling)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

import eden as en
from eden.managers.curriculum_manager import (
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
    CurriculumTermOptions,
)

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@CURRICULUM_TERM_REGISTRY.register()
class StageRewardWeightCurriculum(CurriculumTerm):
    """Curriculum that modifies a reward weight a given number of training steps.

    Parameters
    ----------
    reward_term_name: str
        The name of the reward term.
    weight_stages: list[tuple[int, float]]
        The list of stages with the training step and weight.
    """

    reward_term_name: str = ""
    weight_stages: list[tuple[int, float]] = []

    def __init__(self, env: RLEnvBase, options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        # make sure the stages are sorted in descending order
        self.weight_stages.sort(key=lambda x: x[0], reverse=True)
        self._reward_term = None

    def compute(self, *args, **kwargs):
        if self._reward_term is None:
            self._reward_term = self._env.reward_manager.get_term(self.reward_term_name)
        # NOTE: the stages are sorted in descending order, so we can break the loop
        for stage_step, stage_weight in self.weight_stages:
            if self._env.common_step_counter > stage_step:
                self._reward_term.weight = stage_weight
                break


@CURRICULUM_TERM_REGISTRY.register()
class PenaltyCurriculum(CurriculumTerm):
    """Curriculum that gradually scales reward weights for a set of reward terms.

    Monitors average episode length and adjusts a scale factor applied to all
    targeted reward terms. When episodes are short (agent struggling), the scale
    decreases to ease penalties. When episodes are long (agent succeeding), the
    scale increases to enforce stricter behaviour.

    The scale factor multiplies the *configured* weight of each term, so the
    effective weight is ``configured_weight * scale``.

    Targets can be specified by name OR by tag — exactly one entry path must be
    used per curriculum instance. Mixing the two would split intent across two
    fields and make refactors error-prone (e.g. renaming a term forgets the
    name-list, but the tag-list still matches). Prefer ``reward_term_tags`` when
    the same curriculum should apply to a group of penalties whose roster may
    grow (just tag new terms); use ``reward_term_names`` for explicit, audited
    targets.

    Parameters
    ----------
    reward_term_names: list[str]
        Explicit names of reward terms to scale. Mutually exclusive with
        ``reward_term_tags``.
    reward_term_tags: list[str]
        Tags to match on ``RewardTermOptions.tags``. A term is selected if any
        of its tags appears in this list. Mutually exclusive with
        ``reward_term_names``.
    initial_scale: float
        Starting scale factor. Default 0.1 (10% of configured weight).
    min_scale: float
        Minimum allowed scale. Default 0.0.
    max_scale: float
        Maximum allowed scale. Default 1.0.
    level_up_threshold: float
        Average episode length (in steps) above which the scale increases.
    level_down_threshold: float
        Average episode length (in steps) below which the scale decreases.
    degree: float
        Amount to adjust the scale per curriculum compute call.
    num_episode_average: int
        Number of most recent episode lengths to average over. Ignored when
        ``tracker_name`` is set (the named tracker owns the EMA).
    tracker_name: str | None
        Optional name of an ``AverageEpisodeLengthTracker`` curriculum term
        whose ``average_episode_length`` should drive this curriculum instead
        of the per-instance buffer. The tracker must be declared **before**
        this PenaltyCurriculum in ``curriculum_options`` so the tracker's
        compute() runs first within a step. When ``None`` (default), the
        legacy per-instance CPU buffer path is used and behavior is unchanged.
    """

    reward_term_names: list[str] = []
    reward_term_tags: list[str] = []
    initial_scale: float = 0.1
    min_scale: float = 0.0
    max_scale: float = 1.0
    level_up_threshold: float = 750.0
    level_down_threshold: float = 150.0
    degree: float = 0.00025
    num_episode_average: int = 1000
    tracker_name: str | None = None

    def __init__(self, env: RLEnvBase, options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        if not self.reward_term_names and not self.reward_term_tags:
            raise ValueError("PenaltyCurriculum requires either `reward_term_names` or `reward_term_tags` to be set.")
        if self.reward_term_names and self.reward_term_tags:
            raise ValueError(
                "PenaltyCurriculum: `reward_term_names` and `reward_term_tags` are mutually exclusive — "
                "set exactly one. Use names for explicit targets, tags for groups whose roster may grow."
            )
        self._scale = self.initial_scale
        self._targeted_terms: list = []
        self._base_weights: list[float] = []
        if self.tracker_name is None:
            # Rolling buffer for episode lengths. Kept on CPU on purpose: the running
            # mean is consumed by compute() every env step and used to branch on
            # ``level_up_threshold`` / ``level_down_threshold``. A device-side mean
            # would force a GPU→CPU sync per step, defeating async kernel launch.
            # Episode lengths are tiny ints; the CPU buffer is cheap.
            self._episode_lengths = torch.zeros(self.num_episode_average, device="cpu")
            self._ep_cursor = 0
            self._ep_count = 0
        self._resolved = False

    def _resolve_terms(self) -> None:
        """Lazily resolve reward terms (reward_manager may not exist at __init__ time)."""
        if self._resolved:
            return
        reward_manager = self._env.reward_manager
        if self.reward_term_names:
            # Explicit names: error if any is missing.
            self._targeted_terms = [reward_manager.get_term(name) for name in self.reward_term_names]
        else:
            # Tag match: scan all terms and keep those carrying any listed tag.
            tag_set = set(self.reward_term_tags)
            self._targeted_terms = [
                term
                for _, term in reward_manager.iter_terms()
                if tag_set.intersection(getattr(term, "tags", None) or [])
            ]
        self._base_weights = [term.weight for term in self._targeted_terms]
        # Empty resolution is allowed (e.g. tag-based curriculum on a task that doesn't
        # use that tag yet) but warn loudly so it isn't a silent no-op.
        if not self._targeted_terms:
            en.logger.warning(
                "PenaltyCurriculum resolved zero target reward terms "
                f"(reward_term_names={self.reward_term_names!r}, reward_term_tags={self.reward_term_tags!r}). "
                "The curriculum will be a no-op — check that names exist and that target terms carry the listed tags."
            )
        self._apply_scale()
        self._resolved = True

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self._resolve_terms()
        if self.tracker_name is not None:
            # The named tracker owns the EMA; nothing to update here.
            return
        if envs_idx is None:
            return
        # Pull episode lengths to CPU once per reset and update the running mean
        # incrementally (avoids per-step ``.mean().item()`` syncs in compute()).
        ep_lens = self._env.episode_length_buf[envs_idx].float().cpu()
        n = ep_lens.shape[0]
        if n == 0:
            return
        buf_size = self.num_episode_average
        indices = (torch.arange(n) + self._ep_cursor) % buf_size
        self._episode_lengths[indices] = ep_lens
        self._ep_cursor = (self._ep_cursor + n) % buf_size
        self._ep_count = min(self._ep_count + n, buf_size)
        # Refresh cached running mean as a Python float — no per-step sync below.
        self._avg_ep_len = float(self._episode_lengths[: self._ep_count].mean())

    def _read_tracker_state(self) -> tuple[float, bool] | None:
        """Return ``(average_episode_length, is_primed)`` from the named tracker.

        Returns ``None`` when the named term has not computed yet (caller treats
        this as a config-ordering error).
        """
        state = self._env.curriculum_manager.get_state(self.tracker_name)
        if state is None:
            return None
        if "average_episode_length" not in state:
            raise KeyError(
                f"PenaltyCurriculum(tracker_name={self.tracker_name!r}): the named curriculum term's "
                "compute() output does not contain 'average_episode_length'. Expected an "
                "AverageEpisodeLengthTracker (or compatible term) at this slot."
            )
        # ``is_primed`` is opt-in: legacy or third-party trackers without the flag
        # are assumed primed (preserves prior behavior on integrations that haven't
        # been updated to surface the flag).
        is_primed = bool(state.get("is_primed", True))
        return float(state["average_episode_length"]), is_primed

    def compute(self, *args, **kwargs) -> dict[str, float]:
        self._resolve_terms()

        if self.tracker_name is not None:
            tracker_state = self._read_tracker_state()
            if tracker_state is None:
                # Tracker is registered but hasn't computed yet — it must be declared
                # *before* this term in curriculum_options. Fail loud rather than
                # silently no-op for a step.
                raise RuntimeError(
                    f"PenaltyCurriculum(tracker_name={self.tracker_name!r}) saw no state from "
                    f"curriculum term {self.tracker_name!r}. Declare it **before** the "
                    "PenaltyCurriculum in `curriculum_options`; CurriculumManager evaluates terms "
                    "in declaration order."
                )
            avg_ep_len, is_primed = tracker_state
            # Mirror the legacy ``_ep_count > 0`` gate: only move the scale once
            # the tracker has actually consumed at least one reset. Otherwise the
            # initial-zero estimate would push ``_scale`` toward ``min_scale`` on
            # every pre-first-reset step.
            if is_primed:
                if avg_ep_len > self.level_up_threshold:
                    self._scale = min(self._scale + self.degree, self.max_scale)
                elif avg_ep_len < self.level_down_threshold:
                    self._scale = max(self._scale - self.degree, self.min_scale)
                self._apply_scale()
            return {"scale": self._scale, "avg_episode_length": avg_ep_len}

        avg_ep_len = getattr(self, "_avg_ep_len", 0.0)

        # Only update the scale once we have at least one recorded episode length.
        # Otherwise compute() runs every env.step() from start of training and the
        # default avg_ep_len=0.0 would push _scale toward min_scale before any
        # real episode data exists.
        if self._ep_count > 0:
            if avg_ep_len > self.level_up_threshold:
                self._scale = min(self._scale + self.degree, self.max_scale)
            elif avg_ep_len < self.level_down_threshold:
                self._scale = max(self._scale - self.degree, self.min_scale)
            self._apply_scale()
        return {"scale": self._scale, "avg_episode_length": avg_ep_len}

    def _apply_scale(self) -> None:
        for term, base_w in zip(self._targeted_terms, self._base_weights):
            term.weight = base_w * self._scale
