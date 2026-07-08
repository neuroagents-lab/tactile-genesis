"""Sequential metric term that succeeds when sub-metrics hold in order."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import torch

import eden as en
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.options.managers.metrics import MetricTermOptions, PhaseOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class _HoldTuple(NamedTuple):
    term: MetricTerm
    threshold: float
    until_idx: int
    guard_idx: int | None  # index of another hold that must also fail


@METRIC_TERM_REGISTRY.register()
class SequentialMetricTerm(MetricTerm):
    """Metric that tracks ordered sub-goals (phases).

    Each phase is a predicate that must be satisfied before the next phase is
    evaluated.  The metric value is the fraction of phases completed
    (``completed / total``).  With the default ``success_threshold=1.0`` and
    ``direction="hib"``, the episode is considered successful once **all**
    phases are complete.

    Phase advancement is **per-environment** -- different parallel
    environments can be in different phases.

    Parameters
    ----------
    phases : list[PhaseOptions]
        Ordered list of phase definitions.  Each ``PhaseOptions`` must contain
        **one of**:

        * ``term`` -- a ``MetricTermOptions`` from ``.configure()``.
        * ``ref`` -- a string referencing a sibling term in ``MetricManagerOptions``.

        And optionally:

        * ``threshold`` -- (default ``1.0``) minimum value returned by the
          phase predicate that counts as "passed".
        * ``hold`` -- if ``True``, reuse the phase's own term as hold predicate.
        * ``hold_term`` -- a separate ``MetricTermOptions`` for the hold predicate
          (overrides ``hold``).
        * ``hold_until`` -- controls when the hold expires (phase name or index).

    allow_skip : bool
        If ``True``, an env may advance through multiple phases in a single
        ``compute()`` call.  If ``False`` (default), at most one phase
        transition happens per call.

    Example
    -------
    ::

        SequentialMetricTerm.configure(
            phases=[
                PhaseOptions(
                    name="reach",
                    term=EeNearEntity.configure(
                        robot_name="robot",
                        ee_link_names=["finger_l", "finger_r"],
                        entity_name="apple",
                        threshold=0.15,
                    ),
                    threshold=0.5,
                ),
                PhaseOptions(
                    name="grasp",
                    term=IsGrasping.configure(
                        robot_name="robot",
                        left_gripper_link_name="left_gripper_pad",
                        right_gripper_link_name="right_gripper_pad",
                        force_threshold=0.5,
                    ),
                    threshold=1.0,
                    hold=True,
                    hold_until="place",
                ),
            ],
            allow_skip=False,
        )
    """

    phases: list[PhaseOptions] = []
    allow_skip: bool = False

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env=env, options=options)
        self._phase_buf = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._resolved_phases: list[tuple[MetricTerm, float]] = []
        self._phase_names: list[str] = []
        self._phase_name_to_idx: dict[str, int] = {}
        self._child_terms: list[MetricTerm] = []
        self._hold_conditions: list[_HoldTuple | None] = []
        self._has_any_holds: bool = False
        self._phase_tensors: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _instantiate_term(self, term_options: MetricTermOptions) -> MetricTerm:
        """Instantiate a MetricTerm (class-based or function-based) from options."""
        term = METRIC_TERM_REGISTRY.get(term_options.name)(env=self._env, options=term_options)
        term.build()
        self._child_terms.append(term)
        return term

    def _resolve_hold_until(self, phase: PhaseOptions, phase_idx: int, num_phases: int) -> int:
        """Resolve ``hold_until`` to an exclusive upper bound on the hold window.

        Returns an index *U* such that the hold is active while
        ``phase_idx < current_phase < U``.
        """
        hold_until = phase.hold_until
        if hold_until is None:
            return num_phases  # active until task completion

        if isinstance(hold_until, str):
            if hold_until not in self._phase_name_to_idx:
                raise ValueError(f"hold_until='{hold_until}' not found in phase names: {self._phase_names}")
            hold_until = self._phase_name_to_idx[hold_until]

        if not isinstance(hold_until, int):
            raise TypeError(f"Invalid hold_until for phase {phase_idx}: {hold_until!r}")

        if hold_until <= phase_idx or hold_until >= num_phases:
            raise ValueError(f"hold_until={hold_until} for phase {phase_idx} must be in ({phase_idx}, {num_phases})")
        # Inclusive: hold is active *through* the given phase index
        return hold_until + 1

    def _resolve_hold(self, phase: PhaseOptions, phase_idx: int) -> tuple[MetricTerm, float] | None:
        """Resolve the hold condition of a phase.

        Returns ``None`` when no hold is requested, or a ``(term, threshold)``
        tuple otherwise.
        """
        if phase.hold_term is not None:
            hold_term_instance = self._instantiate_term(phase.hold_term)
            return (hold_term_instance, 1.0)
        if not phase.hold:
            return None
        # hold=True: reuse the phase's own term
        term_instance = self._resolved_phases[phase_idx][0]
        threshold = self._resolved_phases[phase_idx][1]
        return (term_instance, threshold)

    # ------------------------------------------------------------------
    # Ref resolution (called by MetricManager after all terms are built)
    # ------------------------------------------------------------------

    def resolve_refs(self, manager_options) -> None:
        """Resolve ``ref`` phases by looking up sibling MetricTermOptions.

        Called by ``MetricManager._prepare_terms()`` in a second pass after
        all terms have been built.
        """
        for i, phase in enumerate(self.phases):
            if phase.ref is not None and self._resolved_phases[i][0] is None:
                sibling_options = getattr(manager_options, phase.ref, None)
                if sibling_options is None:
                    raise ValueError(f"Phase {i} ref='{phase.ref}' not found in MetricManagerOptions")
                term_instance = self._instantiate_term(sibling_options)
                self._resolved_phases[i] = (term_instance, phase.threshold)

        # Now resolve hold conditions (requires all phases to be resolved)
        self._resolve_all_holds()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build(self):
        for i, phase in enumerate(self.phases):
            if not isinstance(phase, PhaseOptions):
                raise TypeError(
                    f"Phase {i} must be a PhaseOptions instance, got {type(phase).__name__}. "
                    f"Use PhaseOptions(term=MyTerm.configure(...), ...) instead of dict."
                )

            if phase.term is not None and phase.ref is not None:
                raise ValueError(f"Phase {i}: 'term' and 'ref' are mutually exclusive")
            if phase.term is None and phase.ref is None:
                raise ValueError(f"Phase {i}: must specify either 'term' or 'ref'")

            # Derive phase name
            name = phase.name
            if not name:
                if phase.term is not None:
                    name = phase.term.name
                elif phase.ref is not None:
                    name = phase.ref
                else:
                    name = f"phase_{i}"
            self._phase_names.append(name)
            self._phase_name_to_idx[name] = i

            # Instantiate term (or defer if ref)
            if phase.term is not None:
                term_instance = self._instantiate_term(phase.term)
                self._resolved_phases.append((term_instance, phase.threshold))
            else:
                # ref — will be resolved in resolve_refs()
                self._resolved_phases.append((None, phase.threshold))

        # If no refs, resolve holds now; otherwise MetricManager will call resolve_refs()
        has_refs = any(p.ref is not None for p in self.phases)
        if not has_refs:
            self._resolve_all_holds()

    def _resolve_hold_guard(self, phase: PhaseOptions, phase_idx: int) -> int | None:
        """Resolve ``hold_guard`` to a phase index.

        Returns ``None`` when no guard is set, or the index of the guard phase.
        The guard phase must have a hold condition itself.
        """
        guard = phase.hold_guard
        if guard is None:
            return None
        if guard not in self._phase_name_to_idx:
            raise ValueError(
                f"hold_guard='{guard}' for phase {phase_idx} ('{self._phase_names[phase_idx]}') "
                f"not found in phase names: {self._phase_names}"
            )
        return self._phase_name_to_idx[guard]

    def _resolve_all_holds(self):
        """Resolve hold conditions for all phases. Must be called after all terms are instantiated."""
        num_phases = len(self._resolved_phases)
        self._hold_conditions = []
        self._has_any_holds = False

        for i, phase in enumerate(self.phases):
            hold_base = self._resolve_hold(phase, i)
            if hold_base is not None:
                h_term, h_threshold = hold_base
                until = self._resolve_hold_until(phase, i, num_phases)
                guard_idx = self._resolve_hold_guard(phase, i)
                self._hold_conditions.append(_HoldTuple(h_term, h_threshold, until, guard_idx))
                self._has_any_holds = True
            else:
                self._hold_conditions.append(None)

        # Validate hold_guard references point to phases that actually have holds
        for i, hold in enumerate(self._hold_conditions):
            if hold is not None and hold.guard_idx is not None:
                guard_hold = self._hold_conditions[hold.guard_idx]
                if guard_hold is None:
                    raise ValueError(
                        f"hold_guard='{self.phases[i].hold_guard}' for phase {i} "
                        f"('{self._phase_names[i]}') references phase "
                        f"'{self._phase_names[hold.guard_idx]}' which has no hold condition"
                    )

        self._phase_tensors = [torch.tensor(i, dtype=torch.long, device=self._env.device) for i in range(num_phases)]

        names = ", ".join(f"{i}:{n}" for i, n in enumerate(self._phase_names))
        holds_count = sum(1 for h in self._hold_conditions if h is not None)
        guarded = sum(1 for h in self._hold_conditions if h is not None and h.guard_idx is not None)
        en.logger.info(
            f"SequentialMetricTerm built with {num_phases} phases: [{names}]"
            + (
                f" ({holds_count} hold conditions" + (f", {guarded} guarded" if guarded else "") + ")"
                if holds_count
                else ""
            )
        )

    @property
    def phase_names(self) -> list[str]:
        """Names of all phases (read-only)."""
        return list(self._phase_names)

    def get_phase_params(self, phase: int | str) -> dict:
        """Return the mutable params dict for a phase (by index or name).

        Only meaningful for function-based terms that have a ``params`` attribute.
        Modifying the returned dict changes future ``compute()`` calls.
        """
        idx = self._resolve_phase_ref(phase)
        term_instance = self._resolved_phases[idx][0]
        return term_instance.params

    def get_phase_term(self, phase: int | str) -> MetricTerm:
        """Return the underlying term instance for a phase (by index or name)."""
        idx = self._resolve_phase_ref(phase)
        return self._resolved_phases[idx][0]

    def _resolve_phase_ref(self, phase: int | str) -> int:
        if isinstance(phase, int):
            if phase < 0 or phase >= len(self._resolved_phases):
                raise IndexError(f"Phase index {phase} out of range [0, {len(self._resolved_phases)})")
            return phase
        if phase not in self._phase_name_to_idx:
            raise KeyError(f"Phase '{phase}' not found in {self._phase_names}")
        return self._phase_name_to_idx[phase]

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._phase_buf.zero_()
        else:
            self._phase_buf[envs_idx] = 0
        # Reset child terms as well
        for child in self._child_terms:
            child.reset(envs_idx=envs_idx)

    def compute(self) -> torch.Tensor:
        num_phases = len(self._resolved_phases)
        if num_phases == 0:
            return torch.ones(self._env.num_envs, device=self.device)

        # ------------------------------------------------------------------
        # Hold-condition check: revert to earliest failing hold
        # ------------------------------------------------------------------
        if self._has_any_holds:
            # Track the earliest failing phase per env (num_phases = "no failure")
            revert_target = torch.full((self._env.num_envs,), num_phases, dtype=torch.long, device=self.device)

            # Pre-compute hold values so guards can reference them.
            # hold_failed[p] is a per-env bool tensor (True = hold failing).
            hold_failed: list[torch.Tensor | None] = [None] * num_phases
            for p, hold in enumerate(self._hold_conditions):
                if hold is None:
                    continue
                candidates = (self._phase_buf > p) & (self._phase_buf < hold.until_idx)
                if not candidates.any():
                    continue
                h_value = hold.term.compute()
                h_failed = h_value < hold.threshold
                hold_failed[p] = candidates & h_failed

                if en.logger.level <= logging.DEBUG:
                    for e in range(self._env.num_envs):
                        if candidates[e]:
                            en.logger.debug(
                                f"  [env {e}] hold check phase {p} ({self._phase_names[p]}): "
                                f"value={h_value[e].item():.4f}  threshold={hold.threshold}  "
                                f"failed={bool(h_failed[e])}"
                            )

            # Apply hold failures, respecting guards.
            for p, hold in enumerate(self._hold_conditions):
                if hold is None or hold_failed[p] is None:
                    continue
                failing = hold_failed[p]
                if not failing.any():
                    continue

                # If this hold has a guard, suppress it for envs where the
                # guard's hold is still passing (i.e. guard not failing).
                if hold.guard_idx is not None:
                    guard_failing = hold_failed[hold.guard_idx]
                    if guard_failing is None:
                        # Guard hold wasn't even evaluated (no envs in window) → guard is passing → suppress all
                        if en.logger.level <= logging.DEBUG:
                            en.logger.debug(
                                f"  hold on phase {p} ({self._phase_names[p]}) fully suppressed "
                                f"by guard '{self._phase_names[hold.guard_idx]}' (guard passing)"
                            )
                        continue
                    # Only enforce this hold where the guard is also failing
                    failing = failing & guard_failing
                    if not failing.any():
                        if en.logger.level <= logging.DEBUG:
                            en.logger.debug(
                                f"  hold on phase {p} ({self._phase_names[p]}) suppressed "
                                f"by guard '{self._phase_names[hold.guard_idx]}' (guard passing for all envs)"
                            )
                        continue

                revert_target = torch.where(failing & (p < revert_target), self._phase_tensors[p], revert_target)

            # Apply reverts
            needs_revert = revert_target < num_phases
            if needs_revert.any():
                if en.logger.level <= logging.INFO:
                    for e in range(self._env.num_envs):
                        if needs_revert[e]:
                            old_p = self._phase_buf[e].item()
                            new_p = revert_target[e].item()
                            en.logger.info(
                                f"[env {e}] HOLD FAILED: reverting from phase {old_p} ({self._phase_names[old_p]}) "
                                f"→ phase {new_p} ({self._phase_names[new_p]})"
                            )
                self._phase_buf = torch.where(needs_revert, revert_target, self._phase_buf)

        # ------------------------------------------------------------------
        # Phase advancement
        # ------------------------------------------------------------------
        prev_phases = self._phase_buf.clone()

        # When allow_skip is False, use a snapshot so each env advances at
        # most one phase per compute().  When True, read _phase_buf directly
        # so envs that just advanced are re-evaluated for the next phase.
        phase_view = self._phase_buf if self.allow_skip else prev_phases

        for p, (term_instance, threshold) in enumerate(self._resolved_phases):
            mask = phase_view == p
            if not mask.any():
                continue
            value = term_instance.compute()
            passed = value >= threshold
            advance = mask & passed
            self._phase_buf[advance] = p + 1

            if en.logger.level <= logging.DEBUG:
                for e in range(self._env.num_envs):
                    if mask[e]:
                        en.logger.debug(
                            f"  [env {e}] phase {p} ({self._phase_names[p]}): "
                            f"value={value[e].item():.4f}  threshold={threshold}  "
                            f"passed={bool(passed[e])}"
                        )

        # Clamp to [0, num_phases]
        self._phase_buf.clamp_(max=num_phases)

        if en.logger.level <= logging.INFO:
            for e in range(self._env.num_envs):
                if self._phase_buf[e] != prev_phases[e]:
                    old_p = prev_phases[e].item()
                    new_p = self._phase_buf[e].item()
                    old_name = self._phase_names[old_p]
                    if new_p >= num_phases:
                        en.logger.info(
                            f"[env {e}] Phase {old_p} ({old_name}) DONE → task COMPLETE ({new_p}/{num_phases})"
                        )
                    else:
                        en.logger.info(
                            f"[env {e}] Phase {old_p} ({old_name}) DONE → now at phase {new_p} ({self._phase_names[new_p]})  [{new_p}/{num_phases}]"
                        )

        return self._phase_buf.float() / num_phases
