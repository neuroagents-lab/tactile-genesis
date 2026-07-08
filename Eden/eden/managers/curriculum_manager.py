"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING
from types import FunctionType

import torch

from eden.managers.base import ManagerBase, ManagerTermBase, ManagerTermFuncWrapperBase
from eden.options.managers.curricula import (
    CurriculumManagerOptions,
    CurriculumTermOptions,
)
from eden.utils.common import ConfigurableMixin, ConfigurableFuncWrapperMixin
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class CurriculumTerm(ManagerTermBase, ConfigurableMixin[CurriculumTermOptions]):
    """Base class for curriculum terms."""

    def __init__(self, env: EnvBase, options: CurriculumTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)


class CurriculumTermFuncWrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[CurriculumTermOptions]):
    def __init__(self, func: FunctionType, env: EnvBase, options: CurriculumTermOptions):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)


CURRICULUM_TERM_REGISTRY = TermRegistry("CURRICULUM_TERM", CurriculumTerm, CurriculumTermFuncWrapper)


class CurriculumManager(ManagerBase[CurriculumManagerOptions]):
    """Curriculum manager for managing the curriculum of the environment."""

    def __init__(self, env: EnvBase, options: CurriculumManagerOptions):
        super().__init__(env=env, options=options)

        self._curriculum_state = dict()
        for term_name in self._term_names:
            self._curriculum_state[term_name] = None

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Curriculum Terms",
            field_names=["Index", "Name"],
            rows=([index, name] for index, name in enumerate(self._term_names)),
            align={"Name": "l"},
        )

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> dict[str, float]:
        extras = {}
        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                if isinstance(term_state, dict):
                    for key, value in term_state.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        extras[f"Curriculum/{term_name}/{key}"] = value
                else:
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.item()
                    extras[f"Curriculum/{term_name}"] = term_state
        for term in self._reset_terms:
            term.reset(envs_idx=envs_idx)
        return extras

    def compute(self) -> None:
        for name, term in self._terms.items():
            state = term.compute()
            self._curriculum_state[name] = state

    def get_state(self, term_name: str) -> dict | None:
        """Return the most recent ``compute()`` output cached for the named term.

        Returns ``None`` when the term has never computed yet (e.g. the named
        term is declared *after* the caller in ``curriculum_options``, so its
        first ``compute()`` happens later in this same step). Callers that
        depend on a fresh value should declare the producer term before
        themselves.
        """
        return self._curriculum_state.get(term_name)

    def _prepare_terms(self):
        self._terms: dict[str, CurriculumTerm | CurriculumTermFuncWrapper] = dict()
        self._reset_terms: list[CurriculumTerm] = []

        for term_name in self._options.term_keys():
            term: CurriculumTerm | CurriculumTermFuncWrapper = self._build_term(term_name, CURRICULUM_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
            if isinstance(term, CurriculumTerm):
                self._reset_terms.append(term)
