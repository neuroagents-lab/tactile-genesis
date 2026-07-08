"""Termination manager for computing done signals."""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, TypeAlias

import torch

from eden.managers.base import ManagerBase, ManagerTermBase, ManagerTermFuncWrapperBase
from eden.options.managers.terminations import (
    TerminationManagerOptions,
    TerminationTermOptions,
)
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class TerminationTerm(ManagerTermBase, ConfigurableMixin[TerminationTermOptions]):
    time_out: bool = False

    def __init__(self, env: EnvBase, options: TerminationTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)


class TerminationTermFuncWrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[TerminationTermOptions]):
    time_out: bool = False

    def __init__(self, func: FunctionType, env: EnvBase, options: TerminationTermOptions):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)


TerminationTermLike: TypeAlias = TerminationTerm | TerminationTermFuncWrapper
"""Union of a class-based termination term and its function-wrapper form."""


TERMINATION_TERM_REGISTRY = TermRegistry("TERMINATION_TERM", TerminationTerm, TerminationTermFuncWrapper)


class TerminationManager(ManagerBase[TerminationManagerOptions]):
    def __init__(self, env: EnvBase, options: TerminationManagerOptions):
        super().__init__(env=env, options=options)

        self._term_dones: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._term_dones[term_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._terminated_buf = torch.zeros_like(self._truncated_buf)

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Termination Terms",
            field_names=["Index", "Name", "Time Out"],
            rows=([index, name, term.time_out] for index, (name, term) in enumerate(self._terms.items())),
            align={"Name": "l"},
        )

    @property
    def dones(self) -> torch.Tensor:
        return self._truncated_buf | self._terminated_buf

    @property
    def timeouts(self) -> torch.Tensor:
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated_buf

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, torch.Tensor]:
        if envs_idx is None:
            envs_idx = slice(None)
        extras = {}
        for key in self._term_dones.keys():
            extras["Episode_Termination/" + key] = torch.count_nonzero(self._term_dones[key][envs_idx]).item()
        for term in self._reset_terms:
            term.reset(envs_idx=envs_idx)
        return extras

    def compute(self) -> torch.Tensor:
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        for name, term in self._terms.items():
            value = term.compute()
            if term.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            self._term_dones[name][:] = value
        # NOTE: reset corrupted state (call set_qpos to reset) based on #2239
        error_envs_mask = self._env.rigid_solver.get_error_envs_mask()
        self._terminated_buf |= error_envs_mask
        return self._truncated_buf | self._terminated_buf

    def get_term(self, name: str) -> torch.Tensor:
        return self._term_dones[name]

    def _prepare_terms(self):
        self._terms: dict[str, TerminationTermLike] = dict()
        self._reset_terms: list[TerminationTerm] = []

        for term_name in self._options.term_keys():
            term: TerminationTermLike = self._build_term(term_name, TERMINATION_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
            if isinstance(term, TerminationTerm):
                self._reset_terms.append(term)
