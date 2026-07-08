"""Boolean combinators (AND/OR/NOT) over metric terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.options.managers.metrics import MetricTermOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class _BooleanMetricTermBase(MetricTerm):
    """Base class for boolean composite metric terms.

    Manages a list of child terms instantiated from ``MetricTermOptions``.
    Subclasses implement ``_combine`` to define how child outputs are merged.
    """

    terms: list[MetricTermOptions] = []

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env=env, options=options)
        self._child_terms: list[MetricTerm] = []

    def _instantiate_term(self, term_options: MetricTermOptions) -> MetricTerm:
        term = METRIC_TERM_REGISTRY.get(term_options.name)(env=self._env, options=term_options)
        term.build()
        self._child_terms.append(term)
        return term

    def build(self):
        if len(self.terms) == 0:
            raise ValueError(f"{type(self).__name__} requires at least one term in 'terms'")
        for i, term_opts in enumerate(self.terms):
            if not isinstance(term_opts, MetricTermOptions):
                raise TypeError(
                    f"terms[{i}] must be a MetricTermOptions (from .configure()), got {type(term_opts).__name__}"
                )
            self._instantiate_term(term_opts)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        for child in self._child_terms:
            child.reset(envs_idx=envs_idx)

    def resolve_refs(self, manager_options) -> None:
        """Resolve ``ref``-style terms if needed in the future."""
        pass

    def compute(self) -> torch.Tensor:
        values = [child.compute().float() for child in self._child_terms]
        return self._combine(values)

    def _combine(self, values: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


@METRIC_TERM_REGISTRY.register()
class MetricTermAND(_BooleanMetricTermBase):
    """Metric term that returns the element-wise minimum of its child terms.

    All child terms must be "passing" for this term to pass.

    Parameters
    ----------
    terms : list[MetricTermOptions]
        Child term configurations (from ``.configure()``).

    Example
    -------
    ::

        MetricTermAND.configure(
            terms=[
                EeNearEntity.configure(robot_name="robot", ...),
                IsGrasping.configure(robot_name="robot", ...),
            ],
        )
    """

    def _combine(self, values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).min(dim=0).values


@METRIC_TERM_REGISTRY.register()
class MetricTermOR(_BooleanMetricTermBase):
    """Metric term that returns the element-wise maximum of its child terms.

    Any child term passing is sufficient for this term to pass.

    Parameters
    ----------
    terms : list[MetricTermOptions]
        Child term configurations (from ``.configure()``).

    Example
    -------
    ::

        MetricTermOR.configure(
            terms=[
                entity_near_target.configure(entity_name="apple", ...),
                entity_near_target.configure(entity_name="orange", ...),
            ],
        )
    """

    def _combine(self, values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).max(dim=0).values


@METRIC_TERM_REGISTRY.register()
class MetricTermNOT(MetricTerm):
    """Metric term that inverts a single child term (``1.0 - value``).

    Parameters
    ----------
    term : MetricTermOptions
        The child term configuration (from ``.configure()``).

    Example
    -------
    ::

        MetricTermNOT.configure(
            term=IsGrasping.configure(robot_name="robot", ...),
        )
    """

    term: MetricTermOptions = None

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env=env, options=options)
        self._child_term: MetricTerm | None = None

    def build(self):
        if self.term is None:
            raise ValueError("MetricTermNOT requires a 'term' option")
        if not isinstance(self.term, MetricTermOptions):
            raise TypeError(f"'term' must be a MetricTermOptions (from .configure()), got {type(self.term).__name__}")
        self._child_term = METRIC_TERM_REGISTRY.get(self.term.name)(env=self._env, options=self.term)
        self._child_term.build()

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if self._child_term is not None:
            self._child_term.reset(envs_idx=envs_idx)

    def resolve_refs(self, manager_options) -> None:
        pass

    def compute(self) -> torch.Tensor:
        return 1.0 - self._child_term.compute().float()
