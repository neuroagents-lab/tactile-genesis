"""Metric manager and MetricTerm base for task success / evaluation signals."""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, TypeAlias

import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.base import ManagerBase, ManagerTermBase, ManagerTermFuncWrapperBase
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class SuccessCheckerMixin:
    """Mixin for checking if a metric is successful."""

    if TYPE_CHECKING:
        _env: EnvBase
        success_threshold: float
        direction: MetricDirection
        metric_mode: MetricMode
        is_cumulative: bool

    def is_success(self, value: torch.Tensor, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Check whether a metric is successful (default implementation).

        This method can be overridden by subclasses to provide a custom implementation.

        Parameters
        ----------
        value: torch.Tensor
            The value of the metric.
        envs_idx: torch.Tensor, optional
            The indices of the environments to evaluate. If None, all environments are used. Defaults to None.

        Returns
        -------
        torch.Tensor
            A boolean tensor indicating whether the metric is successful.
        """
        if envs_idx is None:
            envs_idx = slice(None)
        if self.is_cumulative:
            value_norm = value[envs_idx] / torch.clamp_min(self._env.episode_length_buf[envs_idx], 1)
        else:
            value_norm = value[envs_idx]
        if self.direction == MetricDirection.HIB:
            return value_norm >= self.success_threshold
        elif self.direction == MetricDirection.LIB:
            return value_norm <= self.success_threshold
        else:
            raise ValueError(f"Invalid metric direction: {self.direction}")


class MetricTerm(ManagerTermBase, ConfigurableMixin[MetricTermOptions], SuccessCheckerMixin):
    """
    Base class for metric terms.

    Parameters
    ----------
    success_threshold: float
        The threshold for success. The threshold is inclusive. Defaults to 1.0.
    direction: MetricDirection
        The metric direction. Defaults to ``MetricDirection.HIB`` (``"hib"``).
    metric_mode: MetricMode
        The metric success mode. Defaults to ``MetricMode.INTERVAL`` (``"interval"``).
    is_cumulative: bool
        Whether the metric is cumulative. Defaults to False.
    """

    success_threshold: float = 1.0
    direction: MetricDirection = MetricDirection.HIB
    metric_mode: MetricMode = MetricMode.INTERVAL
    is_cumulative: bool = False

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        # NOTE: resets the term state if needed.
        pass


class MetricTermFuncWrapper(
    ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[MetricTermOptions], SuccessCheckerMixin
):
    """
    Base class for metric terms defined as function.

    Parameters
    ----------
    success_threshold: float
        The threshold for success. The threshold is inclusive. Defaults to 1.0.
    direction: MetricDirection
        The metric direction. Defaults to ``MetricDirection.HIB`` (``"hib"``).
    metric_mode: MetricMode
        The metric success mode. Defaults to ``MetricMode.INTERVAL`` (``"interval"``).
    is_cumulative: bool
        Whether the metric is cumulative. Defaults to False.
    """

    success_threshold: float = 1.0
    direction: MetricDirection = MetricDirection.HIB
    metric_mode: MetricMode = MetricMode.INTERVAL
    is_cumulative: bool = False

    def __init__(self, func: FunctionType, env: EnvBase, options: MetricTermOptions):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)


MetricTermLike: TypeAlias = MetricTerm | MetricTermFuncWrapper
"""Union of a class-based metric term and its function-wrapper form."""


METRIC_TERM_REGISTRY = TermRegistry("METRIC_TERM", MetricTerm, MetricTermFuncWrapper)


class MetricManager(ManagerBase[MetricManagerOptions]):
    """Manager for computing metrics."""

    def __init__(self, env: EnvBase, options: MetricManagerOptions):
        super().__init__(env=env, options=options)
        self._step_metric = {}
        for term_name in self._term_names:
            self._step_metric[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._progress_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

    @property
    def success_buf(self) -> torch.Tensor:
        return self._success_buf

    @property
    def progress_buf(self) -> torch.Tensor:
        return self._progress_buf

    def summary(self) -> str:
        return self._format_summary_table(
            title="Metrics Terms",
            field_names=[
                "Index",
                "Name",
                "Success Threshold",
                "Direction",
                "Success Mode",
                "Cumulative",
            ],
            rows=(
                [
                    index,
                    name,
                    term.success_threshold,
                    term.direction,
                    term.metric_mode,
                    term.is_cumulative,
                ]
                for index, (name, term) in enumerate(self._terms.items())
            ),
            align={"Name": "l"},
        )

    @torch.inference_mode()
    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if envs_idx is None:
            envs_idx = slice(None)
        extras = {}
        for name, term in self._terms.items():
            if term.is_cumulative:
                val = self._step_metric[name][envs_idx] / torch.clamp_min(self._env.episode_length_buf[envs_idx], 1)
            else:
                val = self._step_metric[name][envs_idx].float()
            extras[f"Metrics/{name}"] = val.sum() / max(val.numel(), 1)
            # NOTE: check if the episode maintained the success status on the reset.
            if term.metric_mode == MetricMode.RESET:
                is_success = term.is_success(self._step_metric[name], envs_idx=envs_idx)
                self._progress_buf[envs_idx] += is_success.long()
            self._step_metric[name][envs_idx] = 0.0

        # NOTE: compute the success and progress metrics based on the maintenance and attainment terms.
        success_val = (self._progress_buf[envs_idx] >= self._num_terms).float()
        extras["Metrics/Success"] = success_val.sum() / max(success_val.numel(), 1)
        progress_val = self._progress_buf[envs_idx] / self._num_terms
        extras["Metrics/Progress"] = progress_val.sum() / max(progress_val.numel(), 1)
        self._progress_buf[envs_idx] = 0
        self._success_buf[envs_idx] = 0.0
        for terms in self._reset_terms:
            terms.reset(envs_idx=envs_idx)
        return extras

    @torch.inference_mode()
    def compute(self) -> dict[str, torch.Tensor]:
        if len(self._terms) == 0:
            return self._step_metric
        self._progress_buf.zero_()
        self._success_buf.zero_()
        for name, term in self._terms.items():
            value = term.compute().float()
            if term.is_cumulative:
                self._step_metric[name] += value
            else:
                self._step_metric[name][:] = value

            if term.metric_mode == MetricMode.INTERVAL:
                is_success = term.is_success(self._step_metric[name])
                self._progress_buf += is_success.long()
        if self._num_interval_terms > 0:
            self._success_buf[:] = self._progress_buf >= self._num_interval_terms
        return self._step_metric

    def _prepare_terms(self):
        self._terms: dict[str, MetricTermLike] = dict()
        self._reset_terms: list[MetricTerm] = []
        self._num_terms = 0
        self._num_interval_terms = 0

        # First pass: build all terms normally
        for term_name in self._options.term_keys():
            term: MetricTermLike = self._build_term(term_name, METRIC_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
            if isinstance(term, MetricTerm):
                self._reset_terms.append(term)
            self._num_terms += 1
            if term.metric_mode == MetricMode.INTERVAL:
                self._num_interval_terms += 1

        # Second pass: resolve refs for terms that need it (e.g. SequentialMetricTerm)
        for term in self._terms.values():
            if hasattr(term, "resolve_refs"):
                term.resolve_refs(self._options)
