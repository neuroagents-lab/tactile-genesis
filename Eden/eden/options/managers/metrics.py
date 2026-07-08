"""Metric manager and metric-term configuration options."""

from pydantic import field_validator

from eden.constants import MetricDirection, MetricMode
from eden.options.managers.base import ManagerOptions
from eden.options.options import ConfigurableOptions


class MetricTermOptions(ConfigurableOptions):
    """
    Metric term specification.

    Parameters
    ----------
    success_threshold: float
        The threshold for success. The threshold is inclusive. Defaults to 1.0.
    direction: MetricDirection | str
        The metric direction. Accepts either a ``MetricDirection`` member
        or its string value (e.g. ``"hib"``). Defaults to ``"hib"``.
    metric_mode: MetricMode | str
        The metric success mode. Accepts either a ``MetricMode`` member
        or its string value. Defaults to ``"interval"``.
    is_cumulative: bool
        Whether the metric is cumulative. Defaults to False.
    """

    success_threshold: float = 1.0
    direction: MetricDirection = MetricDirection.HIB
    metric_mode: MetricMode = MetricMode.INTERVAL
    is_cumulative: bool = False

    @field_validator("direction", mode="before")
    @classmethod
    def _coerce_direction(cls, v):
        return MetricDirection(v) if isinstance(v, str) else v

    @field_validator("metric_mode", mode="before")
    @classmethod
    def _coerce_metric_mode(cls, v):
        return MetricMode(v) if isinstance(v, str) else v


class PhaseOptions(ConfigurableOptions):
    """Configuration for a single phase in SequentialMetricTerm.

    Parameters
    ----------
    term : MetricTermOptions, optional
        Inline term configuration (from ``.configure()``).  Mutually exclusive with ``ref``.
    ref : str, optional
        Reference to a sibling term in ``MetricManagerOptions``.  Mutually exclusive with ``term``.
    name : str
        Phase display name.  Auto-derived from the term if empty.
    threshold : float
        Phase advancement threshold.  Defaults to 1.0.
    hold : bool
        If ``True``, reuse the phase predicate as hold condition.
    hold_term : MetricTermOptions, optional
        Separate hold predicate (overrides ``hold``).
    hold_until : str or int, optional
        Scoped hold expiration — a phase name or index.
    hold_guard : str, optional
        Name of another phase whose hold must **also** be failing for this
        hold to be enforced.  If the guard phase's hold is still passing,
        this hold is suppressed.  For example, ``hold_guard="grasp"`` on
        a lift hold means: "only revert for lift failure if the robot also
        lost its grasp — lowering while grasping is intentional."
    """

    term: MetricTermOptions | None = None
    ref: str | None = None
    name: str = ""
    threshold: float = 1.0
    hold: bool = False
    hold_term: MetricTermOptions | None = None
    hold_until: (str | int) | None = None
    hold_guard: str | None = None


class MetricManagerOptions(ManagerOptions[MetricTermOptions]):
    """
    Metric manager options.

    Parameters
    ----------
    <metric_term_name>: MetricTermOptions
        The metric terms configuration to be used.
    """
