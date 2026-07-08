"""Event manager and event-term configuration options."""

from pydantic import field_validator

from eden.constants import EventMode
from eden.options.managers.base import ManagerOptions
from eden.options.options import ConfigurableOptions


class EventTermOptions(ConfigurableOptions):
    """
    Event term specification.

    Parameters
    ----------
    mode: EventMode | str
        The mode of the event term. Accepts either an ``EventMode`` member
        or its string value (e.g. ``"reset"``).
    interval_range_s: tuple[float, float] | None
        The range of the interval in seconds.
    is_global_time: bool
        Whether the event term is global time.
    min_step_count_between_reset: int
        The minimum number of steps between resets.
    priority: int
        The priority of the event term. Lower priority numbers run first within the same mode.
        Default is 0 (highest priority).
    """

    mode: EventMode = EventMode.RESET
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0
    priority: int = 0

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v):
        return EventMode(v) if isinstance(v, str) else v


class EventManagerOptions(ManagerOptions[EventTermOptions]):
    """
    Event manager options.

    Parameters
    ----------
    <event_term_name>: EventTermOptions
        The event terms configuration to be used.
    """
