"""Built-in termination terms."""

from eden.managers.terms.terminations.common import (
    time_out,
    on_success,
    on_failure,
    on_metric_success,
    illegal_contact,
    cumulative_reward_below_threshold,
)
from eden.managers.terms.terminations.manipulation import (
    object_out_of_reach,
)


__all__ = [
    "time_out",
    "on_success",
    "on_failure",
    "on_metric_success",
    "illegal_contact",
    "cumulative_reward_below_threshold",
    "object_out_of_reach",
]
