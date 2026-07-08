"""Reward manager and reward-term configuration options."""

from pydantic import field_validator

from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions


class RewardTermOptions(ConfigurableOptions):
    """
    Reward term specification.

    Parameters
    ----------
    range_s: tuple[float, float] | None
        The temporal range in seconds where the reward is active given as (start_s, end_s) or a list of such tuples.
        The both ends are inclusive. If None, the reward is active for the entire episode. Default is None.
    weight: float
        A reward weight for the term.
    tags: list[str]
        Optional labels for grouping reward terms (e.g. ``["penalty"]`` or ``["tracking", "feet"]``).
        Curricula and other consumers can filter terms by tag instead of hard-coding term names.
    """

    range_s: tuple[float, float] | list[tuple[float, float]] | None = None
    weight: float = 1.0
    tags: list[str] = []

    @field_validator("range_s", mode="before")
    @classmethod
    def _wrap_single_range(cls, v):
        # accept a single (start_s, end_s) tuple as shorthand for [(start_s, end_s)]
        return [v] if isinstance(v, tuple) else v


class RewardManagerOptions(ManagerOptions[RewardTermOptions]):
    """
    Reward manager options.

    Parameters
    ----------
    <reward_term_name>: RewardTermOptions
        The reward terms configuration to be used.
    """
