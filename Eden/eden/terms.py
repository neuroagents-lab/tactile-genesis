"""Short aliases for FuncWrapper term classes.

Usage::

    from eden.terms import ObsTerm, RewardTerm, DoneTerm, EventTerm, CurrTerm, MetricTerm
"""

from eden.managers.observation_manager import ObservationTermFuncWrapper as ObsTerm
from eden.managers.reward_manager import RewardTermFuncWrapper as RewardTerm
from eden.managers.termination_manager import TerminationTermFuncWrapper as DoneTerm
from eden.managers.event_manager import EventTermFuncWrapper as EventTerm
from eden.managers.curriculum_manager import CurriculumTermFuncWrapper as CurrTerm
from eden.managers.metric_manager import MetricTermFuncWrapper as MetricTerm

__all__ = [
    "ObsTerm",
    "RewardTerm",
    "DoneTerm",
    "EventTerm",
    "CurrTerm",
    "MetricTerm",
]
