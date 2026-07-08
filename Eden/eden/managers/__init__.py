"""Manager classes that compose pluggable MDP terms (actions, observations, rewards, ...)."""

from eden.managers.action_manager import ActionManager, ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.command_manager import (
    CommandManager,
    COMMAND_TERM_REGISTRY,
    CommandTerm,
)
from eden.managers.curriculum_manager import (
    CurriculumManager,
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
    CurriculumTermFuncWrapper,
)
from eden.managers.event_manager import (
    EventManager,
    EVENT_TERM_REGISTRY,
    EventTerm,
    EventTermFuncWrapper,
)
from eden.managers.observation_manager import (
    ObservationManager,
    OBSERVATION_TERM_REGISTRY,
    ObservationTerm,
    ObservationTermFuncWrapper,
)
from eden.managers.reward_manager import (
    RewardManager,
    REWARD_TERM_REGISTRY,
    RewardTerm,
    RewardTermFuncWrapper,
)
from eden.managers.termination_manager import (
    TerminationManager,
    TERMINATION_TERM_REGISTRY,
    TerminationTerm,
    TerminationTermFuncWrapper,
)
from eden.managers.metric_manager import (
    MetricManager,
    METRIC_TERM_REGISTRY,
    MetricTerm,
    MetricTermFuncWrapper,
)

from eden.managers.recorder_manager import (
    RecorderManager,
    RECORDER_TERM_REGISTRY,
    RecorderTerm,
)

__all__ = [
    "ActionManager",
    "ACTION_TERM_REGISTRY",
    "ActionTerm",
    "CommandManager",
    "COMMAND_TERM_REGISTRY",
    "CommandTerm",
    "CurriculumManager",
    "CURRICULUM_TERM_REGISTRY",
    "CurriculumTerm",
    "CurriculumTermFuncWrapper",
    "EventManager",
    "EVENT_TERM_REGISTRY",
    "EventTerm",
    "EventTermFuncWrapper",
    "ObservationManager",
    "OBSERVATION_TERM_REGISTRY",
    "ObservationTerm",
    "ObservationTermFuncWrapper",
    "RewardManager",
    "REWARD_TERM_REGISTRY",
    "RewardTerm",
    "RewardTermFuncWrapper",
    "TerminationManager",
    "TERMINATION_TERM_REGISTRY",
    "TerminationTerm",
    "TerminationTermFuncWrapper",
    "MetricManager",
    "METRIC_TERM_REGISTRY",
    "MetricTerm",
    "MetricTermFuncWrapper",
    "RecorderManager",
    "RECORDER_TERM_REGISTRY",
    "RecorderTerm",
]
