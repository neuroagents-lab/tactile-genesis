"""Shared enums and constants (event modes, metric directions, reference sources)."""

from strenum import StrEnum


class EventMode(StrEnum):
    """
    Event mode.

    Options
    -------
    STARTUP: Event is only triggered at the startup of the environment.
    RESET: Event is triggered at the reset of the episode.
    INTERVAL: Event is triggered at the each environment step.
    """

    STARTUP = "startup"
    RESET = "reset"
    INTERVAL = "interval"


class NoiseOperation(StrEnum):
    """
    Noise operation.

    Options
    -------
    ADD: additive noise.
    SCALE: multiplicative noise.
    """

    ADD = "add"
    SCALE = "scale"


class MetricDirection(StrEnum):
    """
    Metric direction.

    Options
    -------
    LIB: lower is better.
    HIB: higher is better.
    """

    LIB = "lib"
    HIB = "hib"


class MetricMode(StrEnum):
    """
    Metric success mode.

    Options
    -------
    INTERVAL: evaluates at each step.
    RESET: evaluates at on the reset.
    """

    INTERVAL = "interval"
    RESET = "reset"


class ReferenceSource(StrEnum):
    """Source of the reference offset added on top of the joint PD controller setpoint.

    The offset is added on top of ``raw_action * scale + offset`` by the joint PD
    controllers in :mod:`eden.managers.terms.actions.joint_actions`.

    Options
    -------
    ZERO: no reference offset.
    DEFAULT: the entity's ``default_dofs_pos`` for the controlled DOFs.
    DELTA: per-step joint position captured from the entity at the start of
        each control step (``entity.get_dofs_pos(...)``); the action is then a
        delta on top of the previous step's DOF position. Constant across
        decimation substeps.
    """

    ZERO = "zero"
    DEFAULT = "default"
    DELTA = "delta"


class DatasetExportMode(StrEnum):
    """
    Dataset export mode.

    Options
    -------
    EXPORT_NONE: Export none of the episodes.
    EXPORT_ALL: Export all episodes to a single dataset file.
    EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES: Export succeeded and failed episodes in separate files.
    EXPORT_SUCCEEDED_ONLY: Export only succeeded episodes to a single dataset file.
    """

    EXPORT_NONE = "export_none"
    EXPORT_ALL = "export_all"
    EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES = "export_succeeded_failed_in_separate_files"
    EXPORT_SUCCEEDED_ONLY = "export_succeeded_only"
