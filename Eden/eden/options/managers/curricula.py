"""Curriculum manager and curriculum-term configuration options."""

from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions


class CurriculumTermOptions(ConfigurableOptions):
    """Curriculum term specification."""


class CurriculumManagerOptions(ManagerOptions[CurriculumTermOptions]):
    """
    Curriculum manager options.

    Parameters
    ----------
    <curriculum_term_name>: CurriculumTermOptions
        The curriculum terms configuration to be used.
    """
