"""Termination manager and termination-term configuration options."""

from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions


class TerminationTermOptions(ConfigurableOptions):
    """
    Termination term specification.

    Parameters
    ----------
    time_out: bool
        Whether the term contributes towards episodic timeouts. Defaults to False.
    """

    time_out: bool = False


class TerminationManagerOptions(ManagerOptions[TerminationTermOptions]):
    """
    Termination manager options.

    Parameters
    ----------
    <termination_term_name>: TerminationTermOptions
        The termination terms configuration to be used.
    """
