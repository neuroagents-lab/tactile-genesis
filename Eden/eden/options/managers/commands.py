"""Command manager and command-term configuration options."""

from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions


class CommandTermOptions(ConfigurableOptions):
    """
    Command term specification.

    Parameters
    ----------
    resampling_time_range: tuple[float, float]
        The range of time before commands are changed [s].
    debug_vis: bool
        Whether to visualize debug information. Defaults to False.
    """

    resampling_time_range: tuple[float, float] = None
    debug_vis: bool = False


class CommandManagerOptions(ManagerOptions[CommandTermOptions]):
    """
    Command manager options.

    Parameters
    ----------
    <command_term_name>: CommandTermOptions
        The command terms configuration to be used.
    """
