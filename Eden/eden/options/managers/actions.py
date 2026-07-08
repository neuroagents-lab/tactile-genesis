"""Action manager and action-term configuration options."""

from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions


class ActionTermOptions(ConfigurableOptions):
    """
    Action term specification.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: str | list[str]
        The names of the DOFs to control.
    """

    entity_name: str
    dofs_name: str | list[str] = []


class ActionManagerOptions(ManagerOptions[ActionTermOptions]):
    """
    Action manager options.

    Parameters
    ----------
    <action_term_name>: ActionTermOptions
        The action terms configuration to be used.
    """
