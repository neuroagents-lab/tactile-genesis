"""Configuration options for action/observation modifiers (including noise)."""

from eden.constants import NoiseOperation
from eden.options.options import ConfigurableOptions


class NoiseOptions(ConfigurableOptions):
    """
    Noise options.

    Parameters
    ----------
    operation: NoiseOperation, optional
        The operation to apply to the noise. Default is NoiseOperation.ADD.
    """

    operation: NoiseOperation = NoiseOperation.ADD


class ActionModifierOptions(ConfigurableOptions):
    """Options for action modifiers."""

    ...
