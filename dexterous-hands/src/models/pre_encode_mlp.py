from typing import Any

from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlActorRNNOptions,
    RslRlMLPModelOptions,
    RslRlRNNModelOptions,
)

# The PreEncodeMLPModel implementation is in the Milotrince/rsl_rl repo.


class RslRlPreEncodeMLPOptions(RslRlMLPModelOptions):
    """Actor model configuration with with some observations passed through encoder MLPs before the main MLP head.

    Parameters
    ----------
    encoder_cfg: dict[str, Any]
    """

    class_name: str = "PreEncodeMLPModel"
    encoder_cfg: dict[str, Any] | None = None


class RslRlPreEncodeRNNOptions(RslRlRNNModelOptions):
    """Actor model configuration with with some observations passed through encoder MLPs before the main RNN head.

    Parameters
    ----------
    encoder_cfg: dict[str, Any]
    """

    class_name: str = "PreEncodeRecurrentModel"
    encoder_cfg: dict[str, Any] | None = None


class RslRlActorPreEncodeMLPOptions(RslRlActorOptions):
    """Actor model configuration with with some observations passed through encoder MLPs before the main MLP head.

    Parameters
    ----------
    encoder_cfg: dict[str, Any]
    """

    class_name: str = "PreEncodeMLPModel"
    encoder_cfg: dict[str, Any] | None = None


class RslRlActorPreEncodeRNNOptions(RslRlActorRNNOptions):
    """Actor model configuration with with some observations passed through encoder MLPs before the main RNN head.

    Parameters
    ----------
    encoder_cfg: dict[str, Any]
    """

    class_name: str = "PreEncodeRecurrentModel"
    encoder_cfg: dict[str, Any] | None = None


# --- Tactile per-sensor encoders (CNN / IntersectionRNN ConvRNN) --------
# Implementation lives in models/tactile_pre_encode.py; resolved via qualified
# name so rsl_rl.utils.resolve_callable finds it (simple-name search is
# rsl_rl-internal only).

_TACTILE_MLP_CLASS = "models.tactile_pre_encode:TactilePreEncodeMLPModel"
_TACTILE_RNN_CLASS = "models.tactile_pre_encode:TactilePreEncodeRecurrentModel"


class RslRlTactilePreEncodeMLPOptions(RslRlMLPModelOptions):
    """Critic-style options for the tactile pre-encode MLP model."""

    class_name: str = _TACTILE_MLP_CLASS
    encoder_cfg: dict[str, Any] | None = None


class RslRlActorTactilePreEncodeMLPOptions(RslRlActorOptions):
    """Actor-style options for the tactile pre-encode MLP model."""

    class_name: str = _TACTILE_MLP_CLASS
    encoder_cfg: dict[str, Any] | None = None


class RslRlTactilePreEncodeRNNOptions(RslRlRNNModelOptions):
    """Critic-style options for the tactile pre-encode RNN model."""

    class_name: str = _TACTILE_RNN_CLASS
    encoder_cfg: dict[str, Any] | None = None


class RslRlActorTactilePreEncodeRNNOptions(RslRlActorRNNOptions):
    """Actor-style options for the tactile pre-encode RNN model."""

    class_name: str = _TACTILE_RNN_CLASS
    encoder_cfg: dict[str, Any] | None = None
