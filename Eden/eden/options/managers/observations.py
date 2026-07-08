"""Observation manager, group, and term configuration options."""

from typing import ClassVar, TYPE_CHECKING

from genesis.typing import Vec2FType
from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions, _disable_extra_term
from eden.options.managers.modifiers import NoiseOptions


class ObservationTermOptions(ConfigurableOptions):
    """
    Observation term specification.

    Parameters
    ----------
    noise: float, optional
        Standard deviation of Gaussian noise added to the observation (default=0.0).
    clip: array-like[float, float] or None, optional
        Range to clip the observation values (default=None, no clipping).
    scale: float, optional
        Scaling factor applied to the observation (default=1.0).
    history_length: int, optional
        Number of previous observations to include (default=0, no history).
    flatten_history_dim: bool, optional
        Whether to flatten the history dimension (default=False).
    backfill: bool, optional
        Whether to backfill the history with the first observation (default=True).
        If False, the history will be filled with zeros.
    """

    noise: NoiseOptions | None = None
    clip: Vec2FType | None = None
    scale: float = 1.0
    history_length: int = 0
    flatten_history_dim: bool = False
    backfill: bool = True

    #: Single source of truth for the post-compute modifier fields — those applied by the manager *after*
    #: ``term.compute()`` (so they don't change a term's raw output). Co-located with the field declarations
    #: above so the set can't drift from the schema. Used by the observation manager to (a) exclude these
    #: from the cross-group dedup signature and (b) treat them as options-only params not mirrored as term
    #: attributes. Keep in sync with the fields above when editing.
    POST_COMPUTE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"noise", "clip", "scale", "history_length", "flatten_history_dim", "backfill"}
    )


class ObservationGroupOptions(ConfigurableOptions):
    """
    Observation group options.

    Parameters
    ----------
    concatenate_terms: bool, optional
        Whether to concatenate the terms (default=True).
    concatenate_dim: int, optional
        The dimension to concatenate the terms (default=-1).
    terms_order: list[str] | None, optional
        The order of the terms in the group (default=None).
    enable_corruption: bool, optional
        Whether to allow adding noise to each term in the group (default=True).
    history_length: int | None, optional
        The history length enforced for all terms in the group (default=None).
    flatten_history_dim: bool, optional
        Whether to flatten the history dimension (default=True).
    backfill: bool, optional
        Whether to backfill the history with the first observation (default=True).
        If False, the history will be filled with zeros.
    <term_name>: ObservationTermOptions, optional
        The observation term configuration to be used for the given group.
    """

    concatenate_terms: bool = True
    concatenate_dim: int = -1
    terms_order: list[str] | None = None
    enable_corruption: bool = True
    history_length: int | None = None
    flatten_history_dim: bool = True
    backfill: bool = True

    if TYPE_CHECKING:
        # NOTE: these will be set automatically in the observation manager
        terms: dict[str, ObservationTermOptions]

    def model_post_init(self, context) -> None:
        super().model_post_init(context)

        if self.history_length is not None:
            if self.history_length <= 0:
                raise ValueError("History length must be greater than 0!")
            if not self.concatenate_terms:
                raise ValueError("Concatenate terms must be True if history length is enforced for the group!")

        # Validate terms_order against provided term names when possible.
        # Pydantic guarantees extras and declared fields never overlap, so it is
        # enough to filter out the metadata keys injected by ConfigurableOptions.
        extra = getattr(self, "__pydantic_extra__", None) or {}
        term_names = [key for key in extra if not key.startswith("_option_")]
        if self.terms_order is not None and term_names:
            for name in self.terms_order:
                if name not in term_names:
                    raise ValueError(f"Term `{name}` not found in the group!")
            for name in term_names:
                if name not in self.terms_order:
                    raise ValueError(f"Term `{name}` not found in the terms order!")

    def disable_term(self, name: str) -> None:
        """Remove a term from this manager options by name."""
        _disable_extra_term(self, name)


class ObservationManagerOptions(ManagerOptions[ObservationGroupOptions]):
    """
    Observation manager options.

    Parameters
    ----------
    <group_name>: ObservationGroupOptions
        The observation terms configuration to be used for the given group.
    """
