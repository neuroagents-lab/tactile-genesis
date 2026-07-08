"""Base manager-options class with dynamic per-term fields."""

from typing import Generic, TypeVar

from eden.options.options import ConfigurableOptions


T_TermOptions = TypeVar("T_TermOptions", bound=ConfigurableOptions)


def _disable_extra_term(options: ConfigurableOptions, name: str) -> None:
    """Remove a term stored as a pydantic-extra field by ``name``.

    Shared helper for ``ManagerOptions.disable_term`` and observation-group's
    ``disable_term`` (both store their per-term options as ``__pydantic_extra__``).
    Raises if the holder is locked or the name is missing.
    """
    if getattr(options, "_locked", False):
        raise RuntimeError(f"Cannot disable term '{name}' because {type(options).__name__} is locked.")
    extra = options.__pydantic_extra__
    if extra and name in extra:
        del extra[name]
    else:
        raise KeyError(f"Term '{name}' not found: {extra.keys() if extra else ()}")


class ManagerOptions(ConfigurableOptions, Generic[T_TermOptions]):
    """
    Generic base class for manager options.

    Stores term options as extra fields, validates their types via ``model_post_init``,
    and provides ``disable_term`` for runtime removal.

    Usage::

        class EventManagerOptions(ManagerOptions[EventTermOptions]):
            pass

    The term options class is automatically extracted from the generic parameter.
    """

    _term_options_class_: type | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Pydantic rewrites __orig_bases__, so extract the type arg from __bases__
        # via __pydantic_generic_metadata__ on the parametrized base class.
        for base in cls.__bases__:
            meta = getattr(base, "__pydantic_generic_metadata__", None)
            if meta and meta.get("origin") is ManagerOptions and meta.get("args"):
                cls._term_options_class_ = meta["args"][0]
                break

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        term_cls = self._term_options_class_
        if term_cls is None:
            return
        extra = self.__pydantic_extra__ or {}
        for key, value in extra.items():
            if key.startswith("_option_"):
                continue
            if not isinstance(value, term_cls):
                raise TypeError(f"Term '{key}' must be an instance of {term_cls.__name__}, got {type(value).__name__}")

    def term_keys(self) -> tuple[str, ...]:
        """Return only the registered ``ManagerTerm`` entry names.

        These are the extras validated by :meth:`model_post_init` to be
        ``_term_options_class_`` instances.

        This is intentionally distinct from :meth:`ConfigurableOptions.keys`,
        which returns every named child of the options object — including
        declared model fields that are config knobs (e.g.
        ``RecorderManagerOptions.dataset_filename``), not terms.
        """
        extra = self.__pydantic_extra__ or {}
        return tuple(key for key in extra if not key.startswith("_option_"))

    def disable_term(self, name: str) -> None:
        """Remove a term from this manager options by name."""
        _disable_extra_term(self, name)
