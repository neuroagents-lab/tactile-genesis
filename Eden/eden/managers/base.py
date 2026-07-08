"""Base classes and shared logic for managers and their terms.

A manager owns an ordered set of *terms* resolved from its options through a
:class:`~eden.utils.registry.Registry`. Terms come in two flavors:

- function-based (stateless): a plain function registered with ``@REGISTRY.register()``
  and wrapped in a ``FuncWrapper``.
- class-based (stateful): subclass :class:`ManagerTermBase` + ``ConfigurableMixin`` and
  implement ``compute()``, optionally ``build()`` and ``reset()``.

Concrete managers (action, observation, reward, termination, ...) live in sibling
modules and are imported lazily inside the env's ``_load_managers()`` to break the
manager <-> env circular import.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

import torch
from genesis.repr_base import RBC
from prettytable import PrettyTable

from eden.options.options import Options
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


T = TypeVar("T", bound=Options)


class ManagerTermBase(RBC):
    """Base class for all manager terms.

    Manager-owned output cache
    --------------------------
    Each manager pre-allocates one contiguous buffer holding every term's output
    and hands each term a view into it via :attr:`_cache` (set after shapes are
    known, in the manager's build). A cache-aware :meth:`compute` writes its
    result straight into ``self._cache`` (e.g. ``torch.sum(..., out=self._cache)``
    or ``self._cache.copy_(...)``) and returns it; the manager then reads the
    output from its buffer with no per-term allocation or copy. This mirrors the
    Genesis sensors pipeline, which writes every reading into one shared cache.

    ``self._cache`` is ``None`` until the manager assigns it — and during the
    build-time shape probe (the manager calls ``compute()`` once to learn each
    term's output shape *before* the buffer exists). A cache-aware ``compute``
    must therefore allocate-and-return when ``self._cache is None`` and
    write-into-and-return ``self._cache`` otherwise. Terms that ignore
    ``self._cache`` entirely (all function-wrapped terms) just return a fresh
    tensor and the manager copies it into the buffer — so this is fully backward
    compatible.

    Writing into the cache is worthwhile for *computed* terms whose final op can
    target it (norms, reductions, ``exp``); for pure state reads (returning an
    entity getter's tensor verbatim) the copy into the buffer is unavoidable
    either way, so those terms gain nothing and may leave ``compute`` cache-
    unaware — their win comes from caching the entity/indices in :meth:`build`.
    """

    #: Manager-owned view this term writes its output into; ``None`` until the
    #: manager assigns it (and during the build-time shape probe). See the class
    #: docstring. Underscore-prefixed so ``ConfigurableMixin`` does not treat it
    #: as a configurable parameter.
    _cache: torch.Tensor | None = None

    def __init__(self, env: EnvBase):
        self._env = env

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        """The torch device the term's tensors live on."""
        return self._env.device

    def reset(self, envs_idx: torch.Tensor | None = None) -> Any:
        """Reset the manager term."""
        del envs_idx  # Unused.

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Step the manager term.

        Cache-aware terms write their result into ``self._cache`` (when set) and
        return it; see the class docstring for the cache protocol.
        """
        raise NotImplementedError

    def compute_cached(self) -> Any:
        """Run :meth:`compute` and guarantee the result lands in ``self._cache``.

        This is what managers call so they never have to branch on the return
        value. A cache-aware term writes its result straight into ``self._cache``
        (``out=self._cache``) and returns it, so the identity check below is True
        and **no copy happens** — zero-copy. A term that returns a fresh tensor
        (function-wrapped terms, bool→float penalties, compiled-helper results,
        and any class term that hasn't opted in) is copied into the cache once,
        here. When no cache is assigned (the build-time shape probe, or managers
        that don't use the cache) the result is returned unchanged.
        """
        result = self.compute()
        if self._cache is not None and result is not self._cache:
            self._cache.copy_(result)
        return self._cache if self._cache is not None else result

    def build(self) -> None:
        """Deferred build hook for terms that resolve entities after env setup."""
        return None


class ManagerTermFuncWrapperBase(ManagerTermBase, ConfigurableMixin):
    """Base class for manager terms defined as function."""

    params: dict[str, Any] = {}

    def __init__(self, func: FunctionType, env: EnvBase):
        self._func = func
        self._env = env

    def compute(self, *args, **kwargs) -> Any:
        """Invoke the wrapped function with the environment and configured params."""
        return self._func(env=self._env, **self.params, **kwargs)


class ManagerBase(RBC, Generic[T], metaclass=ABCMeta):
    """Base class for all managers."""

    def __init__(self, env: EnvBase, options: T):
        self._env = env
        self._options: T = options

        self._term_names: list[str] = list()
        self._terms: dict[str, ManagerTermBase] = dict()
        self._prepare_terms()

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """The torch device the manager's tensors live on."""
        return self._env.device

    @property
    def terms(self) -> dict[str, ManagerTermBase]:
        """Mapping from term name to the built :class:`ManagerTermBase` instance."""
        return self._terms

    @property
    def active_terms(self) -> list[str]:
        """The names of the active terms, in registration order."""
        return self._term_names

    def get_term(self, name: str) -> ManagerTermBase:
        """Return the active term registered under ``name``."""
        return self._terms[name]

    def get_term_options(self, term_name: str) -> Options:
        """Return the options object for the active term ``term_name``.

        Raises
        ------
        ValueError
            If ``term_name`` is not an active term.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._terms[term_name].options

    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, Any]:
        """Reset the manager and return logging info for the current step."""
        del envs_idx  # Unused.
        return {}

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Compute the manager."""
        raise NotImplementedError

    @abstractmethod
    def _prepare_terms(self):
        raise NotImplementedError

    def _build_term(self, term_name: str, registry: Registry) -> ManagerTermBase:
        """Resolve and build a term from this manager's options.

        Centralizes the ``getattr(options, term_name)`` → ``registry.get(term_option.name)``
        → ``term.build()`` sequence shared by every manager's ``_prepare_terms``.
        """
        term_option = getattr(self._options, term_name)
        term = registry.get(term_option.name)(env=self._env, options=term_option)
        term.build()
        return term

    def _format_summary_table(
        self,
        *,
        title: str,
        field_names: list[str],
        rows: Iterable[Iterable[Any]],
        align: dict[str, str] | None = None,
    ) -> str:
        r"""Build the canonical ``<{Manager}> contains N active terms.\\n<table>\\n`` summary string."""
        msg = f"<{type(self).__name__}> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = title
        table.field_names = field_names
        if align:
            for col, a in align.items():
                table.align[col] = a
        for row in rows:
            table.add_row(list(row))
        msg += table.get_string()
        msg += "\n"
        return msg
