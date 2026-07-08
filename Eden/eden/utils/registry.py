"""Generic Registry plus term/variant registries for pluggable components.

A :class:`Registry` maps name -> object so user code can register custom terms, tasks,
robots, etc. Auto-derived keys use ``to_snake_case(obj.__name__)`` (pass ``name=`` to
override, ``override=True`` to replace an existing entry). Two naming caveats: keys
starting with ``_`` emit a warning, and class names with a digit suffix (e.g. ``Foo1``)
do **not** gain an underscore before the digit — pass an explicit ``name=`` there.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence, Type

# Modified from fvcore
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import warnings
from types import FunctionType
from functools import partial
import inspect

from eden.utils.misc import to_snake_case

if TYPE_CHECKING:
    from eden.managers.base import ManagerTermFuncWrapperBase, ManagerTermBase


__all__ = ["Registry", "TermRegistry", "VariantRegistry"]


def _derive_name(obj: Any, explicit_name: str | None, registry_name: str) -> str:
    """Return the registry key for ``obj``.

    If ``explicit_name`` is given, it wins verbatim. Otherwise the object's
    ``__name__`` is converted to ``snake_case`` and a warning is emitted when
    the result starts with an underscore (likely a private class that should
    not be registered, or an intentional internal registration that should
    declare ``name=`` explicitly).
    """
    if explicit_name is not None:
        return explicit_name
    derived = to_snake_case(obj.__name__)
    if derived.startswith("_"):
        warnings.warn(
            f"Registering '{obj.__name__}' into '{registry_name}' produces the "
            f"leading-underscore key '{derived}'. Leading underscores usually mark "
            f"internals — pass an explicit name= if this is intentional.",
            stacklevel=3,
        )
    return derived


class Registry(Iterable[tuple[str, Any]]):
    """Registry providing a name -> object mapping to support third-party custom modules.

    Registered names default to ``to_snake_case(obj.__name__)``; pass
    ``name=`` to override.

    To create a registry (e.g. a robot registry):

    ```python

    ROBOT_REGISTRY = Registry('ROBOT')
    ```

    To register an object:

    ```python

    @ROBOT_REGISTRY.register()
    class Go1():
        ...
    ```

    Or:

    ```python

    ROBOT_REGISTRY.register(Go1)
    ```

    Parameters
    ----------
        name (str): the name of this registry
    """

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert name not in self._obj_map, (
            f"An object named {name} was already registered in {self._name} registry with keys {self._obj_map.keys()}"
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None, name: str = None) -> Any:
        """Register the given object, usable as either a decorator or a direct call.

        See the class docstring for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                self._do_register(_derive_name(func_or_class, name, self._name), func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._do_register(_derive_name(obj, name, self._name), obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            available = ", ".join(sorted(self._obj_map.keys())) or "none"
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry. Available options: {available}")
        return ret

    def keys(self):
        return self._obj_map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        if not self._obj_map:
            return f"Registry of {self._name}: (empty)"
        max_key_length = max(len(key) for key in self._obj_map.keys())
        return f"Registry of {self._name}:\n" + "\n".join(
            [f"{key.ljust(max_key_length)} : {value}" for key, value in self._obj_map.items()]
        )

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


class TermRegistry(Registry):
    """
    Registry for manager terms.

    Parameters
    ----------
    name: str
        The name of the registry.
    term_class: Type[ManagerTermBase]
        The class of the manager term.
    term_fn_wrapper_class: Type[ManagerTermFuncWrapperBase] | None = None
        The class of the manager term function wrapper.
    """

    def __init__(
        self,
        name: str,
        term_class: Type[ManagerTermBase],
        term_fn_wrapper_class: Type[ManagerTermFuncWrapperBase] | None = None,
    ) -> None:
        super().__init__(name)
        self._term_class = term_class
        self._term_fn_wrapper_class = term_fn_wrapper_class

    def _register_term(self, obj: Any, key: str, term_defaults: dict[str, Any]) -> None:
        if inspect.isclass(obj) and issubclass(obj, self._term_class):
            for field, value in term_defaults.items():
                setattr(obj, field, value)
            self._do_register(key, obj)
        elif isinstance(obj, FunctionType):
            if self._term_fn_wrapper_class is None:
                raise ValueError("function wrapping is not supported for this registry")
            wrapper_cls = self._term_fn_wrapper_class
            if term_defaults:
                wrapper_cls = type(
                    f"{wrapper_cls.__name__}_{key}",
                    (wrapper_cls,),
                    dict(term_defaults),
                )
            self._do_register(key, partial(wrapper_cls, func=obj))
        elif (
            self._term_fn_wrapper_class is not None
            and inspect.isclass(obj)
            and issubclass(obj, self._term_fn_wrapper_class)
        ):
            for field, value in term_defaults.items():
                setattr(obj, field, value)
            self._do_register(key, obj)
        else:
            raise ValueError(f"Invalid object type: {type(obj)}")

    def register(self, obj: Any = None, *, name: str | None = None, **term_defaults: Any) -> Any:
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                key = _derive_name(func_or_class, name, self._name)
                self._register_term(func_or_class, key, term_defaults)
                return func_or_class

            return deco

        # used as a function call
        key = _derive_name(obj, name, self._name)
        self._register_term(obj, key, term_defaults)


class VariantRegistry(Registry):
    """Registry for asset families with multiple mesh-file variants.

    A single decorated class declares its source directory name and the list of
    available variant ids. The decorator stamps ``_base_name`` and
    ``_default_variant`` (the first id in ``variant_list``) onto the class,
    merges family metadata into ``cls.metadata``, and registers one
    ``partial(cls, variant=N)`` callable per variant under the key
    ``f"{base_name}_{N}"``. The decorated class itself is also returned so it
    stays importable for type-level use.

    Asset families (objaverse, robotwin, ...) follow this contract by declaring
    a base options class that defines ``_base_name``/``_default_variant`` as
    ``ClassVar`` and an ``__init__`` (or ``model_post_init``) that materializes
    ``file`` from those + the requested variant.

    Passing ``short_name`` additionally registers the bare class under that key,
    which lets role-based code resolve a family without knowing about a specific
    variant (e.g. RoboTwin task specs reference ``bottle`` -> ``RobotwinBottle``).

    Calling ``register()`` with neither ``base_name`` nor ``variant_list`` falls
    through to :class:`Registry`'s plain-name registration, so procedural
    primitives (boxes, spheres) can share the same registry as variant assets.
    """

    def register(
        self,
        obj: Any = None,
        *,
        name: str | None = None,
        base_name: str = "",
        variant_list: Sequence[int] | None = None,
        short_name: str | None = None,
    ) -> Any:
        if base_name == "" and variant_list is None:
            return super().register(obj, name=name)

        if not variant_list:
            raise ValueError(f"variant_list is required when base_name is set (base_name={base_name!r})")

        def _register(options):
            options._base_name = base_name
            options._default_variant = variant_list[0]
            merged_metadata = dict(getattr(options, "metadata", None) or {})
            merged_metadata.update(
                {
                    "base_name": base_name,
                    "variant_list": list(variant_list),
                    "num_variants": len(variant_list),
                    "variants_name": [f"{base_name}_{variant}" for variant in variant_list],
                }
            )
            options.metadata = merged_metadata
            for variant in variant_list:
                self._do_register(f"{base_name}_{variant}", partial(options, variant=variant))
            if short_name is not None:
                self._do_register(short_name, options)
            return options

        if obj is None:
            return _register
        return _register(obj)
