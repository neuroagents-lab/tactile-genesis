"""ConfigurableMixin: bind options classes to their runtime objects."""

from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING, Generic, TypeVar, get_args
from types import FunctionType
from functools import cached_property
import copy
import inspect

import genesis as gs

from eden.utils.misc import to_snake_case

if TYPE_CHECKING:
    from eden.options.options import ConfigurableOptions

T = TypeVar("T", bound="ConfigurableOptions")


class ConfigurableMixin(Generic[T]):
    _options_class_: type[T] | None = None

    #: Names of options fields that ``configure()`` accepts and stores in the options, but which are NOT
    #: mirrored as runtime instance attributes (so the runtime object can't read them). Default empty —
    #: a pure no-op for every configurable except those that opt in. Used by observation terms to keep the
    #: post-compute modifier fields (noise/clip/scale/history) off the term object while still configurable.
    _extra_option_params: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_options_class_" not in cls.__dict__:
            for base in getattr(cls, "__orig_bases__", ()):
                origin = getattr(base, "__origin__", None)
                if origin is not None and issubclass(origin, ConfigurableMixin):
                    args = get_args(base)
                    if args and not isinstance(args[0], TypeVar):
                        cls._options_class_ = args[0]
                        break

    def __init__(self, options: T) -> None:
        self._uid = gs.UID()
        self._options: T = options

        for name in self.get_parameter_names():
            if name in self._options.model_dump():
                setattr(self, name, getattr(self._options, name))
            else:
                setattr(self, name, copy.deepcopy(getattr(self, name)))

        # check if there are unused parameters in the options. ``_extra_option_params`` are legitimate
        # options-only fields (not mirrored as attrs), so they are expected to be "unused" here.
        unused_params = set(self._options.keys()) - set(self.get_parameter_names())
        invalid_params = (
            unused_params - {"name", "class_name", "_option_module_", "_option_class_"} - self._extra_option_params
        )
        if invalid_params:
            raise ValueError(f"The following parameters are invalid: {invalid_params}")

    @classmethod
    def get_parameter_names(cls) -> list[str]:
        params = set()
        # Only collect attributes from classes in the ConfigurableMixin
        # hierarchy, skipping unrelated bases like nn.Module.
        for base_cls in cls.__mro__:
            if base_cls is ConfigurableMixin or not issubclass(base_cls, ConfigurableMixin):
                continue
            for name, value in base_cls.__dict__.items():
                if name.startswith("__") or name.startswith("_"):
                    continue
                if isinstance(value, (property, cached_property, classmethod, staticmethod)):
                    continue
                if isinstance(value, FunctionType):
                    continue
                params.add(name)
        return list(params)

    @classmethod
    def _build_config_dict(cls, **kwargs) -> dict:
        if "name" in kwargs:
            raise ValueError("The 'name' parameter is reserved and cannot be used.")
        invalid_params = set(kwargs.keys()) - set(cls.get_parameter_names()) - cls._extra_option_params
        if invalid_params:
            raise ValueError(
                f"The following parameters are invalid: {invalid_params}, allowed parameters: {cls.get_parameter_names()}"
            )
        config = {
            "name": to_snake_case(cls.__name__),
        }

        for name in cls.get_parameter_names():
            if name in kwargs:
                config[name] = kwargs[name]
            else:
                config[name] = copy.deepcopy(getattr(cls, name))
        # Options-only params: passed through to the options when provided, never read from a class attr.
        for name in cls._extra_option_params:
            if name in kwargs:
                config[name] = kwargs[name]
        return config

    @classmethod
    def configure(cls, **kwargs) -> T:
        """
        Configure the class with the given parameters.

        Parameters
        ----------
        **kwargs: dict
            The parameters to configure the class with.

        Returns
        -------
        T
            The options instance for the class.
        """
        config = cls._build_config_dict(**kwargs)
        if cls._options_class_ is None:
            raise TypeError(
                f"{cls.__name__} must parameterize ConfigurableMixin[T] with a concrete Options type "
                f"or set _options_class_ explicitly."
            )
        return cls._options_class_(**config)

    @property
    def name(self) -> str:
        return to_snake_case(self.__class__.__name__)

    @property
    def uid(self):
        return self._uid

    @property
    def options(self) -> T:
        return self._options


class ConfigurableFuncWrapperMixin(ConfigurableMixin[T]):
    @staticmethod
    def get_function_parameter_names(func: FunctionType) -> list[str]:
        return list(inspect.signature(func).parameters.keys())

    @classmethod
    def _build_config_dict(cls, func: FunctionType = None, **kwargs) -> dict:
        if "name" in kwargs:
            raise ValueError("The 'name' parameter is reserved and cannot be used.")
        config = {
            "name": to_snake_case(func.__name__) if func is not None else to_snake_case(cls.__name__),
        }

        for name in cls.get_parameter_names():
            if name in kwargs:
                if name == "params":
                    assert isinstance(kwargs[name], dict), "params parameter must be a dictionary"
                    invalid_params = set(list(kwargs[name].keys())) - set(cls.get_function_parameter_names(func))
                    if invalid_params:
                        raise ValueError(f"The following parameters are invalid: {invalid_params} for {cls}")
                    # NOTE: update provided params for the missing ones
                    params = {**cls.params}
                    params.update(kwargs[name])
                    kwargs[name] = params
                config[name] = kwargs[name]
            else:
                config[name] = copy.deepcopy(getattr(cls, name))
        # Options-only params (e.g. obs post-compute modifiers): passed through to the options when provided.
        for name in cls._extra_option_params:
            if name in kwargs:
                config[name] = kwargs[name]
        return config

    @classmethod
    def configure(cls, func: FunctionType, **kwargs) -> T:
        """
        Configure the class with the given parameters.

        Parameters
        ----------
        func: FunctionType
            The function to configure the class with.
        **kwargs: dict
            The parameters to configure the class with.

        Returns
        -------
        T
            The options instance for the class.
        """
        config = cls._build_config_dict(func=func, **kwargs)
        if cls._options_class_ is None:
            raise TypeError(
                f"{cls.__name__} must parameterize ConfigurableMixin[T] with a concrete Options type "
                f"or set _options_class_ explicitly."
            )
        return cls._options_class_(**config)

    @property
    def name(self) -> str:
        return to_snake_case(self._func.__name__)
