"""ConfigurableOptions: base class for all Eden options.

:class:`ConfigurableOptions` extends the Genesis ``Options`` Pydantic model with Eden
conventions: ``lock()`` to freeze a config against further mutation, a best-effort
``__deepcopy__`` that tolerates unpicklable objects (e.g. USD stages), and
``extra="allow"`` so subclasses like ``SceneOptions`` / ``ManagerOptions`` can carry
named children (entities/terms) as dynamic fields. Root config classes in
:mod:`eden.utils.configs` override this with ``extra="forbid"``.
"""

import copy
from functools import cached_property
from typing import TypeVar

from genesis.options.options import Options
from pydantic import BaseModel, PrivateAttr

T = TypeVar("T", bound=Options)


class ConfigurableOptions(Options):
    """Base class for configurable options.

    Use with ConfigurableMixin.
    """

    # NOTE: allow extra fields
    model_config = {"extra": "allow"}

    _locked: bool = PrivateAttr(default=False)

    def __deepcopy__(self, memo):
        """Deep-copy the options, falling back to shared references for unpicklable objects.

        Pydantic deep-copies defaults during validation. Some pxr objects
        (e.g., Usd.Stage inside USD morphs) are not picklable. Perform a best-effort
        deepcopy and fall back to shared references when copying fails.
        """

        def safe_deepcopy(value):
            try:
                return copy.deepcopy(value, memo)
            except Exception:
                return value

        # Copy declared fields
        field_data = {name: safe_deepcopy(getattr(self, name)) for name in self.__class__.model_fields.keys()}

        new_obj = self.__class__.model_construct(**field_data)

        # Copy extras and private attrs with the same safety guard
        if self.__pydantic_extra__:
            new_obj.__pydantic_extra__ = {key: safe_deepcopy(value) for key, value in self.__pydantic_extra__.items()}
        if self.__pydantic_private__:
            new_obj.__pydantic_private__ = {
                key: safe_deepcopy(value) for key, value in self.__pydantic_private__.items()
            }
        new_obj.__pydantic_fields_set__ = set(self.__pydantic_fields_set__)

        return new_obj

    def lock(self):
        """Lock the config so that no further modifications (setattr recursively) are allowed."""
        if self._locked:
            return

        self._locked = True

        def lock_recursive(obj):
            if isinstance(obj, ConfigurableOptions):
                obj.lock()
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    lock_recursive(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    lock_recursive(item)

        # Lock fields in __dict__
        for value in self.__dict__.values():
            lock_recursive(value)

        # Lock fields in __pydantic_extra__
        if self.__pydantic_extra__:
            for value in self.__pydantic_extra__.values():
                lock_recursive(value)

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name != "_locked":
            raise RuntimeError(f"Cannot modify attribute '{name}' of {self.__class__.__name__} because it is locked.")
        super().__setattr__(name, value)

    def __init__(self, **data):
        # Only inject metadata into extras when the model allows extra fields.
        # Config classes (EdenConfig, EdenRLConfig) use extra="forbid"
        # and handle their type metadata via the serialization envelope instead.
        if self.model_config.get("extra") != "forbid":
            data["_option_module_"] = self.__module__
            data["_option_class_"] = self.__class__.__name__
        # NOTE: all of the option modifications/validations should be done in the __init__ method
        BaseModel.__init__(self, **data)

    def __getattribute__(self, item: str):
        # Bypass __getattribute__ to directly access the internal dictionary
        try:
            extra = object.__getattribute__(self, "__pydantic_extra__")
            if extra is not None and item in extra:
                return extra[item]
        except AttributeError:
            pass  # Handle case where __pydantic_extra__ is not yet initialized
        return super().__getattribute__(item)

    @cached_property
    def keys(self):
        """Mimic dict.keys() — return declared field names + dynamic extras, without triggering value serialization."""
        excluded = ("_option_module_", "_option_class_")

        def _names():
            for key in type(self).model_fields:
                if key not in excluded:
                    yield key
            extras = self.__pydantic_extra__
            if extras:
                for key in extras:
                    if key not in excluded:
                        yield key

        return lambda: tuple(_names())

    def dict(self, include_extra: bool = False):
        def clean_field(input_dict):
            return {
                key: clean_field(value) if isinstance(value, dict) else value
                for key, value in input_dict.items()
                if key not in ["_option_module_", "_option_class_"]
            }

        dumped = self.model_dump(serialize_as_any=True)
        if include_extra:
            return dumped
        return clean_field(dumped)

    def to_dict(self):
        """Interface for rsl_rl."""
        return self.dict()
