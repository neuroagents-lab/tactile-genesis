"""Eden config hierarchy (EdenConfig/EdenRLConfig) and serialization.

- :class:`EdenConfig` — base config for all environments (``EnvBase``).
- :class:`EdenRLConfig` — adds reward, termination, command, curriculum, and runner options.

Root config classes use Pydantic ``extra="forbid"`` (so misspelled fields raise), while
inner ``ManagerOptions`` / ``SceneOptions`` keep ``extra="allow"`` to hold dynamic
term/entity children. Saved configs use a ``_meta``/``config`` envelope; nested Options
recover their type from ``_option_module_`` / ``_option_class_`` keys.
"""

import json
import os
from enum import Enum
from types import UnionType
from typing import Union, get_args, get_origin

import numpy as np
import torch
import yaml
from genesis.options.renderers import RendererOptions
from pydantic import BaseModel

import eden as en
from eden.options import (
    ActionManagerOptions,
    CamerasOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RecorderManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SensorsOptions,
    TerminationManagerOptions,
)
from eden.options.options import ConfigurableOptions, Options
from eden.utils.misc import get_editable_package_commit


def _is_namedtuple_class(cls: type) -> bool:
    return bool(isinstance(cls, type) and cls is not tuple and issubclass(cls, tuple) and hasattr(cls, "_fields"))


class EdenConfig(ConfigurableOptions):
    """
    Base configuration for all Eden environments (EnvBase).

    Parameters
    ----------
    env_options: EnvOptions
        Environment options (sim dt, num_envs, etc.).
    scene_options: SceneOptions
        Scene options (entities, attachments).
    observation_options: ObservationManagerOptions
        Observation manager options.
    action_options: ActionManagerOptions
        Action manager options.
    event_options: EventManagerOptions
        Event manager options.
    cameras_options: CamerasOptions
        Cameras options.
    sensors_options: SensorsOptions
        Sensors options.
    metric_options: MetricManagerOptions
        Metric manager options.
    recorder_options: RecorderManagerOptions
        Recorder manager options.
    renderer_options: RendererOptions
        Renderer options.
    language_instructions: list[str]
        Language instructions for the task.
    """

    model_config = {"extra": "forbid"}

    env_options: EnvOptions = EnvOptions()
    scene_options: SceneOptions = SceneOptions()
    observation_options: ObservationManagerOptions = ObservationManagerOptions()
    action_options: ActionManagerOptions | None = ActionManagerOptions()
    event_options: EventManagerOptions = EventManagerOptions()
    cameras_options: CamerasOptions | None = CamerasOptions()
    sensors_options: SensorsOptions | None = SensorsOptions()
    metric_options: MetricManagerOptions | None = MetricManagerOptions()
    recorder_options: RecorderManagerOptions | None = RecorderManagerOptions()
    renderer_options: RendererOptions | None = RendererOptions()

    language_instructions: list[str] = []

    def _serialize(self) -> dict:
        """
        Serialize the config with metadata envelope.

        The ``_meta`` key is placed under a dedicated namespace to avoid collisions with config field names.
        """
        config_data = {}
        for key in type(self).model_fields:
            config_data[key] = serialize_obj_with_metadata(getattr(self, key))
        return {
            "_meta": {
                "eden_version": get_editable_package_commit(package_name="eden"),
                "genesis_version": get_editable_package_commit(package_name="genesis-world"),
                "_option_module_": self.__module__,
                "_option_class_": self.__class__.__name__,
            },
            "config": config_data,
        }

    def save_as_file(self, path: str, lock_config: bool = True):
        """
        Save the config as either JSON or YAML file with metadata envelope.

        Parameters
        ----------
        path: str
            The path to save the config.
        lock_config: bool
            Whether to lock the config after saving to prevent further modifications.
        """
        file_type = path.split(".")[-1].lower()
        if file_type not in ["json", "yaml", "yml"]:
            raise ValueError(f"Unsupported config file extension: {file_type}. Use .json, .yaml, or .yml")

        result = self._serialize()

        if lock_config:
            self.lock()

        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w") as f:
            if file_type == "json":
                json.dump(result, f, indent=4)
            else:
                yaml.safe_dump(result, f, sort_keys=False)

    def save_as_json(self, path: str | None = None, lock_config: bool = True):
        """
        Save the config as a JSON file with metadata envelope.

        Parameters
        ----------
        path: str | None
            The path to save the config. If not provided, the config will be saved to the log directory.
        lock_config: bool
            Whether to lock the config after saving to prevent further modifications.
        """
        path = path or f"{en.log_dir}/config.json"
        assert path.endswith(".json")
        self.save_as_file(path, lock_config)

    def save_as_yaml(self, path: str | None = None, lock_config: bool = True):
        """
        Save the config as a YAML file with metadata envelope.

        Parameters
        ----------
        path: str | None
            The path to save the config. If not provided, the config will be saved to the log directory.
        lock_config: bool
            Whether to lock the config after saving to prevent further modifications.
        """
        path = path or f"{en.log_dir}/config.yaml"
        assert path.endswith((".yaml", ".yml"))
        self.save_as_file(path, lock_config)

    @classmethod
    def load_from_file(cls, path: str) -> "EdenConfig":
        """
        Load the config from a JSON or YAML file.

        Parameters
        ----------
        path: str
            The path to load the config from.
        """
        path_lower = path.lower()
        if path_lower.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
        elif path_lower.endswith((".yaml", ".yml")):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {path}. Use .json, .yaml, or .yml")

        # Handle metadata envelope (new format)
        meta = data.pop("_meta", None)
        if meta is not None and "config" in data:
            # New envelope format: {"_meta": {...}, "config": {...}}
            data = data["config"]
        else:
            # Legacy flat format: strip old metadata keys
            data.pop("_option_module_", None)
            data.pop("_option_class_", None)
            data.pop("eden_version", None)
            data.pop("genesis_version", None)

        options = to_options(data, cls())

        # Version mismatch warnings
        if meta:
            saved_eden = meta.get("eden_version", "unknown")
            saved_genesis = meta.get("genesis_version", "unknown")
        else:
            saved_eden = "unknown"
            saved_genesis = "unknown"

        current_eden = get_editable_package_commit(package_name="eden")
        current_genesis = get_editable_package_commit(package_name="genesis-world")
        if saved_eden != "unknown" and saved_eden != current_eden:
            en.logger.warning(f"Eden version mismatch: {saved_eden} != {current_eden}")
        if saved_genesis != "unknown" and saved_genesis != current_genesis:
            en.logger.warning(f"Genesis version mismatch: {saved_genesis} != {current_genesis}")

        return options

    def with_overrides_from_file(self, path: str) -> "EdenConfig":
        """Create a new config by merging this config with values from a partial config file.

        The current config is not modified.

        Parameters
        ----------
        path: str
            The path to the override file (.json, .yaml, or .yml).

        Returns
        -------
        EdenConfig
            A new config instance based on this config with overrides applied.
        """
        path_lower = path.lower()
        if path_lower.endswith(".json"):
            with open(path, "r") as f:
                override_data = json.load(f)
        elif path_lower.endswith((".yaml", ".yml")):
            with open(path, "r") as f:
                override_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {path}. Use .json, .yaml, or .yml")

        return self.with_overrides_from_dict(override_data)

    def with_overrides_from_dict(self, data_dict: dict) -> "EdenConfig":
        merged = _deep_merge(self._to_dict(), data_dict)
        return to_options(merged, self)

    def _to_dict(self) -> dict:
        """Export config to a dict with option metadata (same shape as saved JSON/YAML)."""
        result = {}
        for key in type(self).model_fields:
            result[key] = serialize_obj_with_metadata(getattr(self, key))
        return result


class EdenRLConfig(EdenConfig):
    """
    Configuration for RL environments (RLEnvBase).

    Extends EdenConfig with reward, termination, command, curriculum,
    and runner options needed for reinforcement learning.

    Parameters
    ----------
    reward_options: RewardManagerOptions
        Reward manager options.
    termination_options: TerminationManagerOptions
        Termination manager options.
    command_options: CommandManagerOptions
        Command manager options.
    curriculum_options: CurriculumManagerOptions
        Curriculum manager options.
    runner_options: ConfigurableOptions | None
        Runner/training algorithm options. Framework-agnostic base type;
        use RslRlOnPolicyRunnerOptions, RlGamesPpoRunnerOptions, etc.
    """

    reward_options: RewardManagerOptions | None = RewardManagerOptions()
    termination_options: TerminationManagerOptions | None = TerminationManagerOptions()
    command_options: CommandManagerOptions | None = CommandManagerOptions()
    curriculum_options: CurriculumManagerOptions | None = CurriculumManagerOptions()
    runner_options: ConfigurableOptions | None = None


# ------------------------------------------------------------
# --------------------- Helper Functions ---------------------
# ------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Joins dicts but not lists."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def serialize_obj_with_metadata(obj):
    """Recursively serialize an object, capturing Options type metadata.

    Captures ``_option_module_`` and ``_option_class_`` metadata from Options
    instances. This works by using ``model_dump()`` for data but accessing actual
    Option objects to extract their metadata.

    NamedTuple values (e.g. Genesis ``TemperatureProperties``) subclass ``tuple`` and are
    serialized as plain lists of fields in field order; ``to_options`` reconstructs them
    using the surrounding Pydantic field annotations on load.
    """
    if isinstance(obj, Options):
        # Get the serialized data using model_dump
        result: dict = obj.model_dump()

        # Add metadata from this Options object
        result["_option_module_"] = getattr(obj, "_option_module_", obj.__module__)
        result["_option_class_"] = getattr(obj, "_option_class_", obj.__class__.__name__)

        # Now recursively process each field value to capture nested Options metadata
        for field_name in list(result.keys()):
            if field_name not in ["_option_module_", "_option_class_"]:
                # Try to get the actual object using getattr to handle all cases
                # (including fields not in __dict__ due to defaults, properties, etc)
                try:
                    actual_value = getattr(obj, field_name)
                    # If the actual value is an Options object, recursively serialize it
                    if isinstance(actual_value, Options):
                        result[field_name] = serialize_obj_with_metadata(actual_value)
                    else:
                        # Recurse over the live field value so container fields
                        # (for example list[Union[...Options]]) keep subclass data
                        # that may be erased by model_dump() schema narrowing.
                        result[field_name] = serialize_obj_with_metadata(actual_value)
                except AttributeError:
                    # Field doesn't exist, keep the dumped value and recurse
                    result[field_name] = serialize_obj_with_metadata(result[field_name])

        return result
    elif isinstance(obj, BaseModel):
        # For other BaseModel instances, serialize and recurse
        result = obj.model_dump()
        for key, value in list(result.items()):
            result[key] = serialize_obj_with_metadata(value)
        return result
    elif isinstance(obj, dict):
        return {key: serialize_obj_with_metadata(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_obj_with_metadata(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    # in case of np/torch scalars, cast to plain numeric types
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    return obj


def get_options_class_from_qualname(qualname: str):
    """
    Dynamically import and retrieve an Options class from its fully qualified name.

    Parameters
    ----------
    qualname : str
        The fully qualified name of the Options class (e.g., "eden.options.managers.actions.ActionTermOptions")

    Returns
    -------
    Type[Options]
        The Options class

    Raises
    ------
    ImportError
        If the module cannot be imported
    AttributeError
        If the class cannot be found in the module
    """
    import importlib

    # Split the qualname into module path and class name
    parts = qualname.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid qualified name: {qualname}. Expected format: 'module.path.ClassName'")

    module_path, class_name = parts

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")

    # Get the class from the module
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")


def to_options(config: dict, base_options: Options):
    """
    Reconstruct the original Options instance from serialized data.

    For classes with ``extra="forbid"`` (config root classes), only known model
    fields are passed to ``model_construct`` — unknown keys are silently
    dropped.  This prevents stale or legacy keys from leaking through as
    instance attributes when bypassing ``__init__``.

    For classes with ``extra="allow"`` (ManagerOptions, SceneOptions, etc.),
    all keys are passed through, preserving dynamic term/entity fields.
    """

    def is_dynamic_config(value):
        """Check if value is a dict with _option_module_ metadata."""
        if isinstance(value, dict):
            return "_option_module_" in value and "_option_class_" in value
        return False

    def is_options_class(value) -> bool:
        return isinstance(value, type) and issubclass(value, Options)

    def option_class_from_annotation(annotation, value=None):
        if annotation is None:
            return None
        if is_options_class(annotation):
            return annotation

        origin = get_origin(annotation)
        if origin in (Union, UnionType):
            option_classes = []
            for arg in get_args(annotation):
                option_cls = option_class_from_annotation(arg)
                if option_cls is not None and option_cls not in option_classes:
                    option_classes.append(option_cls)
            if len(option_classes) == 1:
                return option_classes[0]
            # Multiple candidates (e.g. `SurfaceOptions | Surface`): a dict with class
            # metadata is handled before we get here, so this is a legacy metadata-less
            # dict. Disambiguate by field-key overlap — the class whose declared fields
            # best cover the dict's keys.
            if len(option_classes) > 1 and isinstance(value, dict):
                keys = set(value) - {"_option_module_", "_option_class_"}
                return max(option_classes, key=lambda cls: len(keys & set(getattr(cls, "model_fields", {}))))
        return None

    def item_annotation(annotation):
        origin = get_origin(annotation)
        if origin in (list, tuple):
            args = get_args(annotation)
            if args:
                return args[0]
        return None

    def value_annotation(annotation):
        origin = get_origin(annotation)
        if origin is dict:
            args = get_args(annotation)
            if len(args) == 2:
                return args[1]
        return None

    def key_annotation(annotation):
        origin = get_origin(annotation)
        if origin is dict:
            args = get_args(annotation)
            if len(args) == 2:
                return args[0]
        return None

    def coerce_dict_key(key, key_type):
        if key_type is int and isinstance(key, str):
            try:
                return int(key)
            except ValueError:
                return key
        return key

    def construct_options(options_cls: type[Options], value: dict):
        option_dict = {}
        for key, item in value.items():
            field_info = options_cls.model_fields.get(key)
            field_annotation = field_info.annotation if field_info is not None else None
            option_dict[key] = convert_value(item, field_annotation)
        return _safe_model_construct(options_cls, option_dict)

    def convert_value(value, target_annotation=None):
        """Recursively convert values, handling dynamic configs and nested structures."""
        # Handle dynamic config instances
        if isinstance(value, dict) and is_dynamic_config(value):
            option_module = value.get("_option_module_", None)
            option_class = value.get("_option_class_", None)
            ResolvedClass = get_options_class_from_qualname(f"{option_module}.{option_class}")
            return construct_options(ResolvedClass, value)

        target_options_cls = option_class_from_annotation(target_annotation, value)
        if isinstance(value, dict) and target_options_cls is not None:
            return construct_options(target_options_cls, value)

        # Handle dicts (may contain dynamic configs)
        elif isinstance(value, dict):
            nested_annotation = value_annotation(target_annotation)
            kt = key_annotation(target_annotation)
            return {coerce_dict_key(k, kt): convert_value(v, nested_annotation) for k, v in value.items()}
        # Handle lists (may contain nested dynamic configs; NamedTuple rows load via target_annotation)
        elif isinstance(value, (list, tuple)):
            if target_annotation is not None and _is_namedtuple_class(target_annotation):
                seq = value
                fields = target_annotation._fields
                if len(seq) == len(fields):
                    return target_annotation(*(convert_value(v, None) for v in seq))
            nested_annotation = item_annotation(target_annotation)
            return [convert_value(item, nested_annotation) for item in value]
        # Return as-is for other types
        else:
            return value

    # Convert all config attributes
    res = {}
    for key, value in config.items():
        field_info = type(base_options).model_fields.get(key)
        field_annotation = field_info.annotation if field_info is not None else None
        res[key] = convert_value(value, field_annotation)

    return _safe_model_construct(type(base_options), res)


def _safe_model_construct(cls: type, data: dict):
    """Call model_construct, filtering out unknown keys for extra='forbid' classes."""
    if cls.model_config.get("extra") == "forbid":
        known_fields = set(cls.model_fields.keys())
        internal_keys = {"_option_module_", "_option_class_"}
        dropped = {k for k in data if k not in known_fields and k not in internal_keys}
        if dropped:
            en.logger.warning(f"Dropping unknown keys when loading {cls.__name__}: {dropped}")
        data = {k: v for k, v in data.items() if k in known_fields}
    return cls.model_construct(**data)


def load_json(path: str):
    import json

    with open(path, "r") as f:
        data = json.load(f)
    return data
