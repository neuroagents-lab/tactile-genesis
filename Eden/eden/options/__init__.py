"""update genesis options."""

from genesis.options.options import Options
from pydantic import BaseModel

from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import (
    GHOST_DEFAULTS,
    BoxOptions,
    CylinderOptions,
    EntityOptions,
    GroupedEntityOptions,
    PlaneOptions,
    PrimitiveOptions,
    SceneOptions,
    SphereOptions,
    TerrainOptions,
)
from eden.options.envs import EnvOptions
from eden.options.managers.base import ManagerOptions
from eden.options.managers.actions import ActionManagerOptions, ActionTermOptions
from eden.options.managers.commands import CommandManagerOptions, CommandTermOptions
from eden.options.managers.curricula import (
    CurriculumManagerOptions,
    CurriculumTermOptions,
)
from eden.options.managers.events import EventManagerOptions, EventTermOptions
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions
from eden.options.managers.observations import (
    NoiseOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
    ObservationTermOptions,
)
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.options.managers.terminations import (
    TerminationManagerOptions,
    TerminationTermOptions,
)
from eden.options.managers.recorders import RecorderManagerOptions, RecorderTermOptions
from eden.options.renderer import RayTracerOptions
from eden.options.sensors import SensorOptions, SensorsOptions

# NOTE: monkeypatch the Options class
__original_options_init = Options.__init__
__original_options_getattribute = Options.__getattribute__
__original_options_model_construct = Options.model_construct


def options_init(self, **data):
    __original_options_init(self, **data)
    # Skip metadata injection for classes that forbid extras (config root classes).
    if self.model_config.get("extra") != "forbid":
        BaseModel.__setattr__(self, "_option_module_", self.__module__)
        BaseModel.__setattr__(self, "_option_class_", self.__class__.__name__)


def __getattribute__(self, item: str):
    # Bypass __getattribute__ to directly access the internal dictionary
    try:
        extra = object.__getattribute__(self, "__pydantic_extra__")
        if extra is not None and item in extra:
            return extra[item]
    except AttributeError:
        pass  # Handle case where __pydantic_extra__ is not yet initialized
    return __original_options_getattribute(self, item)


@classmethod
def options_model_construct(cls, _fields_set=None, **values):
    # Call the original model_construct with the correct class
    # We need to use __func__ to get the underlying function and pass cls explicitly
    m = __original_options_model_construct.__func__(cls, _fields_set=_fields_set, **values)
    # Set the custom attributes (same as in __init__), but skip for extra="forbid" classes
    if cls.model_config.get("extra") != "forbid":
        BaseModel.__setattr__(m, "_option_module_", m.__module__)
        BaseModel.__setattr__(m, "_option_class_", m.__class__.__name__)
    return m


Options.__init__ = options_init
Options.__getattribute__ = __getattribute__
Options.model_construct = options_model_construct
Options.model_config = {"extra": "allow"}


__all__ = [
    "ManagerOptions",
    "ActionManagerOptions",
    "ActionTermOptions",
    "CommandManagerOptions",
    "CommandTermOptions",
    "CurriculumManagerOptions",
    "CurriculumTermOptions",
    "EnvOptions",
    "CameraOptions",
    "CamerasOptions",
    "SensorOptions",
    "SensorsOptions",
    "EntityOptions",
    "GroupedEntityOptions",
    "PlaneOptions",
    "BoxOptions",
    "SphereOptions",
    "CylinderOptions",
    "PrimitiveOptions",
    "TerrainOptions",
    "GHOST_DEFAULTS",
    "SceneOptions",
    "EventManagerOptions",
    "EventTermOptions",
    "ObservationManagerOptions",
    "ObservationTermOptions",
    "ObservationGroupOptions",
    "NoiseOptions",
    "RewardManagerOptions",
    "RewardTermOptions",
    "TerminationManagerOptions",
    "TerminationTermOptions",
    "MetricManagerOptions",
    "MetricTermOptions",
    "RecorderManagerOptions",
    "RecorderTermOptions",
    "RayTracerOptions",
]
