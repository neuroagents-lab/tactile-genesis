import copy
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import eden as en
import eden.options.learning.rsl_rl as rl
import genesis as gs
import numpy as np
from eden.options import SensorOptions
from eden.options.entities import GroupedEntityOptions, RobotOptions
from eden.tasks.registry import TaskMod
from eden.utils.configs import EdenRLConfig

import entities.robots  # noqa: F401  -- populates ROBOT_REGISTRY for RobotLiteral below
from entities.objects import OBJECTS_4CM_16, DexCube
from model_config import GROUP_ENCODER_CFGS, TACTILE_ENCODER_CFGS
from models import (
    RslRlActorPreEncodeMLPOptions,
    RslRlActorPreEncodeRNNOptions,
    RslRlActorTactilePreEncodeMLPOptions,
    RslRlActorTactilePreEncodeRNNOptions,
    RslRlPreEncodeMLPOptions,
    RslRlPreEncodeRNNOptions,
    RslRlTactilePreEncodeMLPOptions,
    RslRlTactilePreEncodeRNNOptions,
)
from registry import ROBOT_REGISTRY
from shared_terms import DECIMATION, UpdateCurriculumWeights, frozen_link_names
from tactile_sensors import TACTILE_SENSORS, spec_for_sensor_name

# Build the CLI Literal choices from whatever is currently registered in
# `ROBOT_REGISTRY` rather than hard-coding a list, so adding a new
# `@ROBOT_REGISTRY.register(...)` hand under `entities/robots/` is the only
# thing needed to expose it through `--robot`.
RobotLiteral = Literal[tuple(sorted(ROBOT_REGISTRY.keys()))]


def _log_info(message: str) -> None:
    logger = getattr(en, "logger", None)
    if logger is not None:
        logger.info(message)
    else:
        print(message)


_ROBOT_DOF_PARAM_TO_ATTR = {
    "stiffness": "default_dofs_stiffness",
    "kp": "default_dofs_kp",
    "kd": "default_dofs_kd",
    "damping": "default_dofs_damping",
    "armature": "default_dofs_armature",
}

_ACTION_MODIFIER_PARAM_TO_FIELDS = {
    "deadband_epsilon": ("deadband", {"deadband_epsilon"}),
    "gear_backlash": ("gear_backlash", {"backlash"}),
    "gear_reversal_threshold": ("gear_backlash", {"reversal_threshold"}),
    "gear_takeup_rate": ("gear_backlash", {"takeup_rate"}),
    "gear_initial_side": ("gear_backlash", {"initial_side"}),
    "torque_kick": ("constant_torque_kick", {"torque_kick"}),
    "activation_epsilon": ("constant_torque_kick", {"activation_epsilon"}),
    "motor_strength": ("motor_strength", {"motor_strength"}),
}


def _as_float_list(values: Any) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


def _controller_dof_names(controller: Any) -> tuple[str, ...] | None:
    dofs_name = getattr(controller, "dofs_name", None)
    if not dofs_name or dofs_name == "*":
        return None
    if isinstance(dofs_name, str):
        return None if "*" in dofs_name else (dofs_name,)
    names = tuple(str(name) for name in dofs_name)
    return None if "*" in names else names


def _filter_param_values_to_controller_dofs(
    property_name: str,
    dof_names: Any,
    values: list[float],
    controller_dof_names: tuple[str, ...] | None,
) -> list[float] | None:
    source_dof_names = tuple(str(name) for name in dof_names)
    if source_dof_names and len(source_dof_names) != len(values):
        _log_info(f"RobotHandMod: skipping action modifier {property_name!r} with mismatched dof/value lengths")
        return None

    if controller_dof_names is None:
        return values
    if len(values) == 1:
        return values

    if source_dof_names:
        values_by_dof = dict(zip(source_dof_names, values, strict=True))
        missing_dofs = [name for name in controller_dof_names if name not in values_by_dof]
        if missing_dofs:
            _log_info(
                f"RobotHandMod: skipping action modifier {property_name!r}; "
                f"calibration is missing controller DOFs {missing_dofs!r}"
            )
            return None
        return [values_by_dof[name] for name in controller_dof_names]

    if len(values) == len(controller_dof_names):
        return values

    _log_info(
        f"RobotHandMod: skipping action modifier {property_name!r}; "
        f"value length {len(values)} does not match controller DOF count {len(controller_dof_names)}"
    )
    return None


def _apply_identified_action_modifier_params(
    config: EdenRLConfig,
    cal: Any,
    *,
    action_term_name: str = "dofs_pos_controller",
) -> None:
    controller = getattr(getattr(config, "action_options", None), action_term_name, None)
    modifier = getattr(controller, "modifier", None)
    if controller is None or modifier is None or not hasattr(modifier, "modifiers"):
        return

    controller_dof_names = _controller_dof_names(controller)
    updates_by_modifier: dict[str, dict[str, list[float] | float]] = {}
    for param in cal:
        field_map = _ACTION_MODIFIER_PARAM_TO_FIELDS.get(param.property)
        if field_map is None:
            continue
        modifier_name, field_names = field_map
        values = _as_float_list(param.value)
        if not values:
            continue
        values = _filter_param_values_to_controller_dofs(
            param.property,
            param.dof_names,
            values,
            controller_dof_names,
        )
        if values is None:
            continue
        modifier_updates = updates_by_modifier.setdefault(modifier_name, {})
        for field_name in field_names:
            modifier_updates[field_name] = values

    if not updates_by_modifier:
        return

    updated_modifiers = []
    modified = False
    for mod in modifier.modifiers:
        mod_updates = updates_by_modifier.get(mod.name)
        if mod_updates:
            updated_modifiers.append(mod.model_copy(update=mod_updates))
            modified = True
        else:
            updated_modifiers.append(mod)

    if modified:
        controller_updates = modifier.model_copy(update={"modifiers": updated_modifiers})
        setattr(config.action_options, action_term_name, controller.model_copy(update={"modifier": controller_updates}))


class RobotHandMod(TaskMod):
    def __init__(
        self,
        robot: Annotated[
            RobotLiteral,
            "The robot to use for the task.",
        ] = "xhand1",
        *,
        entity_name: str = "robot",
        action_term_name: str = "dofs_pos_controller",
        name: str = "",
    ) -> None:
        self.robot = robot
        self._entity_name = entity_name
        self._action_term_name = action_term_name
        if name:
            self.with_prefix(name)

    def _configure_robot_and_actions(self, config: EdenRLConfig, robot_instance: RobotOptions) -> RobotOptions:
        return robot_instance

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        from eden.extensions.sysid import ParameterSet

        robot_cls = ROBOT_REGISTRY.get(self.robot)
        robot_instance = robot_cls()
        cal_path = getattr(robot_instance.metadata, "calibration_params", None)
        cal = None

        if cal_path:
            path = Path(cal_path)
            if not path.is_file():
                _log_info(f"RobotHandMod: calibration_params file not found, skipping: {path}")
            else:
                try:
                    cal = ParameterSet.load_yaml(path)
                except Exception as exc:
                    _log_info(f"RobotHandMod: failed to load calibration YAML {path}: {exc}")
                    cal = None
                else:
                    updates: dict[str, Any] = {}
                    for p in cal:
                        attr = _ROBOT_DOF_PARAM_TO_ATTR.get(p.property)
                        if attr is None or not hasattr(robot_instance, attr):
                            continue
                        cur = getattr(robot_instance, attr)
                        if not isinstance(cur, dict):
                            continue
                        merged = dict(cur)
                        arr = _as_float_list(p.value)
                        for name, v in zip(p.dof_names, arr, strict=True):
                            merged[str(name)] = float(v)
                        updates[attr] = merged
                    if updates:
                        robot_instance = robot_instance.model_copy(update=updates)

        robot_instance = self._configure_robot_and_actions(config, robot_instance)
        if cal is not None:
            _apply_identified_action_modifier_params(config, cal, action_term_name=self._action_term_name)
        setattr(config.scene_options, self._entity_name, robot_instance)

        return config


def _track_link_sensor_args(
    track_link: str | tuple[int, ...] | tuple[str, ...],
) -> tuple[dict, list[str]]:
    """Split a track-link spec into (sensor kwargs, SensorOptions track_link_names).

    A string (``"obj"`` or ``"obj/handle"``) — or a tuple of such strings, to track
    several entities at once — is resolved by name at env build via
    ``SensorOptions.track_link_names``; a tuple of ints is passed straight through as
    the sensor's ``track_link_idx`` (legacy global-index path).

    For the name path the sensor still needs a valid (non-empty) ``track_link_idx``
    at construction time, so a zero placeholder (one per name) is supplied and
    overwritten once ``track_link_names`` resolves during env build.
    """
    if isinstance(track_link, str):
        return {"track_link_idx": (0,)}, [track_link]
    track_link = tuple(track_link)
    if track_link and all(isinstance(t, str) for t in track_link):
        return {"track_link_idx": tuple(0 for _ in track_link)}, list(track_link)
    return {"track_link_idx": track_link}, []


class RobotHandWithPrivSensorsMod(RobotHandMod):
    def __init__(
        self,
        robot: Annotated[
            RobotLiteral,
            "The robot to use for the task.",
        ] = "xhand1",
        *,
        entity_name: str = "robot",
        action_term_name: str = "dofs_pos_controller",
        priv_sensor_cfg_name: str = "fingertips",
        track_link_idx: str | tuple[int, ...] | tuple[str, ...] = "obj",
        name: str = "",
    ) -> None:
        super().__init__(robot=robot, entity_name=entity_name, action_term_name=action_term_name, name=name)
        self._priv_sensor_cfg_name = priv_sensor_cfg_name
        self._track_link_idx = track_link_idx

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)
        track_idx_kwargs, track_link_names = _track_link_sensor_args(self._track_link_idx)
        for link_name, offset in (
            getattr(config.scene_options, self._entity_name)
            .metadata.priv_sensor_cfgs[self._priv_sensor_cfg_name]
            .items()
        ):
            setattr(
                config.sensors_options,
                "priv_surface_distance_" + link_name,
                SensorOptions(
                    sensor=gs.sensors.SurfaceDistanceProbe(
                        probe_local_pos=(offset,),
                        probe_radius=0.5,
                        draw_debug=False,
                        **track_idx_kwargs,
                    ),
                    attach_entity_name=self._entity_name,
                    attach_link_name=link_name,
                    track_link_names=track_link_names,
                ),
            )
            setattr(
                config.sensors_options,
                "priv_contact_" + link_name,
                SensorOptions(
                    sensor=gs.sensors.ContactForce(
                        max_force=10.0,
                        draw_debug=False,
                    ),
                    attach_entity_name=self._entity_name,
                    attach_link_name=link_name,
                ),
            )
        return config


def _make_options_ghost(opts: Any) -> None:
    """Mark an entity-options object as a non-colliding pinned ghost.

    Touches only fields that exist on this options class (``fixed`` is on
    ``PrimitiveOptions`` but not on the URDF/Mesh ``EntityOptions`` or on
    ``GroupedEntityOptions``); pydantic ``extra="allow"`` would otherwise
    silently turn a typo into an inert extra attribute.
    """
    opts.collision = False
    opts.is_fixed_base = True
    if "fixed" in type(opts).model_fields:
        opts.fixed = True


class ManipulationObjectMod(TaskMod):
    """
    Add the object entity to manipulate.

    Parameters
    ----------
    entity_name : str
        The name of the entity to set.
    add_vis_entity : bool
        Whether to additionally add "vis_entity_name" with same morph as the object but fixed and transparent.
    """

    def __init__(
        self,
        obj: Annotated[
            Literal["cube", "primitives", "objaverse"],
            "Object: cube (default), primitives, or objaverse.",
        ] = "cube",
        *,
        primitives: list[Any] | None = None,
        entity_name: str = "obj",
        add_vis_entity: bool = False,
        name: str = "",
    ) -> None:
        self.obj = obj
        self._primitives = primitives or OBJECTS_4CM_16
        self._entity_name = entity_name
        self._vis_entity_name = "vis_" + entity_name if add_vis_entity else None
        if name:
            self.with_prefix(name)

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        if self.obj == "primitives":
            entity_options = GroupedEntityOptions(
                grouped_entities=self._primitives,
                surface=gs.surfaces.Default(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.3, 0.2, 1.0),
                    )
                ),
            )
        elif self.obj == "objaverse":
            raise NotImplementedError("Objaverse objects not implemented yet")
        else:
            entity_options = DexCube()
        setattr(config.scene_options, self._entity_name, entity_options)

        if self._vis_entity_name:
            vis_entity_options = copy.deepcopy(entity_options)
            _make_options_ghost(vis_entity_options)
            # For a GroupedEntityOptions, each variant morph is built from the
            # variant's own ``collision`` / ``fixed`` fields in
            # ``Entity._create_grouped_morphs_from_options``. Setting the flags
            # only on the parent leaves every variant fully collidable, which
            # is the bug that previously forced us to skip vis_obj for grouped
            # entities. Propagate the ghost settings down so each variant's
            # morph is built non-colliding and pinned.
            if isinstance(vis_entity_options, GroupedEntityOptions):
                for variant in vis_entity_options.grouped_entities:
                    _make_options_ghost(variant)
            setattr(config.scene_options, self._vis_entity_name, vis_entity_options)
        return config


class TactileSensorsMod(TaskMod):
    def __init__(
        self,
        sensors: Annotated[
            str,
            'Sensor placement/type[/noisy], e.g. "low-tips/bool" or "low-tips/bool/noisy". '
            "The trailing /noisy flag enables the sensor type's realistic noise model.",
        ] = "none",
        temporal_reduction: Annotated[
            Literal["median", "none", "last"],
            "How to reduce the within-step substep history (DECIMATION=5) of every tactile "
            "sensor before the obs hits the encoder. 'none' (default) keeps all substeps as "
            "extra features per probe; 'median'/'last' collapse the substep axis.",
        ] = "none",
        *,
        track_link_idx: str | tuple[int, ...] = "obj",
        name: str = "",
    ) -> None:
        self.sensors = sensors
        self.temporal_reduction = temporal_reduction
        self._track_link_idx = track_link_idx
        if name:
            self.with_prefix(name)

    @staticmethod
    def _parse_sensors_str(sensors: str) -> tuple[str, str, bool]:
        """Parse ``placement/sensor_type[/noisy]`` into its components.

        The optional trailing ``/noisy`` segment turns on the sensor type's
        realistic noise model (see ``TactileSensorSpec.noise_params``).
        """
        parts = sensors.split("/")
        if len(parts) > 3:
            raise ValueError(f"Invalid sensors spec {sensors!r}: expected at most placement/sensor_type/noisy.")
        placement_type = parts[0]
        sensor_type = parts[1] if len(parts) > 1 else ""
        noisy = False
        if len(parts) > 2:
            if parts[2] != "noisy":
                raise ValueError(f"Invalid sensors spec {sensors!r}: third segment must be 'noisy', got {parts[2]!r}.")
            noisy = True
        return placement_type, sensor_type, noisy

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        has_tactile = self.sensors not in ("", "none")
        if has_tactile:
            placement_type, sensor_type, noisy = self._parse_sensors_str(self.sensors)
            sensors_dict = self._get_sensors_dict(
                config.scene_options.robot,
                placement_type,
                sensor_type,
                noisy=noisy,
                frozen_dofs=self._frozen_dofs(config),
            )
            for key, value in sensors_dict.items():
                setattr(config.sensors_options, key, value)
            group = getattr(getattr(config, "observation_options", None), "tactile_sensors", None)
            term = getattr(group, "tactile_sensors", None) if group is not None else None
            if term is not None:
                term.temporal_reduction = self.temporal_reduction
        else:
            if hasattr(config.observation_options, "tactile_sensors"):
                del config.observation_options.tactile_sensors
        return config

    @staticmethod
    def _frozen_dofs(config: EdenRLConfig) -> tuple[str, ...]:
        """Frozen DOF names from a partial frozen hand controller, if any.

        Scans the action terms for a ``frozen_dofs`` field (set by
        ``PartialFrozenExplicitPDController``); returns ``()`` for a fully
        actuated controller.
        """
        action_options = getattr(config, "action_options", None)
        if action_options is None:
            return ()
        for _name, term in action_options:
            frozen = getattr(term, "frozen_dofs", None)
            if frozen:
                return tuple(str(dof) for dof in frozen)
        return ()

    def _get_sensors_dict(
        self,
        robot_cfg: RobotOptions,
        placement_type: str,
        sensors_type: str,
        noisy: bool = False,
        frozen_dofs: tuple[str, ...] = (),
    ) -> dict[str, SensorOptions]:
        sensors_dict = {}
        spec = TACTILE_SENSORS.get(sensors_type)
        if spec is None:
            raise ValueError(f"Invalid sensors type: {sensors_type}")

        sensor_kwargs = dict(spec.params)
        if noisy:
            if not spec.noise_params:
                _log_info(f"TactileSensorsMod: sensor type {sensors_type!r} has no noise model; '/noisy' is a no-op.")
            sensor_kwargs.update(spec.noise_params)
        sensor_kwargs["draw_debug"] = True
        # Sensor steps DECIMATION times per env step; postprocess takes the temporal
        # median across the in-step history so the obs is robust to per-substep spikes.
        sensor_kwargs["history_length"] = DECIMATION

        if placement_type in ["fingertip_links", "links"]:
            assert spec.placement == "link", f"Sensor type {sensors_type!r} is not a link-attached sensor."

            if placement_type == "fingertip_links":
                links = robot_cfg.metadata.fingertip_links
            else:
                links = robot_cfg.metadata.finger_links + [robot_cfg.metadata.palm_link]

            frozen_links = frozen_link_names(links, robot_cfg.dofs_name, frozen_dofs)
            if frozen_links:
                _log_info(f"TactileSensorsMod: skipping tactile sensors on frozen links {sorted(frozen_links)}")
            for link_name in links:
                if link_name in frozen_links:
                    continue
                sensors_dict[f"tactile_{sensors_type}_{link_name}"] = SensorOptions(
                    sensor=spec.sensor_cls(**sensor_kwargs),
                    attach_entity_name="robot",
                    attach_link_name=link_name,
                )
        else:
            assert spec.placement == "probes", f"Sensor type {sensors_type!r} is not a probe sensor."
            resolution, subset = self._split_placement_key(placement_type)
            probe_cfgs = robot_cfg.metadata.tactile_probe_cfgs
            if resolution not in probe_cfgs:
                raise ValueError(
                    f"Unknown probe resolution {resolution!r} for placement {placement_type!r}; "
                    f"available: {sorted(probe_cfgs)}"
                )
            probe_data = self._parse_probe_config(probe_cfgs[resolution])
            probe_data = self._filter_probe_data_by_subset(probe_data, robot_cfg.metadata, subset)
            track_idx_kwargs, track_link_names = _track_link_sensor_args(self._track_link_idx)
            # Only some gs sensor classes accept per-probe normals as a constructor field.
            sensor_takes_normal = "probe_local_normal" in spec.sensor_cls.model_fields
            frozen_links = frozen_link_names(list(probe_data.keys()), robot_cfg.dofs_name, frozen_dofs)
            if frozen_links:
                _log_info(f"TactileSensorsMod: skipping tactile sensors on frozen links {sorted(frozen_links)}")
            for link_name, probe in probe_data.items():
                if link_name in frozen_links:
                    continue
                kwargs = dict(
                    probe_local_pos=probe["local_pos"],
                    probe_radius=probe["radius"],
                )
                if sensor_takes_normal:
                    kwargs["probe_local_normal"] = probe["local_normal"]
                if spec.needs_track_link:  # Point cloud based sensors require a tracked link
                    kwargs.update(track_idx_kwargs)
                kwargs.update(sensor_kwargs)
                sensors_dict[f"tactile_{sensors_type}_{link_name}"] = SensorOptions(
                    sensor=spec.sensor_cls(
                        **kwargs,
                    ),
                    attach_entity_name="robot",
                    attach_link_name=link_name,
                    track_link_names=track_link_names if spec.needs_track_link else [],
                    # Carried for postprocess funcs (e.g. agg_force) since gs sensor
                    # classes like ContactDepthProbe do not expose probe normals.
                    tactile_probe_local_normal=probe["local_normal"],
                )
        return sensors_dict

    # Subset names recognised by ``_split_placement_key``.
    _PROBE_SUBSETS = ("hand", "tips", "palm", "midfinger", "fingers")

    @classmethod
    def _split_placement_key(cls, placement_type: str) -> tuple[str, str]:
        """``"low-tips"`` -> ``("low", "tips")``; bare ``"low"`` -> ``("low", "hand")``."""
        if "-" in placement_type:
            resolution, subset = placement_type.split("-", 1)
        else:
            resolution, subset = placement_type, "hand"
        if subset not in cls._PROBE_SUBSETS:
            raise ValueError(
                f"Unknown probe subset {subset!r} in placement {placement_type!r}; "
                f"expected one of {cls._PROBE_SUBSETS}."
            )
        return resolution, subset

    @classmethod
    def _filter_probe_data_by_subset(
        cls, probe_data: dict[str, dict[str, Any]], metadata: Any, subset: str
    ) -> dict[str, dict[str, Any]]:
        """Restrict whole-hand ``probe_data`` to the link subset implied by ``subset``."""
        if subset == "hand":
            return probe_data
        if subset == "tips":
            keep = set(metadata.fingertip_links)
        elif subset == "palm":
            keep = {metadata.palm_link}
        elif subset == "midfinger":
            keep = set(metadata.finger_links) - set(metadata.fingertip_links)
        elif subset == "fingers":
            keep = set(metadata.finger_links)
        else:
            raise ValueError(f"Unknown probe subset {subset!r}; expected one of {cls._PROBE_SUBSETS}.")
        return {link_name: probe for link_name, probe in probe_data.items() if link_name in keep}

    @staticmethod
    def _parse_probe_config(probe_cfg_path: str) -> dict[str, dict[str, Any]]:
        """Parse a probe JSON into per-link probe arrays.

        Each link maps to:
          - ``local_pos``:    ``np.ndarray`` of shape ``(R, C, 3)``
          - ``local_normal``: ``np.ndarray`` of shape ``(R, C, 3)``
          - ``radius``:       scalar ``float`` when every probe in the link
            shares the same radius; otherwise ``np.ndarray`` of shape ``(R, C)``
            so zero-radius "padding" cells round-trip alongside real probes.

        Grid records keep their native ``(R, C, ...)`` rectangle -- zero-radius
        padding probes are NOT dropped, so the 2D structure stays intact for the
        canvas-aware encoders. Non-grid records become ``(N, 1, ...)``.

        Ragged grid rows are an error: a grid record with mismatched row widths
        can't be put back into a rectangle without dropping cells.
        """
        import json

        with open(probe_cfg_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Invalid probe JSON format in {probe_cfg_path}: expected top-level list")

        def _probe_row(probe: Any) -> np.ndarray | None:
            if not isinstance(probe, dict):
                return None
            pos, normal, radius = probe.get("pos"), probe.get("normal"), probe.get("radius")
            if not (isinstance(pos, list) and len(pos) == 3):
                return None
            if not (isinstance(normal, list) and len(normal) == 3):
                return None
            try:
                return np.asarray([*pos, *normal, radius], dtype=np.float64)
            except (TypeError, ValueError):
                return None

        grid_by_link: dict[str, list[list[np.ndarray]]] = {}
        flat_by_link: dict[str, list[np.ndarray]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            link_name = record.get("link_name")
            probes = record.get("probes")
            if not isinstance(link_name, str) or not isinstance(probes, list):
                continue
            if record.get("kind") == "grid":
                link_rows = grid_by_link.setdefault(link_name, [])
                for row in probes:
                    if not isinstance(row, list):
                        continue
                    row_arrs = [arr for cell in row if (arr := _probe_row(cell)) is not None]
                    if row_arrs:
                        link_rows.append(row_arrs)
            else:
                link_cells = flat_by_link.setdefault(link_name, [])
                for cell in probes:
                    arr = _probe_row(cell)
                    if arr is not None:
                        link_cells.append(arr)

        probe_data: dict[str, dict[str, Any]] = {}
        # Preserve JSON record order so downstream sensor and encoder-projs ordering
        # is deterministic across processes. `set` iteration is hash-randomised, which
        # gave each run a different per-finger projs order and broke --resume on
        # checkpoints written by an earlier (different-seeded) process.
        for link_name in dict.fromkeys((*grid_by_link, *flat_by_link)):
            grid = grid_by_link.get(link_name)
            flat = flat_by_link.get(link_name)

            if grid:
                row_widths = {len(row) for row in grid}
                if len(row_widths) != 1:
                    raise ValueError(
                        f"_parse_probe_config: link {link_name!r} has ragged grid rows "
                        f"(widths={sorted(row_widths)}); expected a uniform rectangle."
                    )
                arr = np.array(grid, dtype=np.float64)  # (R, C, 7)
            else:
                if not flat:
                    continue
                arr = np.stack(flat)[:, None, :]  # (N, 1, 7)

            radii = arr[..., 6]
            nonzero = radii[radii != 0.0]
            if nonzero.size and not np.all(nonzero == nonzero.flat[0]):
                raise ValueError(
                    f"_parse_probe_config: link {link_name!r} has mixed nonzero radii "
                    f"{sorted(set(nonzero.tolist()))}; expected at most one nonzero radius value per link."
                )
            uniform = nonzero.size == radii.size and nonzero.size > 0
            probe_data[link_name] = {
                "local_pos": arr[..., :3],
                "local_normal": arr[..., 3:6],
                "radius": float(nonzero.flat[0]) if uniform else radii.copy(),
            }
        return probe_data


def derive_tactile_layout(config: EdenRLConfig) -> dict[str, Any]:
    """Derive ``TactileLayout`` kwargs from the active ``sensors_options``.

    Scans every ``tactile_*`` entry in ``config.sensors_options``, infers
    each sensor's ``(H_s, W_s)`` from its ``probe_local_pos`` array and the
    common ``features_per_probe`` from the matching ``TactileSensorSpec``, and
    pulls ``history_length`` from the ``tactile_sensors`` observation group.

    Heterogeneous grids are fine -- the grid-aware tactile encoders run a
    per-sensor stack sized to each patch's native shape.
    """
    sensors_options = getattr(config, "sensors_options", None)
    if sensors_options is None:
        raise ValueError("derive_tactile_layout: config has no sensors_options.")

    # Effective temporal_reduction lives on the TactileSensorRead term (written by
    # TactileSensorsMod from the --temporal_reduction CLI arg). When set to "none",
    # each probe carries history_length * features_per_probe features.
    temporal_reduction = "none"
    tactile_group = getattr(getattr(config, "observation_options", None), "tactile_sensors", None)
    if tactile_group is not None:
        term = getattr(tactile_group, "tactile_sensors", None)
        if term is not None:
            temporal_reduction = getattr(term, "temporal_reduction", "none")

    grid_shapes: list[tuple[int, int]] = []
    features_per_probe: int | None = None
    substep_multiplier: int = 1
    for sensor_name, sensor_options in sensors_options:
        if not sensor_name.startswith("tactile_"):
            continue
        spec = spec_for_sensor_name(sensor_name)
        if spec is None:
            continue
        if features_per_probe is None:
            features_per_probe = spec.features_per_probe
        elif spec.features_per_probe != features_per_probe:
            raise ValueError(
                f"derive_tactile_layout: mixed sensor types in tactile_* sensors "
                f"({features_per_probe}-d vs {spec.features_per_probe}-d per probe)."
            )

        if temporal_reduction == "none":
            this_hist = int(getattr(sensor_options.sensor, "history_length", 0) or 0) or 1
            if substep_multiplier == 1:
                substep_multiplier = this_hist
            elif this_hist != substep_multiplier:
                raise ValueError(
                    f"derive_tactile_layout: temporal_reduction='none' requires a uniform "
                    f"gs sensor history_length across tactile_* sensors "
                    f"({substep_multiplier} vs {this_hist})."
                )

        probe_local_pos = getattr(sensor_options.sensor, "probe_local_pos", None)
        shape = _grid_shape(probe_local_pos)
        if shape is None:
            raise ValueError(
                f"derive_tactile_layout: sensor {sensor_name!r} has no 2D-grid probe_local_pos "
                f"(got {type(probe_local_pos).__name__}). Grid-aware tactile encoders require "
                f"probe arrays loaded as 2D from the probe JSON."
            )
        grid_shapes.append(shape)

    if not grid_shapes:
        raise ValueError("derive_tactile_layout: no tactile_* sensors found in sensors_options.")

    history_length = 1
    tactile_group = getattr(getattr(config, "observation_options", None), "tactile_sensors", None)
    if tactile_group is not None:
        hl = getattr(tactile_group, "history_length", None)
        if isinstance(hl, int) and hl > 0:
            history_length = hl

    return {
        "num_sensors": len(grid_shapes),
        "grid_hw": tuple(grid_shapes),
        # When temporal_reduction='none' the postprocess folds the gs sensor's history axis into
        # the per-probe feature dim, so the layout must reflect the inflated feature count.
        "features_per_probe": int(features_per_probe) * substep_multiplier,
        "history_length": history_length,
    }


_SENSOR_ASSETS_ROOT = Path(__file__).resolve().parent / "assets" / "sensors"


def _iter_tactile_link_names(config: EdenRLConfig) -> list[str]:
    """Return per-tactile-sensor ``attach_link_name`` in the same order ``derive_tactile_layout`` iterates."""
    sensors_options = getattr(config, "sensors_options", None)
    if sensors_options is None:
        raise ValueError("derive_canvas_tactile_layout: config has no sensors_options.")
    links: list[str] = []
    for sensor_name, sensor_options in sensors_options:
        if not sensor_name.startswith("tactile_"):
            continue
        spec = spec_for_sensor_name(sensor_name)
        if spec is None:
            continue
        link = getattr(sensor_options, "attach_link_name", None)
        if link is None:
            raise ValueError(
                f"derive_canvas_tactile_layout: tactile sensor {sensor_name!r} has no attach_link_name."
            )
        links.append(link)
    return links


def _load_canvas_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    for k in ("name", "canvas_hw", "sensors"):
        if k not in data:
            raise ValueError(f"{path}: canvas JSON missing top-level key {k!r}.")
    return data


def _canvas_matches(
    canvas_data: dict[str, Any],
    active_links: list[str],
    grid_hw: tuple[tuple[int, int], ...],
) -> bool:
    """A canvas matches if every active link is placed on it and the canvas
    ``(h, w)`` equals the runtime grid ``(R, C)`` or its transpose. Extra
    placements in the canvas that aren't currently active (e.g. frozen fingers)
    are fine -- they're just dropped from the layout.
    """
    cmap = {s["link"]: (int(s["h"]), int(s["w"])) for s in canvas_data["sensors"]}
    if not set(active_links).issubset(cmap.keys()):
        return False
    for link, hw in zip(active_links, grid_hw, strict=True):
        probe = tuple(hw)
        if cmap[link] != probe and cmap[link] != (probe[1], probe[0]):
            return False
    return True


def _repack_active_placements(
    placements: tuple[tuple[int, int, int, int], ...],
    canvas_hw: tuple[int, int],
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[int, int]]:
    """Drop empty rows / columns from the canvas, returning a tighter layout.

    When some sensors in the canvas JSON aren't active (e.g. frozen fingers
    under ``in_palm_rotate``), their rows/columns sit unused. The conv kernel
    wastes capacity on those zero-padding regions and cross-sensor mixing has
    to bridge larger gaps. Repacking finds the rows + cols that any active
    sensor occupies and remaps each placement to compressed coordinates,
    preserving every sensor's size and relative ordering.
    """
    H, W = canvas_hw
    row_used = [False] * H
    col_used = [False] * W
    for r, c, h, w in placements:
        for i in range(r, r + h):
            row_used[i] = True
        for j in range(c, c + w):
            col_used[j] = True

    row_map: list[int] = [0] * H
    new_H = 0
    for i in range(H):
        row_map[i] = new_H
        if row_used[i]:
            new_H += 1
    col_map: list[int] = [0] * W
    new_W = 0
    for j in range(W):
        col_map[j] = new_W
        if col_used[j]:
            new_W += 1

    new_placements = tuple(
        (row_map[r], col_map[c], h, w) for (r, c, h, w) in placements
    )
    return new_placements, (new_H, new_W)


def derive_canvas_tactile_layout(config: EdenRLConfig) -> dict[str, Any]:
    """Derive ``CanvasTactileLayout`` kwargs by matching active sensors to a canvas JSON asset.

    Scans ``src/assets/sensors/*/*/canvas_*.json`` and selects the one whose
    declared per-link ``(h, w)`` matches the active tactile sensors' grid shapes
    (or their transposes). Per-sensor placements are filtered + ordered to match
    the ``sensors_options`` iteration order so the encoder's slicer and scatter
    line up. The selected placements are then **repacked** -- empty rows /
    columns left by frozen sensors are dropped so the conv operates on a dense
    canvas.

    Raises a clear error if no canvas matches.
    """
    base = derive_tactile_layout(config)
    active_links = _iter_tactile_link_names(config)
    if len(active_links) != base["num_sensors"]:
        raise ValueError(
            f"derive_canvas_tactile_layout: link-name count {len(active_links)} != num_sensors "
            f"{base['num_sensors']} (iteration drift)."
        )

    candidates = sorted(_SENSOR_ASSETS_ROOT.glob("*/*/canvas_*.json"))
    if not candidates:
        raise ValueError(
            f"derive_canvas_tactile_layout: no canvas JSONs found under {_SENSOR_ASSETS_ROOT}."
        )

    for path in candidates:
        canvas_data = _load_canvas_json(path)
        if _canvas_matches(canvas_data, active_links, base["grid_hw"]):
            placement_lookup = {
                s["link"]: (int(s["row"]), int(s["col"]), int(s["h"]), int(s["w"]))
                for s in canvas_data["sensors"]
            }
            placements = tuple(placement_lookup[link] for link in active_links)
            canvas_hw = tuple(int(v) for v in canvas_data["canvas_hw"])
            placements, canvas_hw = _repack_active_placements(placements, canvas_hw)
            return {
                **base,
                "canvas_hw": canvas_hw,
                "placements": placements,
                "canvas_name": canvas_data.get("name", ""),
                "link_names": tuple(active_links),
            }

    raise ValueError(
        "derive_canvas_tactile_layout: no canvas JSON matches the active tactile sensors. "
        f"Active links={active_links}, grid_hw={base['grid_hw']}. "
        f"Searched: {[str(p.relative_to(_SENSOR_ASSETS_ROOT)) for p in candidates]}."
    )


def _grid_shape(probe_local_pos: Any) -> tuple[int, int] | None:
    """Return ``(rows, cols)`` of a 2D probe array; ``None`` if shape is flat / unknown."""
    if probe_local_pos is None:
        return None
    if isinstance(probe_local_pos, np.ndarray):
        if probe_local_pos.ndim == 3 and probe_local_pos.shape[-1] == 3:
            return int(probe_local_pos.shape[0]), int(probe_local_pos.shape[1])
        return None
    if isinstance(probe_local_pos, (list, tuple)):
        if not probe_local_pos:
            return None
        first = probe_local_pos[0]
        # 2D grid: list of rows of triples. 1D flat: list of triples.
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            cols = len(first)
            for row in probe_local_pos:
                if len(row) != cols:
                    return None  # ragged
            return len(probe_local_pos), cols
    return None


class ObservationHistoryLengthMod(TaskMod):
    def __init__(
        self,
        obs_hist: Annotated[int, "Observation history length. -1 to use default history length from config."] = -1,
        *,
        obs_names: list[str],
        name: str = "",
    ) -> None:
        self.obs_hist = obs_hist
        self.obs_names = obs_names
        if name:
            self.with_prefix(name)

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        if self.obs_hist < 0:
            return config

        for obs_name in self.obs_names:
            if hasattr(config.observation_options, obs_name):
                getattr(config.observation_options, obs_name).history_length = self.obs_hist
            else:
                _log_info(
                    f"{self.prefix + '_' if self.prefix else ''}history_len: skipped {obs_name!r}"
                    "(observation group not found on config)"
                )
        return config


class RewardWeightCurriculumMod(TaskMod):
    def __init__(
        self,
        rwc: Annotated[bool, "Whether to use reward weight curriculum."] = False,
        *,
        curriculum_cfg: dict[str, Any],
        name: str = "",
    ):
        self.rwc = rwc
        self.cfg = curriculum_cfg
        if name:
            self.with_prefix(name)

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        if self.rwc:
            config.metric_options.reward_weight_curriculum = UpdateCurriculumWeights.configure(**self.cfg)
        return config


class RslRlRunnerMod(TaskMod):
    DEFAULT_DISTRIBUTION: rl.RslRlDistributionOptions = rl.RslRlGaussianDistributionOptions(init_std=1.0)
    DEFAULT_MLP_CFG: dict[str, Any] = {
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
    }
    DEFAULT_RNN_CFG: dict[str, Any] = {
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "rnn_type": "lstm",
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 4,
    }
    DEFAULT_RND_CFG: dict[str, Any] = {
        "weight": 100.0,
        "learning_rate": 1e-3,
        "num_outputs": 8,
        "state_normalization": True,
        "reward_normalization": True,
    }
    DEFAULT_PPO_OPTIONS = rl.RslRlPpoAlgorithmOptions(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    DEFAULT_DISTILL_OPTIONS = rl.RslRlDistillationAlgorithmOptions(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=2,
        max_grad_norm=None,
        optimizer="adam",
        loss_type="mse",
    )
    _MODEL_CLASSES = {
        "actor": {
            "mlp": rl.RslRlActorOptions,
            "rnn": rl.RslRlActorRNNOptions,
            "cnn": rl.RslRlActorCNNOptions,
            "enc_mlp": RslRlActorPreEncodeMLPOptions,
            "enc_rnn": RslRlActorPreEncodeRNNOptions,
            "tac_mlp": RslRlActorTactilePreEncodeMLPOptions,
            "tac_rnn": RslRlActorTactilePreEncodeRNNOptions,
        },
        "critic": {
            "mlp": rl.RslRlMLPModelOptions,
            "rnn": rl.RslRlRNNModelOptions,
            "cnn": rl.RslRlCNNModelOptions,
            "enc_mlp": RslRlPreEncodeMLPOptions,
            "enc_rnn": RslRlPreEncodeRNNOptions,
            "tac_mlp": RslRlTactilePreEncodeMLPOptions,
            "tac_rnn": RslRlTactilePreEncodeRNNOptions,
        },
    }

    def __init__(
        self,
        actor: Annotated[
            Literal["mlp", "rnn", "enc_mlp", "enc_rnn", "tac_mlp", "tac_rnn"],
            "Actor (or student if distillation) model: mlp (default), rnn, enc_*, or tac_* (per-sensor tactile).",
        ] = "mlp",
        critic: Annotated[
            Literal["mlp", "rnn", "enc_mlp", "enc_rnn", "tac_mlp", "tac_rnn"],
            "Critic (or teacher if distillation) model: mlp (default), rnn, enc_*, or tac_* (per-sensor tactile).",
        ] = "mlp",
        algo: Annotated[
            Literal["ppo", "distill"],
            "Algorithm: ppo (default) or distill.",
        ] = "ppo",
        rnd: Annotated[float, "The RND weight."] = 0.0,
        max_iters: Annotated[
            int,
            "Total training iterations. -1 keeps the task config's default (RUNNER_CFG['max_iterations']).",
        ] = -1,
        *,
        runner_cfg: dict[str, Any] | None = None,
        ppo_options: rl.RslRlPpoAlgorithmOptions | None = None,
        distill_options: rl.RslRlDistillationAlgorithmOptions | None = None,
        mlp_cfg: dict[str, Any] | None = None,
        rnn_cfg: dict[str, Any] | None = None,
        cnn_cfg: dict[str, Any] | None = None,
        rnd_cfg: dict[str, Any] | None = None,
        encoder_cfg: dict[str, Any] | None = None,
        actor_distribution: rl.RslRlDistributionOptions | None = None,
        name: str = "",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.algo = algo
        self.rnd = rnd
        self.max_iters = max_iters
        self.runner_cfg: dict[str, Any] = runner_cfg or {}
        self.ppo_options: rl.RslRlPpoAlgorithmOptions = ppo_options or self.DEFAULT_PPO_OPTIONS
        self.distill_options: rl.RslRlDistillationAlgorithmOptions = distill_options or self.DEFAULT_DISTILL_OPTIONS
        self.actor_distribution: rl.RslRlDistributionOptions = actor_distribution or self.DEFAULT_DISTRIBUTION
        self.mlp_cfg: dict[str, Any] = mlp_cfg or self.DEFAULT_MLP_CFG
        self.rnn_cfg: dict[str, Any] = rnn_cfg or self.DEFAULT_RNN_CFG
        self.rnd_cfg: dict[str, Any] = rnd_cfg or self.DEFAULT_RND_CFG
        self.cnn_cfg: dict[str, Any] | None = cnn_cfg
        self.encoder_cfg: dict[str, Any] | None = encoder_cfg
        if name:
            self.with_prefix(name)

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        task_name = config.__class__.__name__
        runner_dict = {
            "experiment_name": task_name,
            "wandb_project": "eden-" + task_name,
            "logger": "wandb",
            "seed": 42,
            "num_steps_per_env": 24,
            "max_iterations": 100000,
            "save_interval": 5000,
        }
        runner_dict.update(copy.deepcopy(self.runner_cfg))
        if self.max_iters >= 0:
            runner_dict["max_iterations"] = self.max_iters
        filtered_obs_groups: dict[str, list[str]] = {}
        if "obs_groups" in runner_dict:
            filtered_obs_groups = self._filter_obs_groups_by_config(
                runner_dict["obs_groups"], config.observation_options, task_name
            )
            runner_dict["obs_groups"] = filtered_obs_groups

        algo = self.algo
        enc_actor_key: str | None = None
        enc_critic_key: str | None = None
        if "distill" in algo:
            enc_actor_key = "student" if "student" in filtered_obs_groups else "actor"
            enc_critic_key = "teacher" if "teacher" in filtered_obs_groups else "critic"
        actor_model = self._get_model_options(
            self.actor,
            filtered_obs_groups,
            is_actor=True,
            encoder_obs_group_key=enc_actor_key,
        )
        critic_model = self._get_model_options(
            self.critic,
            filtered_obs_groups,
            is_actor=False,
            encoder_obs_group_key=enc_critic_key,
        )
        rnd_weight = self.rnd
        ppo_options = copy.deepcopy(self.ppo_options)
        distill_options = copy.deepcopy(self.distill_options)
        if rnd_weight > 1e-8:
            rnd_cfg = copy.deepcopy(self.rnd_cfg)
            rnd_cfg["weight"] = rnd_weight
            rnd_options = rl.RslRlRndOptions(**rnd_cfg)
            ppo_options = ppo_options.model_copy(update={"rnd_cfg": rnd_options})
            distill_options = distill_options.model_copy(update={"rnd_cfg": rnd_options})

        if "distill" in algo:
            config.runner_options = rl.RslRlDistillationRunnerOptions(
                **runner_dict,
                algorithm=distill_options,
                student=actor_model,
                teacher=critic_model,
            )
        else:  # "ppo" in algo
            config.runner_options = rl.RslRlOnPolicyRunnerOptions(
                **runner_dict,
                algorithm=ppo_options,
                actor=actor_model,
                critic=critic_model,
            )
        return config

    @staticmethod
    def _filter_obs_groups_by_config(
        obs_groups: dict[str, list[str]],
        observations: Any,
        task_name: str,
    ) -> dict[str, list[str]]:
        filtered = {k: [] for k in obs_groups}
        for policy, names in obs_groups.items():
            for name in names:
                if hasattr(observations, name):
                    filtered[policy].append(name)
                else:
                    _log_info(f"obs_group {name!r} skipped (observation group not found)")
        if not filtered:
            raise ValueError(f"obs_groups is empty after filtering observation groups for {task_name!r}")
        return filtered

    def _get_model_options(
        self,
        model_type: str,
        obs_groups: dict[str, list[str]],
        is_actor: bool,
        encoder_obs_group_key: str | None = None,
    ) -> rl.RslRlModelOptions:
        model_dict = {}
        policy_name = "actor" if is_actor else "critic"
        enc_obs_key = (
            encoder_obs_group_key
            if encoder_obs_group_key is not None and encoder_obs_group_key in obs_groups
            else policy_name
        )

        if is_actor:
            model_dict["distribution_cfg"] = self.actor_distribution
        if "mlp" in model_type:
            model_dict.update(**self.mlp_cfg)
        elif "rnn" in model_type:
            model_dict.update(**self.rnn_cfg)
        elif "cnn" in model_type:
            model_dict.update(**self.cnn_cfg)
        if "enc" in model_type or model_type.startswith("tac_"):
            if self.encoder_cfg is None:
                raise ValueError(
                    f"encoder_cfg is required for model type {model_type!r}; pass encoder_cfg=... into RslRlRunnerMod."
                )
            # filter encoder_cfg to keys present in this policy's obs group (actor/critic or student/teacher)
            encoder_cfg = {}
            obs_names = obs_groups[enc_obs_key]
            for enc_name, enc_cfg in self.encoder_cfg.items():
                if enc_name in obs_names:
                    encoder_cfg[enc_name] = enc_cfg
                else:
                    _log_info(f"{model_type} ({enc_obs_key}): encoder {enc_name!r} skipped (not in observation set)")

            if not encoder_cfg:
                raise ValueError(f"encoder_cfg is empty after filtering observation groups for {model_type}")
            model_dict["encoder_cfg"] = encoder_cfg

        return self._MODEL_CLASSES[policy_name][model_type](**model_dict)


class DexHandRslRlRunnerMod(RslRlRunnerMod):
    """
    Stage 1: Teacher training. RND enabled. MLP model.
    Stage 2: Student-teacher distillation. Student can have different models (--model flag.)
    Stage 3: Student RL. RND disabled.
    """

    MODEL_PER_STAGE = {
        1: ("mlp", "mlp"),  # teacher actor, critic
        2: ("enc_mlp", "mlp"),  # student, teacher
        3: ("enc_mlp", "mlp"),  # student actor, critic
    }

    def __init__(
        self,
        stage: Annotated[
            Literal[1, 2, 3],
            "Stage of training: 1 (teacher RL), 2 (distillation), or 3 (student RL).",
        ] = 1,
        model: Annotated[
            Literal["mlp", "rnn", "enc_mlp", "enc_rnn", "tac_mlp", "tac_rnn"],
            "Model to use for student (mlp/rnn/enc_*/tac_*).",
        ] = "enc_mlp",
        tactile_encoder: Annotated[
            Literal[tuple(TACTILE_ENCODER_CFGS.keys())],
            "Encoder for the `tactile_sensors` obs group "
            "(mlp/rnn/tactile_cnn/tactile_convrnn/tactile_convrnn_big/tactile_canvas_cnn/tactile_canvas_convrnn). "
            "Use rnn or mlp for sensor types without a probe grid (agg_force, agg_bool, link_*, none).",
        ] = "mlp",
        encoder: Annotated[
            Literal[tuple(GROUP_ENCODER_CFGS.keys())],
            "Encoder for other encoded obs groups (proprio, ...).",
        ] = "rnn",
        rnd: Annotated[float, "The RND weight."] = 0.0,
        priv_student: Annotated[bool, "Test distillation with student obs same as teacher obs."] = False,
        max_iters: Annotated[
            int,
            "Total training iterations. -1 keeps the task config's default (RUNNER_CFG['max_iterations']).",
        ] = -1,
        *,
        runner_cfg: dict[str, Any] | None = None,
        ppo_options: rl.RslRlPpoAlgorithmOptions | None = None,
        distill_options: rl.RslRlDistillationAlgorithmOptions | None = None,
        mlp_cfg: dict[str, Any] | None = None,
        rnn_cfg: dict[str, Any] | None = None,
        cnn_cfg: dict[str, Any] | None = None,
        rnd_cfg: dict[str, Any] | None = None,
        encoder_cfg: dict[str, Any] | None = None,
        actor_distribution: rl.RslRlDistributionOptions | None = None,
        name: str = "",
    ) -> None:
        # encoder_cfg can stay None at construction; ``apply`` resolves the final
        # dict from the two named cfgs once it has the live sensors_options to
        # derive a TactileLayout from. A task-author override (explicit dict) is
        # stashed so it survives re-applies of the same modifier instance.
        super().__init__(
            rnd=rnd,
            max_iters=max_iters,
            runner_cfg=runner_cfg,
            ppo_options=ppo_options,
            distill_options=distill_options,
            mlp_cfg=mlp_cfg,
            rnn_cfg=rnn_cfg,
            cnn_cfg=cnn_cfg,
            rnd_cfg=rnd_cfg,
            encoder_cfg=encoder_cfg,
            actor_distribution=actor_distribution,
            name=name,
        )
        self._explicit_encoder_cfg = encoder_cfg
        self.stage = stage
        self.model = model
        self.tactile_encoder = tactile_encoder
        self.encoder = encoder
        self.priv_student = priv_student

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        task_name = config.__class__.__name__
        runner_dict = {
            "experiment_name": task_name,
            "wandb_project": "eden-" + task_name,
            "logger": "wandb",
            "seed": 42,
            "num_steps_per_env": 24,
            "max_iterations": 10000,
            "save_interval": 1000,
        }
        runner_dict.update(copy.deepcopy(self.runner_cfg))
        if self.max_iters >= 0:
            runner_dict["max_iterations"] = self.max_iters

        stage = self.stage
        if stage == 1:  # teacher RL
            runner_dict["obs_groups"]["actor"] = runner_dict["obs_groups"]["teacher"]
        if stage == 2 and self.priv_student:  # priv student distillation
            runner_dict["obs_groups"]["student"] = runner_dict["obs_groups"]["teacher"]
        if stage == 3:  # student RL
            runner_dict["obs_groups"]["actor"] = runner_dict["obs_groups"]["student"]
            # config.runner_options.algorithm.learning_rate *= 0.1

        filtered_obs_groups: dict[str, list[str]] = {}
        if "obs_groups" in runner_dict:
            filtered_obs_groups = self._filter_obs_groups_by_config(
                runner_dict["obs_groups"], config.observation_options, task_name
            )
            runner_dict["obs_groups"] = filtered_obs_groups

        is_distill = stage == 2
        enc_actor_key: str | None = None
        enc_critic_key: str | None = None
        if is_distill:
            enc_actor_key = "student" if "student" in filtered_obs_groups else "actor"
            enc_critic_key = "teacher" if "teacher" in filtered_obs_groups else "critic"

        # Resolve encoder_cfg from the two CLI flags unless a task author wired
        # a full dict in at registration. Layout for grid-aware tactile
        # encoders is derived from the live sensors_options on every apply so
        # the same modifier instance can be reused across configs.
        if self._explicit_encoder_cfg is not None:
            self.encoder_cfg = self._explicit_encoder_cfg
        else:
            self.encoder_cfg = self._resolve_encoder_cfg(config, filtered_obs_groups)

        actor_model, critic_model = self.MODEL_PER_STAGE[stage]
        if stage in (2, 3):
            actor_model = self.model

        actor_model_options = self._get_model_options(
            actor_model,
            filtered_obs_groups,
            is_actor=True,
            encoder_obs_group_key=enc_actor_key,
        )
        critic_model_options = self._get_model_options(
            critic_model,
            filtered_obs_groups,
            is_actor=False,
            encoder_obs_group_key=enc_critic_key,
        )
        rnd_weight = self.rnd
        ppo_options = copy.deepcopy(self.ppo_options)
        distill_options = copy.deepcopy(self.distill_options)
        if rnd_weight > 1e-8:
            rnd_cfg = copy.deepcopy(self.rnd_cfg)
            rnd_cfg["weight"] = rnd_weight
            rnd_options = rl.RslRlRndOptions(**rnd_cfg)
            ppo_options = ppo_options.model_copy(update={"rnd_cfg": rnd_options})
            distill_options = distill_options.model_copy(update={"rnd_cfg": rnd_options})

        if is_distill:
            config.runner_options = rl.RslRlDistillationRunnerOptions(
                **runner_dict,
                algorithm=distill_options,
                student=actor_model_options,
                teacher=critic_model_options,
            )
        else:
            config.runner_options = rl.RslRlOnPolicyRunnerOptions(
                **runner_dict,
                algorithm=ppo_options,
                actor=actor_model_options,
                critic=critic_model_options,
            )
        return config

    def _resolve_encoder_cfg(
        self,
        config: EdenRLConfig,
        filtered_obs_groups: dict[str, list[str]],
    ) -> dict[str, dict[str, Any]]:
        """Assemble encoder_cfg from --tactile_encoder + --encoder selections.

        Only includes obs group keys present in at least one filtered policy
        group (so e.g. a task without ``tactile_sensors`` skips that entry).
        Injects a derived ``TactileLayout`` for grid-aware tactile kinds.
        """
        all_group_names = {name for names in filtered_obs_groups.values() for name in names}
        encoder_cfg: dict[str, dict[str, Any]] = {}

        if "tactile_sensors" in all_group_names:
            tcfg = copy.deepcopy(TACTILE_ENCODER_CFGS[self.tactile_encoder])
            kind = tcfg.get("kind")
            if kind in ("tactile_cnn", "tactile_convrnn"):
                tcfg["tactile_layout"] = derive_tactile_layout(config)
            elif kind in ("tactile_canvas_cnn", "tactile_canvas_convrnn"):
                tcfg["tactile_layout"] = derive_canvas_tactile_layout(config)
            encoder_cfg["tactile_sensors"] = tcfg

        if "proprio" in all_group_names:
            encoder_cfg["proprio"] = copy.deepcopy(GROUP_ENCODER_CFGS[self.encoder])

        return encoder_cfg
