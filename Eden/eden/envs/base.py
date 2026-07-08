"""Core vectorized environment base classes: EnvBase and RLEnvBase.

The class hierarchy mirrors the config hierarchy in :mod:`eden.utils.configs`:

- :class:`EnvBase` — scene, entities, sensors, and the action/observation/event managers.
- :class:`RLEnvBase` — adds reward, termination, command, and curriculum managers plus the RL ``step``.

Build environments with ``RLEnvBase.from_config(cfg)`` (not ``from_cfg``); the ``config``
property reconstructs the config object. In :meth:`RLEnvBase.step`, auto-reset happens
**after** reward/termination but **before** observation, so the returned observations
reflect the fresh post-reset state for done environments.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable

import genesis as gs
import numpy as np
import torch
from genesis.options.renderers import BatchRenderer, Rasterizer, RendererOptions
from genesis.repr_base import RBC
from genesis.typing import Vec2FType

import eden as en
from eden.constants import EventMode
from eden.entities.camera import Camera
from eden.entities.sensor import Sensor
from eden.envs.draw_debug_mixin import DrawDebugMixin
from eden.options import (
    ActionManagerOptions,
    CameraOptions,
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
    SensorOptions,
    SensorsOptions,
    TerminationManagerOptions,
    TerrainOptions,
)
from eden.types import VecEnvReset, VecEnvStep
from eden.utils.configs import EdenConfig
from eden.utils.misc import sanitize_envs_idx
from eden.utils.random import set_random_seed

if TYPE_CHECKING:
    from genesis.engine.couplers import IPCCoupler, LegacyCoupler, SAPCoupler
    from genesis.engine.solvers import RigidSolver

    from eden.entities.base import Entity
    from eden.entities.camera import Camera
    from eden.options.entities import EntityOptionsLike
    from eden.utils.configs import EdenRLConfig


_STARTUP, _RESET, _INTERVAL = EventMode


def _entity_class_for_options(options):
    """Return the appropriate Entity subclass for the given options."""
    from eden.entities.fem import FEMEntity
    from eden.entities.particle import ParticleEntity
    from eden.entities.rigid import RigidEntity
    from eden.entities.terrain import Terrain
    from eden.options.materials import (
        FEMClothMaterialOptions,
        FEMElasticMaterialOptions,
        FEMMuscleMaterialOptions,
        MPMElasticMaterialOptions,
        MPMElastoPlasticMaterialOptions,
        MPMLiquidMaterialOptions,
        PBDClothMaterialOptions,
        PBDElasticMaterialOptions,
        PBDLiquidMaterialOptions,
        PBDParticleMaterialOptions,
        SPHLiquidMaterialOptions,
    )

    if isinstance(options, TerrainOptions):
        return Terrain
    material = getattr(options, "material", None)
    if isinstance(
        material,
        (
            MPMElasticMaterialOptions,
            MPMElastoPlasticMaterialOptions,
            MPMLiquidMaterialOptions,
            SPHLiquidMaterialOptions,
            PBDLiquidMaterialOptions,
            PBDClothMaterialOptions,
            PBDElasticMaterialOptions,
            PBDParticleMaterialOptions,
        ),
    ):
        return ParticleEntity
    if isinstance(
        material,
        (
            FEMElasticMaterialOptions,
            FEMClothMaterialOptions,
            FEMMuscleMaterialOptions,
        ),
    ):
        return FEMEntity
    return RigidEntity


class EnvBase(DrawDebugMixin, RBC):
    """
    Base class for all environments.

    Parameters
    ----------
    env_options: EnvOptions
        Environment options.
    scene_options: SceneOptions
        Scene options.
    observation_options: ObservationManagerOptions
        Observation manager options.
    event_options: EventManagerOptions
        Event manager options.
    action_options: ActionManagerOptions
        Action manager options.
    metrics_options: MetricManagerOptions
        Metric manager options.
    recorder_options: RecorderManagerOptions
        Recorder manager options.
    cameras_options: CamerasOptions
        Cameras options.
    sensors_options: SensorsOptions
        Sensors options.
    renderer_options: RendererOptions
        Renderer options.
    show_viewer: bool
        Whether to show the viewer.
    eval_mode: bool
        Whether to run in eval mode.
    """

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        # Handling of default arguments
        self.env_options = env_options or EnvOptions()
        self.scene_options = scene_options or SceneOptions()
        self.observation_options = observation_options or ObservationManagerOptions()
        self.event_options = event_options or EventManagerOptions()
        self.action_options = action_options or ActionManagerOptions()
        self.metric_options = metric_options or MetricManagerOptions()
        self.recorder_options = recorder_options or RecorderManagerOptions()
        self.cameras_options = cameras_options or CamerasOptions()
        self.sensors_options = sensors_options or SensorsOptions()
        self.renderer_options = renderer_options or Rasterizer()

        if isinstance(self.renderer_options, BatchRenderer):
            en.logger.info(f"You are using {type(self.renderer_options).__name__} as the renderer options.")
            en.logger.info("Consider adding lights to the scene by `env.scene.add_light()`.")

        self.headless = not show_viewer
        if eval_mode:
            self.training = False
            self.num_envs = self.env_options.num_eval_envs
        else:
            self.training = True
            self.num_envs = self.env_options.num_envs

        self.sim_dt = self.env_options.sim_dt
        self.dt = self.env_options.decimation * self.env_options.sim_dt
        self._sim_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.obs_buf = None
        self.reset_info = None

        # hooks
        self._pre_build_hooks: list[Callable[[], None]] = []
        self._post_build_hooks: list[Callable[[], None]] = []

        # create scene
        # Build default scene kwargs from env_options fields
        scene_kwargs = dict(
            sim_options=gs.options.SimOptions(
                dt=env_options.sim_dt,
                substeps=env_options.sim_substeps,
                requires_grad=env_options.requires_grad,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / env_options.sim_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                enable_default_keybinds=env_options.enable_default_keybinds,
            ),
            vis_options=gs.options.VisOptions(
                env_separate_rigid=env_options.env_separate_rigid,
                show_link_frame=env_options.show_link_frame,
                show_world_frame=env_options.show_world_frame,
                show_cameras=env_options.show_camera_frustums,
                segmentation_level=env_options.segmentation_level,
                background_color=env_options.background_color,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=env_options.sim_dt,
                constraint_solver={
                    "newton": gs.constraint_solver.Newton,
                    "cg": gs.constraint_solver.CG,
                }[env_options.solver],
                constraint_timeconst=env_options.constraint_timeconst
                if env_options.constraint_timeconst is not None
                else max(0.01, 2 * env_options.sim_dt / env_options.sim_substeps),
                use_gjk_collision=env_options.use_gjk_collision,
                max_collision_pairs=env_options.max_collision_pairs,
                multiplier_collision_broad_phase=env_options.multiplier_collision_broad_phase,
                iterations=env_options.iterations,
                tolerance=env_options.tolerance,
                ls_iterations=env_options.ls_iterations,
                ls_tolerance=env_options.ls_tolerance,
                enable_collision=True,
                enable_self_collision=env_options.enable_self_collision,
                enable_joint_limit=True,
                use_hibernation=env_options.use_hibernation,
                batch_links_info=env_options.batch_links_info,
                batch_joints_info=env_options.batch_joints_info,
                batch_dofs_info=env_options.batch_dofs_info,
                enable_multi_contact=env_options.enable_multi_contact,
                noslip_iterations=env_options.noslip_iterations,
            ),
            coupler_options=env_options.coupler_options,
            fem_options=env_options.fem_options,
            mpm_options=env_options.mpm_options,
            sph_options=env_options.sph_options,
            pbd_options=env_options.pbd_options,
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=env_options.show_FPS,
            ),
            renderer=renderer_options,
            show_viewer=show_viewer,
        )

        # Allow extra fields on EnvOptions to override auto-constructed defaults
        extras = env_options.__pydantic_extra__ or {}
        internal_keys = {"_option_module_", "_option_class_"}
        for key, value in extras.items():
            if key not in internal_keys:
                scene_kwargs[key] = value

        # Explicit kwargs take highest priority
        scene_kwargs.update(kwargs)

        self.scene = gs.Scene(**scene_kwargs)

        entity_data: dict[str, EntityOptionsLike] = {}
        for key in self.scene_options.keys():
            if key == "attachments_dict":
                continue
            entity_data[key] = getattr(self.scene_options, key)

        attachments_dict = self.scene_options.attachments_dict or {}
        if attachments_dict:
            # NOTE: parent entity must be instantiated before child entity.
            # Build a topological order so parents are created before children.
            in_degree = {name: 0 for name in entity_data.keys()}
            parent_to_child_dict = defaultdict(list)
            for child_entity_name, (parent_entity_name, _) in attachments_dict.items():
                parent_to_child_dict[parent_entity_name].append(child_entity_name)
                if child_entity_name in in_degree:
                    in_degree[child_entity_name] += 1

            queue = deque([name for name, deg in in_degree.items() if deg == 0])
            ordered_keys: list[str] = []
            while queue:
                node = queue.popleft()
                ordered_keys.append(node)
                for child in parent_to_child_dict.get(node, []):
                    if child in in_degree:
                        in_degree[child] -= 1
                        if in_degree[child] == 0:
                            queue.append(child)

            # NOTE: if there's a cycle or unknown nodes, raise an error.
            if len(ordered_keys) != len(entity_data):
                missing = set()
                for child, (parent, _) in attachments_dict.items():
                    if parent not in entity_data:
                        missing.add(parent)
                if missing:
                    raise ValueError(f"attachments_dict references nonexistent entities: {missing}")
                raise ValueError(f"Cycle detected in attachments_dict: {attachments_dict}")
        else:
            ordered_keys = list(entity_data.keys())

        self.entities: dict[str, Entity] = {}
        for entity_name in ordered_keys:
            entity_options: EntityOptionsLike = entity_data[entity_name]
            entity_cls = _entity_class_for_options(entity_options)
            entity = entity_cls(env=self, options=entity_options)
            entity._name = entity_name
            self.entities[entity_name] = entity

        for entity_name, (
            attach_to_entity_name,
            attach_to_link_name,
        ) in self.scene_options.attachments_dict.items():
            self.entities[entity_name].attach_to(self.entities[attach_to_entity_name], attach_to_link_name)

        self.cameras: dict[str, Camera] = {}
        self._setup_cameras()

        self.sensors: dict[str, Sensor] = {}
        self._setup_sensors()

    @classmethod
    def from_config(cls, cfg: EdenConfig, **kwargs) -> "EnvBase":
        """
        Create an environment from a configuration object.

        Parameters
        ----------
        cfg: EdenConfig
            The configuration object.
        **kwargs: dict
            Additional keyword arguments.
            show_viewer: bool
                Whether to show the viewer.
            eval_mode: bool
                Whether to run in eval mode.
        """
        env = cls(
            env_options=cfg.env_options,
            scene_options=cfg.scene_options,
            observation_options=cfg.observation_options,
            event_options=cfg.event_options,
            action_options=cfg.action_options,
            metric_options=cfg.metric_options,
            recorder_options=cfg.recorder_options,
            cameras_options=cfg.cameras_options,
            sensors_options=cfg.sensors_options,
            renderer_options=cfg.renderer_options,
            show_viewer=kwargs.get("show_viewer", False),
            eval_mode=kwargs.get("eval_mode", False),
        )

        return env

    @gs.assert_unbuilt
    def _setup_cameras(self):
        for camera_name in self.cameras_options.keys():
            camera_options: CameraOptions = getattr(self.cameras_options, camera_name)
            camera = Camera(env=self, options=camera_options)
            self.cameras[camera_name] = camera

    @gs.assert_unbuilt
    def _setup_sensors(self):
        for sensor_name in self.sensors_options.keys():
            sensor_options: SensorOptions = getattr(self.sensors_options, sensor_name)
            sensor = Sensor(env=self, options=sensor_options)
            self.sensors[sensor_name] = sensor

    @gs.assert_unbuilt
    def register_pre_build_hook(self, hook: Callable[[], None]) -> None:
        self._pre_build_hooks.append(hook)

    @gs.assert_unbuilt
    def register_post_build_hook(self, hook: Callable[[], None]) -> None:
        self._post_build_hooks.append(hook)

    @gs.assert_unbuilt
    def build(
        self,
        env_spacing: Vec2FType | None = None,
        center_envs_at_origin: bool | None = None,
    ):
        """Build the underlying Genesis scene and finalize all entities.

        Runs :meth:`pre_build_setup`, builds the vectorized scene, then runs
        :meth:`post_build_setup`. Must be called once before stepping the environment.

        Parameters
        ----------
        env_spacing : tuple[float, float], optional
            Spacing between environments in the xy-plane. Falls back to
            ``env_options.env_spacing`` when None.
        center_envs_at_origin : bool, optional
            Whether to center the grid of environments at the world origin.
            Falls back to ``env_options.center_envs_at_origin`` when None.
        """
        env_spacing = env_spacing if env_spacing is not None else self.env_options.env_spacing
        center_envs_at_origin = (
            center_envs_at_origin if center_envs_at_origin is not None else self.env_options.center_envs_at_origin
        )
        self.pre_build_setup()
        self.scene.build(
            n_envs=self.num_envs,
            env_spacing=env_spacing,
            center_envs_at_origin=center_envs_at_origin,
        )
        self.post_build_setup()
        self._verify_build()

    @gs.assert_unbuilt
    def pre_build_setup(self):
        for entity in self.entities.values():
            entity.pre_build()

        for camera in self.cameras.values():
            camera.pre_build()

        for sensor in self.sensors.values():
            sensor.pre_build()

        for hook in self._pre_build_hooks:
            hook()

    @gs.assert_built
    def post_build_setup(self):
        for entity in self.entities.values():
            entity.post_build()

        for camera in self.cameras.values():
            camera.post_build()

        self._load_managers()

        for hook in self._post_build_hooks:
            hook()

        # NOTE: save the post setup state for reset
        self.rigid_solver._queried_states.clear()
        self.reset_state = self.scene.get_state()

    @gs.assert_built
    def _verify_build(self):
        if self.rigid_solver.n_equalities > 0:
            if self.env_options.sim_dt / self.env_options.sim_substeps > 0.004:
                en.logger.warning(
                    f"""Modify the sim_dt or sim_substeps to make effective dt smaller than 0.004
                    for stability when there are closed-link represented by equality constraints.
                    Current sim_dt: {self.env_options.sim_dt}, sim_substeps: {self.env_options.sim_substeps}"""
                )

        # NOTE: check entities
        from eden.entities.fem import FEMEntity
        from eden.entities.particle import ParticleEntity

        for name, entity in self.entities.items():
            # Skip mass check for non-rigid entities (MPM/SPH/FEM) — their mass is
            # determined by material density and vertex/particle count, not a single rigid mass.
            # Also skip kinematic entities which have no meaningful rigid mass.
            if isinstance(entity, (ParticleEntity, FEMEntity)) or isinstance(
                entity.material, en.materials.KinematicMaterialOptions
            ):
                continue
            if entity.is_heterogeneous:
                masses = entity.get_mass()
                if not entity.is_fixed_base and (masses < 0.001).any():
                    en.logger.warning(
                        f"""{name} has mass of {masses.min()} kg, potentially leads to instability in simulation.
                        Setting mass to 0.01 kg."""
                    )
                    entity.set_mass(0.01)
            else:
                if not entity.is_fixed_base and entity.get_mass() < 0.001:
                    en.logger.warning(
                        f"""{name} has mass of {entity.get_mass()} kg, potentially leads to instability in simulation.
                        Setting mass to 0.01 kg."""
                    )
                    entity.set_mass(0.01)

    def destroy(self):
        self.scene.destroy()

    @gs.assert_built
    def _load_managers(self):
        from eden.managers import (
            ActionManager,
            EventManager,
            MetricManager,
            ObservationManager,
            RecorderManager,
        )

        self.metric_manager = MetricManager(self, self.metric_options)
        if not self.metric_manager._terms:
            self.metric_manager = None

        self.recorder_manager = RecorderManager(self, self.recorder_options)

        self.action_manager = ActionManager(self, self.action_options)
        self.event_manager = EventManager(self, self.event_options)
        self.observation_manager = ObservationManager(self, self.observation_options)

        if _STARTUP in self.event_manager.available_modes:
            self.event_manager.compute(mode=_STARTUP)

    @gs.assert_built
    def summary(self) -> str:
        """Return a human-readable summary of the environment and its managers."""
        msg = "Environment Summary:\n"
        msg += f"Number of environments: {self.num_envs}\n"
        msg += f"Step time step: {self.dt} s\n"
        msg += f"Simulation time step: {self.sim_dt} s\n"
        msg += f"Decimation: {self.env_options.decimation}\n"

        msg += "Managers:\n"
        msg += self.action_manager.summary()
        msg += self.event_manager.summary()
        msg += self.observation_manager.summary()
        if self.metric_manager is not None:
            msg += self.metric_manager.summary()
        msg += self.recorder_manager.summary()
        return msg

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Seed all RNGs used by the environment.

        Parameters
        ----------
        seed : int, optional
            The seed to set. ``-1`` (the default) draws a random seed in ``[0, 10000)``.

        Returns
        -------
        int
            The seed that was actually applied.
        """
        if seed == -1:
            seed = np.random.randint(0, 10_000)
        en.logger.info(f"Setting seed: {seed}")
        set_random_seed(seed)
        return seed

    @cached_property
    def _all_indices(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device)

    @property
    def is_built(self) -> bool:
        """Whether the scene has been built."""
        return self.scene._is_built

    @property
    def coupler(self) -> LegacyCoupler | SAPCoupler | IPCCoupler:
        return self.scene.sim.coupler

    @cached_property
    def _has_ipc_coupler(self) -> bool:
        """Whether the scene uses an IPC coupler (requires full-scene reset)."""
        return isinstance(self.env_options.coupler_options, gs.options.IPCCouplerOptions)

    @property
    def rigid_solver(self) -> RigidSolver:
        return self.scene.rigid_solver

    @property
    def segmentation_idx_dict(self) -> dict[str, int]:
        return self.scene.segmentation_idx_dict

    @property
    def device(self) -> torch.device:
        """The torch device the environment's tensors live on."""
        return gs.device

    @property
    def config(self) -> EdenConfig:
        """Reconstruct the :class:`EdenConfig` describing this environment."""
        return EdenConfig(
            env_options=self.env_options,
            scene_options=self.scene_options,
            observation_options=self.observation_options,
            event_options=self.event_options,
            action_options=self.action_options,
            metric_options=self.metric_options,
            recorder_options=self.recorder_options,
            cameras_options=self.cameras_options,
            sensors_options=self.sensors_options,
            renderer_options=self.renderer_options,
        )

    @property
    def gravity(self) -> torch.Tensor:
        """The gravity vector applied by the rigid solver."""
        return self.rigid_solver.get_gravity()

    def set_global_sol_params(self, sol_params) -> None:
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: tuple[float] | list[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        self.rigid_solver.set_global_sol_params(sol_params)

    def set_sol_params(
        self,
        sol_params,
        geoms_idx=None,
        envs_idx=None,
        *,
        joints_idx=None,
        eqs_idx=None,
    ) -> None:
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: tuple[float] | list[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        geoms_idx: array_like, optional
            Indices of the geoms to set parameters for. If None, all geoms are used. Defaults to None.
        envs_idx: array_like, optional
            Indices of the environments to set parameters for. If None, all environments are used. Defaults to None.
        joints_idx: array_like, optional
            Indices of the joints to set parameters for. If None, no joints are targeted. Defaults to None.
        eqs_idx: array_like, optional
            Indices of the equality constraints to set parameters for. If None, none are targeted. Defaults to None.
        """
        self.rigid_solver.set_sol_params(
            sol_params,
            geoms_idx=geoms_idx,
            envs_idx=envs_idx,
            joints_idx=joints_idx,
            eqs_idx=eqs_idx,
        )

    def get_entity(self, name: str) -> "Entity":
        """Return the entity registered under ``name``.

        Parameters
        ----------
        name : str
            The entity's name.

        Returns
        -------
        Entity
            The matching entity.
        """
        return self.entities[name]

    def get_entities(self, names: str | list[str]) -> list["Entity"]:
        """
        Get entities by name(s) with regex pattern matching support.

        This method uses `re.match()` which matches patterns from the beginning
        of the entity name. To match entities containing a pattern anywhere,
        prefix with `.*` (e.g., `".*arm"` to match entities ending with "arm").

        Parameters
        ----------
        names : str | list[str]
            A single regex pattern string or a list of regex pattern strings
            to match against entity names. Patterns are matched from the
            beginning of entity names.

        Returns
        -------
        list[Entity]
            A list of entities whose names match any of the provided regex patterns.

        Examples
        --------
        >>> # Match all entities starting with "robot"
        >>> env.get_entities("robot.*")
        >>> # Match entities ending with a number (won't match "robot" without prefix)
        >>> env.get_entities(".*_[0-9]+")
        >>> # Match multiple patterns
        >>> env.get_entities(["robot_[0-9]+", "box.*"])
        >>> # Exact match
        >>> env.get_entities("^robot$")  # or simply "robot$"
        """
        # Normalize input to a list
        patterns = [names] if isinstance(names, str) else names

        matched_entities = []
        for entity_name in self.entities.keys():
            for pattern in patterns:
                if re.match(pattern, entity_name):
                    matched_entities.append(self.entities[entity_name])
                    break  # Avoid adding the same entity multiple times

        return matched_entities

    def get_camera(self, name: str) -> "Camera":
        """Return the camera registered under ``name``.

        Parameters
        ----------
        name : str
            The camera's name.

        Returns
        -------
        Camera
            The matching camera.
        """
        return self.cameras[name]

    def reset(
        self,
        envs_idx: slice | torch.Tensor | None = None,
        *,
        seed: int | None = None,
    ) -> VecEnvReset:
        """Reset the given environments and return their fresh observations.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor, optional
            The environments to reset. If None, all environments are reset.
        seed : int, optional
            If given, seed the RNGs before resetting.

        Returns
        -------
        VecEnvReset
            A ``(observations, extras)`` tuple, where ``extras`` carries reset
            logging info under the ``"log"`` key.
        """
        if envs_idx is None:
            envs_idx = slice(None)
        if seed is not None:
            self.seed(seed)

        self.recorder_manager._on_reset_started(envs_idx)
        self.reset_info = self._reset_idx(envs_idx)
        self.recorder_manager._on_reset_finished(envs_idx=envs_idx)
        self.clear_debug_objects()

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, {"log": self.reset_info}

    def step(self, action: torch.Tensor | dict[str, torch.Tensor] | None = None) -> VecEnvStep:
        """Advance the environment by one control step.

        Applies ``action`` through the action manager, runs ``decimation`` physics
        sub-steps, and computes observations. ``EnvBase`` itself has no reward or
        termination; subclasses (:class:`RLEnvBase`) extend this.

        Parameters
        ----------
        action : torch.Tensor | dict[str, torch.Tensor], optional
            The action to apply. If None, no action is applied (the sim still steps).

        Returns
        -------
        VecEnvStep
            The post-step observations and associated extras.
        """
        if action is not None:
            self.action_manager.compute(action)

        self.recorder_manager._on_step_started()

        for i in range(self.env_options.decimation):
            self._sim_step_counter += 1
            if action is not None:
                self.action_manager.apply_actions()
            self._scene_step_with_sensor_gating(is_last_substep=(i == self.env_options.decimation - 1))

        self.episode_length_buf += 1

        # TODO: call camera.render() for all cameras?

        if self.metric_manager is not None:
            self.metric_manager.compute()

        if _INTERVAL in self.event_manager.available_modes:
            self.event_manager.compute(mode=_INTERVAL, dt=self.dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager._on_step_finished()

        # NOTE: to match the RLEnvBase.step() return type for consistency
        return (self.obs_buf, None, None, None, {"log": self.reset_info})

    def _scene_step_with_sensor_gating(self, is_last_substep: bool) -> None:
        """Call ``scene.step()``, optionally gating sensor updates on intermediate sub-steps.

        Genesis ``SensorManager.step`` is skipped on intermediate sub-steps when
        ``env_options.update_sensors_every_substep`` is False (the default).

        Intermediate sub-step sensor readings are overwritten by the last
        sub-step's update before anything reads them — ``observation_manager
        .compute()`` runs once at the end of ``step()``. Skipping the
        redundant per-sub-step kernel work (visual-AABB updates, raycast,
        cache population) yields a ~2.3× speedup for perceptive_mimic at
        decimation=4. ``_sensors_by_type`` is the dispatch dict the manager
        loops over; swapping it to empty short-circuits the expensive per-
        sensor-class branch while leaving timeline ring rotations intact
        (rings live in a separate ``_measured_timeline_ring`` dict).

        Safety: when any sensor declares Genesis-level ``delay`` or
        ``history_length`` (so the SensorManager allocates timeline / return
        rings), ring rotations need fresh slot-0 writes every sub-step or
        the delay/history reads return stale, duplicated samples. We detect
        ring presence and transparently fall back to the per-sub-step path
        in that case, so the optimisation stays safe to leave on by default.
        """
        if is_last_substep or self.env_options.update_sensors_every_substep:
            self.scene.step()
            return

        sim = self.scene._sim
        if not hasattr(sim, "_sensor_manager"):
            self.scene.step()
            return

        sm = sim._sensor_manager
        # Auto-disable when any sensor explicitly configures Genesis-level
        # ``delay > 0`` or ``history_length > 0`` — those are the cases where
        # the post-process / delay pipeline genuinely consumes ring slots
        # beyond slot 0, so the slots must advance sim-rate with fresh
        # writes. Genesis allocates N≥2 rings even for default sensors, so
        # ring presence alone over-triggers; checking user-set sensor options
        # is the precise signal. Cache the decision so the per-step check is
        # a single attribute read after the first call.
        cached = getattr(self, "_sensor_substep_skip_safe", None)
        if cached is None:
            cached = self._compute_sensor_substep_skip_safe()
            self._sensor_substep_skip_safe = cached
        if not cached:
            self.scene.step()
            return

        saved = sm._sensors_by_type
        sm._sensors_by_type = {}
        try:
            self.scene.step()
        finally:
            sm._sensors_by_type = saved

    def _compute_sensor_substep_skip_safe(self) -> bool:
        """Return True iff every sensor has Genesis-level ``delay == 0`` and ``history_length == 0``.

        Sensors with non-default delay or history consume ring slots beyond slot 0
        in their post-process / delay pipeline, and those slots only stay coherent if
        fresh writes happen on every sub-step. Anything else is safe to skip.
        """
        for sensor in self.sensors.values():
            inner = getattr(getattr(sensor, "_options", None), "sensor", None)
            if inner is None:
                continue
            if getattr(inner, "delay", 0.0) or getattr(inner, "history_length", 0):
                return False
        return True

    def _reset_idx(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, Any]:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return dict()
        # Genesis MPM coupler kernels don't accept slice; convert to index tensor.
        if isinstance(envs_idx, slice):
            _scene_envs_idx = self._all_indices[envs_idx]
        else:
            _scene_envs_idx = envs_idx

        # IPC coupler only supports full-scene reset (envs_idx=None).
        # TODO: implement per-env IPC reset via libuipc StateAccessor
        _reset_envs_idx = None if self._has_ipc_coupler else _scene_envs_idx
        self.scene.reset(state=self.reset_state, envs_idx=_reset_envs_idx)

        if _RESET in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.env_options.decimation
            self.event_manager.compute(mode=_RESET, envs_idx=envs_idx, global_env_step_count=env_step_count)

        reset_info = dict()
        reset_info.update(self.observation_manager.reset(envs_idx))
        if self.metric_manager is not None:
            reset_info.update(self.metric_manager.reset(envs_idx))
        reset_info.update(self.action_manager.reset(envs_idx))
        reset_info.update(self.event_manager.reset(envs_idx))
        self.episode_length_buf[envs_idx] = 0

        return reset_info


class RLEnvBase(EnvBase):
    """
    Base class for all RL environments.

    Parameters
    ----------
    env_options: EnvOptions
        Environment options.
    observation_options: ObservationManagerOptions
        Observation manager options.
    reward_options: RewardManagerOptions
        Reward manager options.
    termination_options: TerminationManagerOptions
        Termination manager options.
    event_options: EventManagerOptions
        Event manager options.
    curriculum_options: CurriculumManagerOptions
        Curriculum manager options.
    command_options: CommandManagerOptions
        Command manager options.
    action_options: ActionManagerOptions
        Action manager options.
    renderer_options: RendererOptions
        Renderer options.
    show_viewer: bool
        Whether to show the viewer.
    eval_mode: bool
        Whether to run in eval mode.
    """

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        reward_options: RewardManagerOptions | None = None,
        termination_options: TerminationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        curriculum_options: CurriculumManagerOptions | None = None,
        command_options: CommandManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            env_options=env_options,
            scene_options=scene_options,
            observation_options=observation_options,
            event_options=event_options,
            action_options=action_options,
            metric_options=metric_options,
            recorder_options=recorder_options,
            cameras_options=cameras_options,
            sensors_options=sensors_options,
            renderer_options=renderer_options,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            **kwargs,
        )
        # Handling of default arguments
        self.reward_options = reward_options or RewardManagerOptions()
        self.termination_options = termination_options or TerminationManagerOptions()
        self.curriculum_options = curriculum_options or CurriculumManagerOptions()
        self.command_options = command_options or CommandManagerOptions()

        self.max_episode_length_s = self.env_options.episode_length_s
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        self.common_step_counter = 0

        # Lazily allocated zero-initialized cache for ``extras['final_observations']``.
        # See ``EnvOptions.record_final_observations`` and ``RLEnvBase.step()``.
        self._final_obs_cache: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    @classmethod
    def from_config(cls, cfg: EdenRLConfig, **kwargs) -> "RLEnvBase":
        """
        Create an environment from a configuration object.

        Parameters
        ----------
        cfg: EdenRLConfig
            The configuration object.
        **kwargs: dict
            Additional keyword arguments.
            show_viewer: bool
                Whether to show the viewer.
            eval_mode: bool
                Whether to run in eval mode.
        """
        env = cls(
            env_options=cfg.env_options,
            scene_options=cfg.scene_options,
            observation_options=cfg.observation_options,
            reward_options=cfg.reward_options,
            termination_options=cfg.termination_options,
            event_options=cfg.event_options,
            curriculum_options=cfg.curriculum_options,
            command_options=cfg.command_options,
            action_options=cfg.action_options,
            metric_options=cfg.metric_options,
            recorder_options=cfg.recorder_options,
            cameras_options=cfg.cameras_options,
            sensors_options=cfg.sensors_options,
            renderer_options=cfg.renderer_options,
            show_viewer=kwargs.get("show_viewer", False),
            eval_mode=kwargs.get("eval_mode", False),
        )

        return env

    @gs.assert_built
    def _load_managers(self):
        from eden.managers import (
            CommandManager,
            CurriculumManager,
            RewardManager,
            TerminationManager,
        )

        self.command_manager = CommandManager(self, self.command_options)
        if not self.command_manager._terms:
            self.command_manager = None
        # NOTE: reward_manager must be built before curriculum_manager so that
        # curriculum terms (e.g. StageRewardWeightCurriculum) can resolve
        # reward term handles during their `__init__`.
        self.reward_manager = RewardManager(self, self.reward_options)
        self.curriculum_manager = CurriculumManager(self, self.curriculum_options)
        if not self.curriculum_manager._terms:
            self.curriculum_manager = None
        self.termination_manager = TerminationManager(self, self.termination_options)

        # NOTE: will load the startup events if any.
        super()._load_managers()

    @gs.assert_built
    def summary(self) -> str:
        """Return a human-readable summary, including RL managers (reward, termination, etc.)."""
        msg = super().summary()
        if self.command_manager is not None:
            msg += self.command_manager.summary()
        if self.curriculum_manager is not None:
            msg += self.curriculum_manager.summary()
        msg += self.reward_manager.summary()
        msg += self.termination_manager.summary()
        return msg

    @property
    def config(self):
        """Reconstruct the :class:`EdenRLConfig` describing this environment."""
        from eden.utils.configs import EdenRLConfig

        return EdenRLConfig(
            env_options=self.env_options,
            scene_options=self.scene_options,
            observation_options=self.observation_options,
            reward_options=self.reward_options,
            termination_options=self.termination_options,
            curriculum_options=self.curriculum_options,
            command_options=self.command_options,
            event_options=self.event_options,
            action_options=self.action_options,
            metric_options=self.metric_options,
            recorder_options=self.recorder_options,
            cameras_options=self.cameras_options,
            sensors_options=self.sensors_options,
            renderer_options=self.renderer_options,
        )

    @property
    def unwrapped(self) -> "RLEnvBase":
        """The underlying environment (returns ``self``; provided for wrapper compatibility)."""
        return self

    def step(
        self,
        action: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> VecEnvStep:
        """Advance the RL environment by one control step.

        Applies ``action``, runs ``decimation`` physics sub-steps, then computes
        metrics, terminations, and rewards. Done environments are auto-reset
        **after** reward/termination but **before** observation, so the returned
        observations reflect the fresh post-reset state for those environments.

        Parameters
        ----------
        action : torch.Tensor | dict[str, torch.Tensor], optional
            The action to apply. If None, no action is applied (the sim still steps).

        Returns
        -------
        VecEnvStep
            A ``(observations, rewards, dones, extras)`` tuple.
        """
        if action is not None:
            self.action_manager.compute(action)

        self.recorder_manager._on_step_started()

        for i in range(self.env_options.decimation):
            self._sim_step_counter += 1
            if action is not None:
                self.action_manager.apply_actions()
            self._scene_step_with_sensor_gating(is_last_substep=(i == self.env_options.decimation - 1))

        # Update env counters.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Refresh command terms' cached robot state on the just-computed
        # post-physics state, so metric/termination/reward grade the CURRENT
        # step's robot state. Matches IsaacLab/mjlab, which read live robot data
        # at reward time. Without this, MotionCommand's robot cache — otherwise
        # refreshed only in command_manager.compute() (which runs *after* reward
        # below) — is one full control step stale when the reward is computed,
        # inflating every tracking-error term and the termination checks.
        if self.command_manager is not None:
            for _cmd_term in self.command_manager.terms.values():
                _refresh = getattr(_cmd_term, "_update_robot_state_cache", None)
                if _refresh is not None:
                    _refresh()
                # ALSO rebuild the anchor-relative body buffer using the
                # just-refreshed live robot anchor, so the reward/termination
                # sees a relative reference anchored to the robot's CURRENT
                # pose (S_N) — not the pose at end of the previous step's
                # compute() (S_{N-1}). Otherwise body_pos_relative_w /
                # body_quat_relative_w stay one full control step stale on
                # the anchor side, while robot_body_*_w is current; the
                # off-by-one introduces base-velocity-correlated variance
                # in the velocity/orientation rewards (high value loss).
                # Mimic's framework reward computes this on-the-fly inside
                # the reward call, which is why mimic_beyondmimic has much
                # lower value loss at similar reward.
                _rel = getattr(_cmd_term, "update_relative_body_poses", None)
                if _rel is not None:
                    _rel()

        # TODO: call camera.render() for all cameras?

        if self.metric_manager is not None:
            self.metric_manager.compute()

        # Check for termination.
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_timeouts = self.termination_manager.timeouts

        self.reward_buf = self.reward_manager.compute(dt=self.dt)
        # Gate on is_recording so `.any()` doesn't force a GPU→CPU sync every step during training.
        if self.recorder_manager.is_recording:
            if self.metric_manager is not None and hasattr(self.metric_manager, "success_buf") and self.reset_buf.any():
                self.recorder_manager.set_episode_success(
                    self.reset_buf, self.metric_manager.success_buf[self.reset_buf]
                )
        self.recorder_manager._on_step_finished(self.reset_buf)

        # Snapshot pre-reset observations for done envs (opt-in via
        # ``EnvOptions.record_final_observations``). Must run before
        # ``_reset_idx`` since ``scene.reset`` wipes the done envs' state.
        if self.env_options.record_final_observations:
            final_obs = self._record_final_observations()
        else:
            final_obs = None

        # Always call _reset_idx — bool-mask ops are no-ops for all-False
        self.reset_info = self._reset_idx(self.reset_buf)

        # Record new-episode data after reset, such as initial state snapshots.
        self.recorder_manager._on_reset_finished(envs_idx=self.reset_buf)

        if self.curriculum_manager is not None:
            self.curriculum_manager.compute()

        if self.command_manager is not None:
            self.command_manager.compute(dt=self.dt)

        if _INTERVAL in self.event_manager.available_modes:
            self.event_manager.compute(mode=_INTERVAL, dt=self.dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        if not self.training and self.command_manager is not None:
            self.command_manager.draw_vis()

        extras: dict[str, Any] = {"log": self.reset_info}
        if final_obs is not None:
            extras["final_observations"] = final_obs

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_timeouts,
            extras,
        )

    def _compute_final_observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the pre-reset observation dict for snapshotting (hook).

        Default: full-batch ``observation_manager.compute(update_history=False)``.
        ``update_history=False`` is critical — the post-reset compute at the
        end of ``step()`` advances the history buffer once with
        ``update_history=True``; advancing here too would skip a frame on
        non-done envs. For history-bearing groups the manager still computes
        the fresh post-physics frame and exposes it in the most-recent slot of
        the returned tensor (via ``CircularBuffer.peek_buffer``), without
        mutating the underlying buffer. Subclass to swap in motion-frame
        indices or skip groups whose terms are unsafe to evaluate against the
        post-physics state.
        """
        return self.observation_manager.compute(update_history=False)

    def _record_final_observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Snapshot ``_compute_final_observations`` into the cached ``_final_obs_cache``.

        Only the rows for ``self.reset_buf`` are kept.

        Allocates the cache lazily; rebuilds the cache when the source
        tensor's shape or dtype changes (e.g. when the env is rebuilt at a
        different ``num_envs``). The slice-copy from ``snapshot[reset_buf]``
        is required, not just an allocation optimization — for
        history-bearing groups ``compute(update_history=False)`` returns a
        view into ``CircularBuffer.buffer`` that the post-reset
        ``compute(update_history=True)`` mutates.
        """
        snapshot = self._compute_final_observations()
        return self._copy_into_final_obs_cache(snapshot, self._final_obs_cache, self.reset_buf)

    @classmethod
    def _copy_into_final_obs_cache(
        cls,
        source: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        cache: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Sync-free: bool-mask advanced indexing (``cached[mask] = value[mask]``)
        # internally calls ``nonzero()`` on CUDA to size the gather/scatter and
        # forces a GPU→CPU sync — exactly the cost the design avoids elsewhere
        # (see ``EnvOptions.record_final_observations``). ``torch.where`` over a
        # broadcastable bool mask has fully static shapes and stays on-device.
        for key, value in source.items():
            if isinstance(value, dict):
                sub_cache = cache.get(key)
                if not isinstance(sub_cache, dict):
                    sub_cache = {}
                    cache[key] = sub_cache
                cls._copy_into_final_obs_cache(value, sub_cache, mask)
                continue

            cached = cache.get(key)
            if (
                not isinstance(cached, torch.Tensor)
                or cached.shape != value.shape
                or cached.dtype != value.dtype
                or cached.device != value.device
            ):
                cached = torch.zeros_like(value)
                cache[key] = cached
            mask_view = mask.view(mask.shape[0], *([1] * (value.ndim - 1)))
            zero = torch.zeros((), dtype=value.dtype, device=value.device)
            torch.where(mask_view, value, zero, out=cached)
        return cache

    def _reset_idx(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, Any]:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return dict()
        # Genesis MPM coupler kernels don't accept slice; convert to index tensor.
        if isinstance(envs_idx, slice):
            _scene_envs_idx = self._all_indices[envs_idx]
        else:
            _scene_envs_idx = envs_idx

        # IPC coupler only supports full-scene reset (envs_idx=None).
        # TODO: implement per-env IPC reset via libuipc StateAccessor
        _reset_envs_idx = None if self._has_ipc_coupler else _scene_envs_idx
        self.scene.reset(state=self.reset_state, envs_idx=_reset_envs_idx)

        if _RESET in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.env_options.decimation
            self.event_manager.compute(mode=_RESET, envs_idx=envs_idx, global_env_step_count=env_step_count)

        reset_info = dict()
        reset_info.update(self.observation_manager.reset(envs_idx))
        if self.metric_manager is not None:
            reset_info.update(self.metric_manager.reset(envs_idx))
        reset_info.update(self.action_manager.reset(envs_idx))
        reset_info.update(self.event_manager.reset(envs_idx))
        if self.command_manager is not None:
            reset_info.update(self.command_manager.reset(envs_idx))
        if self.curriculum_manager is not None:
            reset_info.update(self.curriculum_manager.reset(envs_idx))
        reset_info.update(self.reward_manager.reset(envs_idx))
        reset_info.update(self.termination_manager.reset(envs_idx))
        self.episode_length_buf[envs_idx] = 0

        return reset_info
