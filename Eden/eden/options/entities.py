"""Entity configuration options (rigid/primitive/grouped) and metadata.

:class:`EntityOptions` describes an entity to add to the scene (file/morph, default root
pose, material). ``PrimitiveOptions`` is an abstract base — use its concrete subclasses
(``BoxOptions`` / ``SphereOptions`` / ``CylinderOptions`` / ``PlaneOptions``) for rigid
primitives. For deformables, position/orientation are baked in via ``default_root_pos``
and ``default_root_quat`` (the runtime ``ParticleEntity.set_pos`` / ``set_quat`` raise).
:class:`GroupedEntityOptions` supports heterogeneous batching of entities that share a
joint structure.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

import genesis as gs
import numpy as np
from genesis.options.morphs import Morph
from genesis.typing import FArrayType, PositiveVec2FType, UnitVec3FType, UnitVec4FType, Vec2FType, Vec3FType
from pydantic import Field

from eden.options.actuators import ActuatorSpecOptions
from eden.options.materials import (
    FEMClothMaterialOptions,
    FEMElasticMaterialOptions,
    FEMMuscleMaterialOptions,
    KinematicMaterialOptions,
    MaterialLike,
    MPMElasticMaterialOptions,
    MPMElastoPlasticMaterialOptions,
    MPMLiquidMaterialOptions,
    PBDClothMaterialOptions,
    PBDElasticMaterialOptions,
    PBDLiquidMaterialOptions,
    PBDParticleMaterialOptions,
    SPHLiquidMaterialOptions,
)
from eden.options.options import ConfigurableOptions
from eden.options.surfaces import SurfaceLike
from eden.types import MorphLike, VisMode
from eden.utils.terrains import FlatTerrain

_NON_RIGID_MATERIAL_TYPES = (
    MPMElasticMaterialOptions,
    MPMElastoPlasticMaterialOptions,
    MPMLiquidMaterialOptions,
    SPHLiquidMaterialOptions,
    PBDLiquidMaterialOptions,
    PBDClothMaterialOptions,
    PBDElasticMaterialOptions,
    PBDParticleMaterialOptions,
    FEMElasticMaterialOptions,
    FEMClothMaterialOptions,
    FEMMuscleMaterialOptions,
)

_DOF_FIELD_NAMES = (
    "dofs_name",
    "default_dofs_pos",
    "default_dofs_vel",
    "default_dofs_stiffness",
    "default_dofs_damping",
    "default_dofs_armature",
    "default_dofs_kp",
    "default_dofs_kd",
    "default_dofs_pos_limits",
    "soft_dofs_pos_limits",
    "default_dofs_force_limits",
)


def _validate_no_dofs_for_non_rigid_material(opts) -> None:
    """Raise ValueError if DOF fields are set on a non-rigid (particle/FEM) entity."""
    if not isinstance(opts.material, _NON_RIGID_MATERIAL_TYPES):
        return
    set_fields = [name for name in _DOF_FIELD_NAMES if getattr(opts, name, None)]
    if set_fields:
        raise ValueError(
            f"DOF fields {set_fields} cannot be used with {type(opts.material).__name__}. "
            f"Non-rigid entities have no DOFs."
        )


class MetadataOptions(ConfigurableOptions):
    """
    Metadata options for Entity, used to store additional metadata about the entity.

    Parameters
    ----------
    <key>: Any
        The value of the metadata.
    """


class BaseEntityOptions(ConfigurableOptions):
    """
    Shared options for entities.

    Parameters
    ----------
    default_root_pos : array-like[float, float, float]
        Default root position.
    default_root_quat : array-like[float, float, float, float]
        Default root orientation as quaternion in (w, x, y, z).
    is_fixed_base : bool
        Whether the entity base is fixed.
    is_articulated : bool
        Whether the entity has articulation.
    is_actuated : bool
        Whether the entity is actuated.
    material : MaterialLike | None
        Material used for the entity.
    surface : SurfaceOptions | Surface | None
        Surface appearance used for the entity. Prefer an Eden :class:`SurfaceOptions`
        (e.g. ``PlasticSurfaceOptions(color=(1.0, 0.0, 0.0))``) in configs constructed
        before ``gs.init()``; a raw ``gs.surfaces.*`` instance is also accepted.
    color : array-like[float] | None
        Convenience shortcut to color the entity. When set (and ``surface`` is not),
        a plastic surface with this color (RGB or RGBA) is created at build time.
    visualize_contact : bool
        Whether to visualize contacts involving this entity.
    vis_mode : VisMode
        Visualization mode for the entity.
    visualization : bool
        Whether to visualize the entity.
    collision : bool
        Whether this entity participates in collision checking.
    contype : int
        32-bit collision-filter bitmask. Two geoms can collide only if the ``contype`` of one
        and the ``conaffinity`` of the other share a set bit. Defaults to 0xFFFF.
    conaffinity : int
        32-bit collision-filter bitmask. See ``contype``. Defaults to 0xFFFF.
    collision_link_patterns : list of str, optional
        Regex/glob patterns matched against link names. When set, only geoms on
        matching links keep collision; geoms on all other links are disabled
        (``contype = conaffinity = 0``). Use to restrict an entity's collisions
        to specific parts — e.g. a parallel-jaw gripper's fingertip pads
        (``[".*pad"]``) so objects are grasped at the pads instead of wedging
        against the inner linkage. Applied before the scene is built (Genesis
        bakes the collision-pair list at build time). ``None`` keeps the morph's
        default per-geom collision.
    collision_friction : float, optional
        Friction coefficient applied to the geoms kept by
        ``collision_link_patterns`` (e.g. high pad friction for stable grasps).
    batch_fixed_verts : bool
        Whether fixed-geometry vertices are batched for per-env transforms.
    is_support_enabled : bool
        Whether support-point computation is enabled.
    dofs_name : list[str]
        Controllable DOF names.
    dofs_spec : ClassVar[dict[str, type[ActuatorSpecOptions]]]
        Class-level default actuator specification map for DOFs.
    default_dofs_pos : dict[str, float]
        Default DOF positions.
    default_dofs_vel : dict[str, float]
        Default DOF velocities.
    default_dofs_kp : dict[str, float]
        Default DOF position gains.
    default_dofs_kd : dict[str, float]
        Default DOF velocity gains.
    default_dofs_stiffness : dict[str, float]
        Default DOF stiffness values.
    default_dofs_damping : dict[str, float]
        Default DOF damping values.
    default_dofs_armature : dict[str, float]
        Default DOF armature values.
    default_dofs_pos_limits : dict[str, array-like[float, float]]
        Hard DOF position limits.
    soft_dofs_pos_limits : dict[str, array-like[float, float]]
        Soft DOF position limits.
    default_dofs_force_limits : dict[str, float]
        Default DOF force limits.
    """

    material: MaterialLike | None = None
    surface: SurfaceLike | None = None
    color: FArrayType | None = None
    visualize_contact: bool = False
    vis_mode: VisMode = "visual"
    metadata: ClassVar[MetadataOptions | None] = None
    visualization: bool = True
    collision: bool = True
    contype: int = 0xFFFF
    conaffinity: int = 0xFFFF
    collision_link_patterns: list[str] | None = None
    collision_friction: float | None = None
    batch_fixed_verts: bool = False
    is_support_enabled: bool = False
    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)
    default_root_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    is_fixed_base: bool = False
    is_articulated: bool = False
    is_actuated: bool = False
    dofs_name: list[str] = []
    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = {}
    default_dofs_pos: dict[str, float] = {}
    default_dofs_vel: dict[str, float] = {}
    default_dofs_kp: dict[str, float] = {}
    default_dofs_kd: dict[str, float] = {}
    default_dofs_stiffness: dict[str, float] = {}
    default_dofs_damping: dict[str, float] = {}
    default_dofs_armature: dict[str, float] = {}
    default_dofs_pos_limits: dict[str, Vec2FType] = {}
    soft_dofs_pos_limits: dict[str, Vec2FType] = {}
    default_dofs_force_limits: dict[str, float] = {}

    def _set_dofs_unactuated(self):
        """Set DOF fields to unactuated defaults."""
        # dofs_spec is a ClassVar — use object.__setattr__ to shadow it on the
        # instance without mutating the class (which would corrupt other instances).
        object.__setattr__(self, "dofs_spec", {})
        self.dofs_name = []
        self.default_dofs_pos = {}
        self.default_dofs_vel = {}
        self.default_dofs_stiffness = {}
        self.default_dofs_damping = {}
        self.default_dofs_armature = {}
        self.default_dofs_kp = {}
        self.default_dofs_kd = {}
        self.default_dofs_pos_limits = {}
        self.soft_dofs_pos_limits = {}
        self.default_dofs_force_limits = {}
        self.is_articulated = False
        self.is_actuated = False


class EntityOptions(BaseEntityOptions):
    """
    Entity options for Eden, wrapping Genesis' Entity related options.

    Parameters
    ----------
    up: array-like[int, int, int]
        The desired up direction of the object in the object coordinate system (use `gs view` to visualize the object coordinate system).
        This direction will be used to align the object's up direction to the desired direction when loading the object.
    front: array-like[int, int, int]
        The desired front direction of the object in the object coordinate system (use `gs view` to visualize the object coordinate system).
        This direction will be used to align the object's front direction to the desired direction when loading the object.
    file: str
        Path to the entity file.
    scale: float
        Scale of the entity.
    collision: bool
        Whether the entity needs to be considered for collision checking. Defaults to True.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Default to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 2.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh will be decomposed into multiple convex components if the convex hull is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_object_error_threshold : bool, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : bool, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_threshold : float, optional:
        The threshold for the CoACD algorithm. Defaults to 0.05.
    coacd_preprocess_resolution : int, optional:
        The resolution for the CoACD algorithm. Defaults to 100.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to [].
    default_armature : float, optional
        The default armature to be used for the entity. Defaults to 0.1.

    is_fixed_base: bool
        Whether the entity is fixed in the environment.
    is_articulated: bool
        Whether the entity is articulated.
    is_actuated: bool
        Whether the entity is actuated.
    is_support_enabled: bool
        Whether the entity is support enabled.

    support_sample_method: Literal["uniform", "grid"]
        The method to sample the support points. Defaults to "uniform".
    support_minimum_clearance: float
        The minimum clearance to be considered for support points. Defaults to 0.05.
    support_shrink: float
        The shrink factor to be applied to the support points. Defaults to 0.1.
    support_num_sample_points: int
        The number of support points to sample. Defaults to 300.
    support_grid_size: float
        The size of the grid to sample the support points when `support_sample_method` is "grid". Defaults to 0.01.
    support_links_name: list[str]
        Exact link names (URDF link names / MJCF body names) whose geometry should
        be considered when extracting support surfaces. Empty (default) means all
        links contribute, preserving prior behavior. Use this on multi-link assets
        (e.g. the Riverway franka station) to restrict placement targets to a
        specific surface like ``["station_table_top"]`` so events don't spawn
        objects onto auxiliary cart / gantry tops.

    default_root_pos: array-like[float, float, float]
        The default position of the loaded entity's root link. Will not take effect if the entity is attached to another entity.
    default_root_quat: array-like[float, float, float, float]
        The default orientation of the loaded entity's root link (w, x, y, z). Will not take effect if the entity is attached to another entity.
    dofs_name: list[str]
        The names of the DOFs to be treated as controllable DOFs.
    dofs_spec : ClassVar[dict[str, type[ActuatorSpecOptions]]]
        Class-level default actuator specification map for entity DOFs.
    default_dofs_pos: dict[str, float]
        The default positions of the DOFs of the entity.
    default_dofs_vel: dict[str, float]
        The default velocities of the DOFs of the entity.
    default_dofs_kp: dict[str, float]
        The default control position gains of the DOFs of the entity.
    default_dofs_kd: dict[str, float]
        The default control velocity gains of the DOFs of the entity.
    default_dofs_stiffness: dict[str, float]
        The default stiffnesses of the DOFs of the entity. If provided, the value will be used over the stiffness in dofs_spec.
    default_dofs_damping: dict[str, float]
        The default dampings of the DOFs of the entity. If provided, the value will be used over the damping in dofs_spec.
    default_dofs_armature: dict[str, float]
        The default armatures of the DOFs of the entity. If provided, the value will be used over the armature in dofs_spec.
    default_dofs_pos_limits: dict[str, array-like[float, float]]
        The position limits of the DOFs of the entity.
    soft_dofs_pos_limits: dict[str, array-like[float, float]]
        The position limits of the DOFs of the entity that are treated as soft DOFs.
    """

    # NOTE: default up (+z) and front direction (+x) in object coordinate system
    # NOTE: this is used to set the default up and front direction for the entity on loading
    # NOTE: init_quat will be the offset from this default direction
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    # NOTE: easier access to morph options with default values
    file: str = ""
    registry: str = "Kashu7100"
    dataset: str = "eden_assets"
    local_dir: str | None = None
    scale: float = 1.0
    decimate: bool = True
    decimate_face_num: int = 1000
    decimate_aggressiveness: int = 3
    convexify: bool = True
    decompose_object_error_threshold: float = 0.15
    decompose_robot_error_threshold: float = float("inf")  # Any objects with joint
    coacd_threshold: float = 0.05
    coacd_preprocess_resolution: int = 100
    links_to_keep: list[str] = []  # only used for URDF
    default_armature: float = 0.1

    if TYPE_CHECKING:
        morph: MorphLike

    support_sample_method: Literal["uniform", "grid"] = "uniform"  # "uniform" or "grid"
    support_minimum_clearance: float = 0.05
    support_shrink: float = 0.1
    support_num_sample_points: int = 300
    support_grid_size: float = 0.01
    support_links_name: list[str] = []  # empty = all links contribute (legacy behavior)

    forward_vec: Vec3FType = (1.0, 0.0, 0.0)

    def model_post_init(self, context):
        super().model_post_init(context)

        # Set forward_vec from front direction
        self.forward_vec = self.front

        # Validate up and front are not parallel
        up_norm = np.linalg.norm(self.up)
        front_norm = np.linalg.norm(self.front)
        if up_norm > 0 and front_norm > 0:
            up_unit = np.array(self.up) / up_norm
            front_unit = np.array(self.front) / front_norm
            dot = np.abs(np.dot(up_unit, front_unit))
            if dot > 0.99:  # Nearly parallel
                raise ValueError("Up and front directions must not be parallel")

        if self.material is not None:
            _validate_no_dofs_for_non_rigid_material(self)


class PrimitiveOptions(BaseEntityOptions):
    """
    Base options for primitive shapes.

    Parameters
    ----------
    default_root_pos : array-like[float, float, float]
        Default root position.
    default_root_quat : array-like[float, float, float, float]
        Default root orientation as quaternion in (w, x, y, z).
    fixed: bool
        Whether the entity is fixed.
    is_articulated : bool
        Whether the entity has articulation.
    is_actuated : bool
        Whether the entity is actuated.
    material : MaterialLike | None
        Material used for the entity.
    surface : SurfaceOptions | Surface | None
        Surface appearance used for the entity. Prefer an Eden :class:`SurfaceOptions`
        in configs constructed before ``gs.init()``.
    color : array-like[float] | None
        Convenience shortcut to color the entity (RGB or RGBA) without a full surface.
    visualize_contact : bool
        Whether to visualize contacts involving this entity.
    vis_mode : VisMode
        Visualization mode for the entity.
    collision : bool
        Whether this entity participates in collision checking.
    batch_fixed_verts : bool
        Whether fixed-geometry vertices are batched for per-env transforms.
    is_support_enabled : bool
        Whether support-point computation is enabled.
    """

    fixed: bool = False
    if TYPE_CHECKING:
        morph: MorphLike

    def _collect_morph_kwargs(self, **kwargs) -> dict[str, object]:
        """Collect kwargs forwarded to Genesis morph constructors."""
        kwargs.update(
            pos=self.default_root_pos,
            quat=self.default_root_quat,
            fixed=self.is_fixed_base,
            batch_fixed_verts=self.batch_fixed_verts,
            collision=self.collision,
            contype=self.contype,
            conaffinity=self.conaffinity,
            visualization=self.visualization,
        )
        if hasattr(self, "morph"):
            if isinstance(self.morph, dict):
                kwargs.update(**self.morph)
            elif isinstance(self.morph, Morph):
                kwargs.update(**self.morph.model_dump())
        return kwargs

    def model_post_init(self, context):
        super().model_post_init(context)
        if type(self) is PrimitiveOptions:
            raise ValueError(
                "PrimitiveOptions is an abstract base. Use one of: "
                "PlaneOptions, BoxOptions, SphereOptions, CylinderOptions."
            )
        # Unify: either ``fixed`` or ``is_fixed_base`` opts the entity in.
        # ``is_fixed_base`` is the canonical Eden field; ``fixed`` is kept
        # for backward-compatibility with Genesis-style construction.
        if self.fixed or self.is_fixed_base:
            self.fixed = True
            self.is_fixed_base = True

        self._set_dofs_unactuated()

        if self.material is not None:
            _validate_no_dofs_for_non_rigid_material(self)


class PlaneOptions(PrimitiveOptions):
    fixed: bool = True
    normal: UnitVec3FType = (0.0, 0.0, 1.0)
    plane_size: PositiveVec2FType = (1e3, 1e3)
    tile_size: PositiveVec2FType = (1.0, 1.0)

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Plane(
            **self._collect_morph_kwargs(
                normal=self.normal,
                plane_size=self.plane_size,
                tile_size=self.tile_size,
            ),
        )


class BoxOptions(PrimitiveOptions):
    size: Vec3FType = (0.1, 0.1, 0.1)

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Box(
            **self._collect_morph_kwargs(
                size=self.size,
            ),
        )


class SphereOptions(PrimitiveOptions):
    radius: float = 0.1

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Sphere(
            **self._collect_morph_kwargs(
                radius=self.radius,
            ),
        )


class CylinderOptions(PrimitiveOptions):
    radius: float = 0.1
    height: float = 0.1

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Cylinder(
            **self._collect_morph_kwargs(
                radius=self.radius,
                height=self.height,
            ),
        )


class TerrainTermOptions(ConfigurableOptions):
    """TerrainGenerator options for Eden, configure by `TerrainGenerator.configure()`."""


class TerrainOptions(BaseEntityOptions):
    """
    Terrain entity options for Eden, wrapping Genesis' Terrain related options.

    Note that all dofs/articulation-related fields are set to empty/False and is_fixed_base is always set to True.

    Parameters
    ----------
    horizontal_scale : float
        The size of each cell in the subterrain in meters. Defaults to 0.25.
    vertical_scale : float
        The height of each step in the subterrain in meters. Defaults to 0.005.
    subterrain_size : tuple of float
        The size of each subterrain in meters. Defaults to (12.0, 12.0).
    uv_scale : float
        The scale of the UV mapping for the terrain. Defaults to 1.0.
    randomize : bool
        Whether to randomize the subterrains that involve randomness. Defaults to False.
    terrain_generators : list[list[TerrainTermOptions]]
        Terrain generator configuration grid.
    use_checker_surface : bool
        Whether to render checkerboard surface instead of a custom surface.
    material : MaterialLike | None
        Material used for terrain.
    surface : SurfaceOptions | Surface | None
        Surface appearance used for terrain.
    visualize_contact : bool
        Whether to visualize contacts involving terrain.
    vis_mode : VisMode
        Visualization mode for terrain.
    is_support_enabled : bool
        Whether support-point computation is enabled.
    default_root_pos : array-like[float, float, float]
        Default root position.
    default_root_quat : array-like[float, float, float, float]
        Default root orientation as quaternion in (w, x, y, z).
    """

    horizontal_scale: float = 0.25  # meter size of each cell in the subterrain
    vertical_scale: float = 0.005  # meter height of each step in the subterrain
    subterrain_size: Vec2FType = (12.0, 12.0)  # meter
    uv_scale: float = 1.0
    randomize: bool = False
    terrain_generators: list[list[TerrainTermOptions]] = [
        [
            FlatTerrain.configure(),
        ],
    ]
    use_checker_surface: bool = False

    if TYPE_CHECKING:
        morph: MorphLike

    def model_post_init(self, context):
        super().model_post_init(context)

        self.is_fixed_base = True
        self._set_dofs_unactuated()

        # Validation
        if self.use_checker_surface and self.surface is not None:
            raise ValueError("Cannot use checker surface with a custom surface.")


class GroupedEntityOptions(BaseEntityOptions):
    """Grouped entity used for heterogeneous batching.

    Supports both non-articulated entities (Mesh, Primitive) and articulated entities (URDF)
    that share the same joint structure (same number of links, joints, and DOFs).

    Parameters
    ----------
    grouped_entities : list[EntityOptions | PrimitiveOptions]
        A list of entity configurations to be grouped.
    is_fixed_base : bool
        Whether grouped entities should be treated as fixed base.
    dofs_name : list[str]
        Controllable DOF names shared by the group.
    dofs_spec : ClassVar[dict[str, type[ActuatorSpecOptions]]]
        Class-level default actuator specification map for grouped DOFs.
    default_dofs_pos : dict[str, float]
        Default grouped DOF positions.
    default_dofs_vel : dict[str, float]
        Default grouped DOF velocities.
    default_dofs_kp : dict[str, float]
        Default grouped DOF position gains.
    default_dofs_kd : dict[str, float]
        Default grouped DOF velocity gains.
    default_dofs_stiffness : dict[str, float]
        Default grouped DOF stiffness values.
    default_dofs_damping : dict[str, float]
        Default grouped DOF damping values.
    default_dofs_armature : dict[str, float]
        Default grouped DOF armature values.
    default_dofs_pos_limits : dict[str, array-like[float, float]]
        Hard grouped DOF position limits.
    soft_dofs_pos_limits : dict[str, array-like[float, float]]
        Soft grouped DOF position limits.
    default_dofs_force_limits : dict[str, float]
        Default grouped DOF force limits.
    """

    # TODO: add TerrainOptions support
    grouped_entities: list[EntityOptions | PrimitiveOptions] = Field(default_factory=list)

    def model_post_init(self, context):
        super().model_post_init(context)

        # Validation for grouped entities
        for entity in self.grouped_entities:
            if not isinstance(entity, (EntityOptions, PrimitiveOptions)):
                raise ValueError(
                    f"Invalid entity type: {type(entity)}, expected one of: "
                    f"EntityOptions, PrimitiveOptions, got {type(entity)}"
                )

        # Detect articulated/actuated flags from grouped entities.
        # All grouped entities must agree on is_articulated and is_actuated.
        any_articulated = any(getattr(e, "is_articulated", False) for e in self.grouped_entities)
        any_actuated = any(getattr(e, "is_actuated", False) for e in self.grouped_entities)

        if any_articulated:
            # Validate all entities are articulated (must share same joint structure)
            for e in self.grouped_entities:
                if not getattr(e, "is_articulated", False):
                    raise ValueError(
                        "All grouped entities must have the same is_articulated flag. "
                        "Cannot mix articulated and non-articulated entities."
                    )

            # Propagate DOF settings from the first entity (canonical structure)
            first = self.grouped_entities[0]
            if isinstance(first, EntityOptions):
                for field_name in _DOF_FIELD_NAMES:
                    val = getattr(first, field_name, None)
                    if val is not None:
                        setattr(self, field_name, val)

        if any_actuated:
            # Validate all entities are actuated
            for e in self.grouped_entities:
                if not getattr(e, "is_actuated", False):
                    raise ValueError(
                        "All grouped entities must have the same is_actuated flag. "
                        "Cannot mix actuated and non-actuated entities."
                    )

        self.is_articulated = any_articulated
        self.is_actuated = any_actuated


class RobotOptions(EntityOptions):
    """Robot entity options for Eden."""

    is_actuated: bool = True
    is_articulated: bool = True
    actuated_dofs_name: ClassVar[list[str]] = []
    ee_links_name: ClassVar[list[str]] = []

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        if "actuated_dofs_name" not in cls.__dict__:
            dofs_field = cls.model_fields.get("dofs_name")
            if dofs_field is not None and isinstance(dofs_field.default, list):
                cls.actuated_dofs_name = list(dofs_field.default)


@functools.lru_cache(maxsize=1)
def _ghost_defaults() -> dict:
    return dict(
        material=KinematicMaterialOptions(),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ColorTexture(color=(0.4, 0.7, 1.0)),
            opacity_texture=gs.textures.ColorTexture(color=(0.5,)),
        ),
    )


class _LazyGhostDefaults:
    """Lazy proxy so ``gs.surfaces.Default()`` isn't called at import time."""

    def __iter__(self):
        return iter(_ghost_defaults())

    def keys(self):
        return _ghost_defaults().keys()

    def __getitem__(self, key):
        return _ghost_defaults()[key]


GHOST_DEFAULTS = _LazyGhostDefaults()


class SceneOptions(ConfigurableOptions):
    """
    Scene options for Eden, specifying the entities to construct the scene.

    Parameters
    ----------
    attachments_dict: dict[str, tuple[str, str]]
        A dictionary of entity_name -> (attach_to_entity_name, attach_to_link_name).
        e.g., {"gripper": ("arm", "attachment")}
    <entity_name>: EntityOptions | PrimitiveOptions | TerrainOptions
        An entity configuration to be used for the given entity name.
    """

    attachments_dict: dict[str, tuple[str, str]] = {}

    def model_post_init(self, context):
        super().model_post_init(context)

        # Validate entity types (deferred ordering handled in build).
        # Declared fields (attachments_dict) live on the instance,
        # not in __pydantic_extra__, so iterating extras gives us only the
        # user-provided named entity kwargs.
        extra = self.__pydantic_extra__ or {}
        # Back-compat: configs serialized before the splat feature was removed carry a
        # now-unknown ``splat_options`` key. Drop it (rather than validate it as a scene
        # entity) so pre-cleanup checkpoints still load.
        extra.pop("splat_options", None)
        for key, entity in extra.items():
            if key.startswith("_option_"):
                continue
            if not isinstance(entity, BaseEntityOptions):
                raise ValueError(
                    f"Invalid entity type: {type(entity)}, expected BaseEntityOptions (or subclass), got {type(entity)}"
                )


EntityOptionsLike: TypeAlias = EntityOptions | PrimitiveOptions | TerrainOptions | GroupedEntityOptions
"""Union of the user-facing entity options classes."""
