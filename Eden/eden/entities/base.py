"""Base Entity wrapper providing shared pose/state accessors over a Genesis entity.

:class:`Entity` holds shared state (morph, material, pose getters/setters) and is
specialized by :class:`~eden.entities.rigid.RigidEntity` (articulated/actuated),
:class:`~eden.entities.particle.ParticleEntity` (MPM/SPH/PBD),
:class:`~eden.entities.fem.FEMEntity`, and :class:`~eden.entities.terrain.Terrain`.
``_entity_class_for_options`` dispatches to the right subclass from the material type.

All quaternions are wxyz. Pose setters (``set_pos`` / ``set_quat``) take an ``envs_idx``
to target specific environments; note that ``ParticleEntity`` bakes its pose into the
morph at creation time and raises from ``set_pos`` / ``set_quat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import genesis as gs
import genesis.utils.geom as gu
import torch
from genesis.engine.materials.base import Material
from genesis.typing import UnitVec4FType, Vec3FType

import eden as en
from eden.options import EntityOptions, GroupedEntityOptions, PrimitiveOptions
from eden.options.materials import MaterialLike
from eden.options.surfaces import PlasticSurfaceOptions, SurfaceLike, resolve_surface
from eden.types import MorphLike, VisMode
from eden.utils.assets import get_asset_path
from eden.utils.common import ConfigurableMixin
from eden.utils.geom import align_up_and_front
from eden.utils.isaac_math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity as GenesisEntity
    from genesis.options.surfaces import Surface

    from eden.envs.base import EnvBase


class Entity(ConfigurableMixin[EntityOptions]):
    """
    Entity class for Eden, wrapping Genesis' Entity class.

    Parameters
    ----------
    default_root_pos: array-like[float, float, float]
        The default position of the entity's root link.
    default_root_quat: array-like[float, float, float, float]
        The default orientation of the entity's root link (w, x, y, z).
    """

    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)
    default_root_quat: UnitVec4FType = (
        1.0,
        0.0,
        0.0,
        0.0,
    )  # (w, x, y, z)

    forward_vec: Vec3FType = (1.0, 0.0, 0.0)

    if TYPE_CHECKING:
        morph: MorphLike
    material: MaterialLike | None = None
    surface: SurfaceLike | None = None
    visualize_contact: bool = False
    vis_mode: VisMode = "visual"

    is_fixed_base: bool = False

    def __init__(self, env: EnvBase, options: EntityOptions):
        self._options = options
        self._env = env
        self._is_attaching = False
        self.is_heterogeneous = isinstance(self._options, GroupedEntityOptions)
        self._grouped_default_root_pos: list[torch.Tensor] = []
        self._grouped_default_root_quat: list[torch.Tensor] = []

        # Create morph from options if not already set
        if not self.is_heterogeneous:
            self._create_morph_from_options()
        else:
            self._create_grouped_morphs_from_options()

        # NOTE: avoid the SupportSurfaceMixin's parameters to be set
        for name in self.get_parameter_names():
            if name in [
                "_kernel_update_aabbs",
                "_kernel_filter_detection",
                "_data_oriented",
                "offset",
            ]:
                continue
            if name in self._options.model_dump():
                setattr(self, name, getattr(self._options, name))
            else:
                setattr(self, name, getattr(self, name))

        self.forward_vec = torch.tensor(self.forward_vec, dtype=gs.tc_float, device=self.device).unsqueeze(0)
        self.default_root_pos = torch.tensor(self.default_root_pos, dtype=gs.tc_float, device=self.device).unsqueeze(0)
        self.default_root_quat = torch.tensor(
            self.default_root_quat,
            dtype=gs.tc_float,
            device=self.device,
        ).unsqueeze(0)

    def _create_morph_from_options(self):
        """Create morph from EntityOptions fields."""
        if isinstance(self._options, PrimitiveOptions):
            self.morph = self._options.get_morph()
            return

        if not isinstance(self._options, EntityOptions):
            return

        # Get the file path
        file = get_asset_path(
            file=self._options.file,
            registry=self._options.registry,
            dataset=self._options.dataset,
            local_dir=self._options.local_dir,
        )

        # Compute alignment quaternion
        quat = gu.R_to_quat(align_up_and_front(self._options.up, self._options.front))

        # Prepare common options for morph creation
        common_options = dict(
            file=file,
            pos=self._options.default_root_pos,
            quat=quat,
            batch_fixed_verts=self._options.batch_fixed_verts,
            scale=self._options.scale,
            visualization=self._options.visualization,
            collision=self._options.collision,
            convexify=self._options.convexify,
            decompose_robot_error_threshold=self._options.decompose_robot_error_threshold,
            decompose_object_error_threshold=self._options.decompose_object_error_threshold,
            decimate=self._options.decimate,
            decimate_face_num=self._options.decimate_face_num,
            decimate_aggressiveness=self._options.decimate_aggressiveness,
            coacd_options=gs.options.CoacdOptions(
                threshold=self._options.coacd_threshold,
                preprocess_resolution=self._options.coacd_preprocess_resolution,
                # NOTE: COACD's PCA preprocessing rotates the mesh to its principal axes
                # before decomposition and returns hulls in that rotated frame without
                # applying the inverse transform. That leaves the collision hulls misaligned
                # with the visual mesh. Default to False (matches Genesis CoacdOptions).
                pca=False,
                decimate=self._options.decimate,
            ),
        )

        # Create morph based on file extension
        if file.endswith(".urdf"):
            self.morph = gs.morphs.URDF(
                fixed=self._options.is_fixed_base,
                links_to_keep=self._options.links_to_keep,
                default_armature=self._options.default_armature,
                merge_fixed_links=True,
                **common_options,
            )
        elif file.endswith(".xml"):
            # Genesis's `gs.morphs.MJCF` has no `fixed` parameter; the free/fixed
            # distinction is encoded inside the MJCF file (e.g. a `<freejoint/>`
            # under the root body). `is_fixed_base=True` cannot be honored here
            # without rewriting the asset, so warn the user instead of silently
            # dropping the flag.
            if self._options.is_fixed_base:
                en.logger.warning(
                    f"`is_fixed_base=True` is not supported for MJCF entities "
                    f"({file!r}); Genesis does not expose a `fixed` parameter on "
                    f"`gs.morphs.MJCF`. The flag will be ignored and the root "
                    f"joint declared in the MJCF (typically `<freejoint/>`) will "
                    f"be used as-is. To pin the base, edit the MJCF to remove "
                    f"the freejoint, or convert the asset to URDF."
                )
            self.morph = gs.morphs.MJCF(default_armature=self._options.default_armature, **common_options)
        elif file.endswith((".usd", ".usda", ".usdc", ".usdz")):
            self.morph = gs.morphs.USD(
                fixed=self._options.is_fixed_base,
                **common_options,
            )
        elif file.endswith((".obj", ".stl", ".ply", ".glb", ".gltf")):
            self.morph = gs.morphs.Mesh(
                fixed=self._options.is_fixed_base,
                contype=self._options.contype,
                conaffinity=self._options.conaffinity,
                **common_options,
            )
        else:
            raise ValueError(f"Invalid file type: {file}")

    def _create_grouped_morphs_from_options(self):
        """Create grouped morphs for heterogeneous entities."""
        from eden.options.entities import GroupedEntityOptions

        if not isinstance(self._options, GroupedEntityOptions):
            return

        _SUPPORTED_MORPH_TYPES = (gs.morphs.Primitive, gs.morphs.Mesh, gs.morphs.URDF, gs.morphs.MJCF)

        morphs_heterogeneous = []
        for entity_options in self._options.grouped_entities:
            # Prevent nested grouped entities, which would cause infinite recursion
            if isinstance(entity_options, GroupedEntityOptions):
                raise ValueError(
                    "Invalid configuration: GroupedEntityOptions cannot contain other GroupedEntityOptions."
                    " 'grouped_entities' must contain only non-grouped EntityOptions."
                )
            # Create a temporary entity to get its morph
            from eden.entities.rigid import RigidEntity

            temp_entity = RigidEntity(self._env, entity_options)

            if not isinstance(temp_entity.morph, _SUPPORTED_MORPH_TYPES):
                raise ValueError(
                    f"Only Primitive, Mesh, URDF, and MJCF morphs are supported for grouped entity, "
                    f"got {type(temp_entity.morph).__name__}."
                )

            self._grouped_default_root_pos.append(temp_entity.default_root_pos.squeeze(0).clone())
            self._grouped_default_root_quat.append(temp_entity.default_root_quat.squeeze(0).clone())

            # Override morph fixed property if needed (MJCF does not support fixed)
            if not isinstance(temp_entity.morph, gs.morphs.MJCF):
                if hasattr(temp_entity.morph, "fixed"):
                    temp_entity.morph.fixed = self._options.is_fixed_base

            morphs_heterogeneous.append(temp_entity.morph)

        self.morph = morphs_heterogeneous

    def get_default_root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get default root pose for all environments."""
        default_pos = self.default_root_pos.repeat(self._env.num_envs, 1)
        default_quat = self.default_root_quat.repeat(self._env.num_envs, 1)
        return default_pos, default_quat

    def pre_build(self) -> None:
        """Add the entity to the Genesis scene before the scene is built."""
        self._entity: GenesisEntity = self._env.scene.add_entity(
            self.morph,
            material=self._parse_material(),
            surface=self._parse_surface(),
            visualize_contact=self.visualize_contact,
            vis_mode=self.vis_mode,
        )

    def _parse_surface(self) -> Surface | None:
        """Resolve the configured surface to a Genesis surface (after ``gs.init()``).

        Deferred :class:`SurfaceOptions` are materialized here. If no surface is set
        but the ``color`` shortcut is, a plastic surface with that color is created.
        """
        if self.surface is None and getattr(self._options, "color", None) is not None:
            self.surface = PlasticSurfaceOptions(color=self._options.color)
        return resolve_surface(self.surface)

    def post_build(self) -> None:
        """Finalize the entity after the scene is built (no-op for the base class)."""
        pass

    def _parse_material(self) -> Material:
        if self.material is None:
            self.material = en.materials.RigidMaterialOptions()
        return self.material.to_genesis_material()

    @property
    def name(self) -> str:
        """The entity's registered name."""
        return getattr(self, "_name", self._options.__class__.__name__)

    @property
    def entity(self):
        """The underlying Genesis entity this wrapper delegates to."""
        return self._entity

    @property
    def metadata(self):
        """User-defined metadata attached to the entity's options."""
        return self._options.metadata

    @property
    def solver(self):
        """The Genesis solver that owns this entity."""
        return self._entity._solver

    @property
    def env(self):
        """The environment this entity belongs to."""
        return self._env

    @property
    def idx(self):
        """The entity's index within its solver."""
        return self._entity.idx

    @property
    def uid(self):
        """The entity's globally unique identifier."""
        return self._entity.uid

    @property
    def is_built(self) -> bool:
        """Whether the owning scene has been built."""
        return self._env.is_built

    @property
    def device(self) -> torch.device:
        """The torch device the entity's tensors live on."""
        return self._env.device

    @property
    def fixed(self) -> bool:
        """Whether the entity has a fixed base (cannot move freely)."""
        return self.is_fixed_base

    # ------------------------------------------------------------------------------------
    # ------------------------------------- universal ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_pos(self, envs_idx=None):
        """Return the position of the entity's base link in world frame."""
        return self._entity.get_pos(envs_idx=envs_idx)

    def get_quat(self, envs_idx=None):
        """Return the quaternion of the entity's base link in world frame.

        The quaternion is in the format of (w, x, y, z).
        """
        return self._entity.get_quat(envs_idx=envs_idx)

    def get_heading(self, envs_idx=None):
        """Return the heading of the entity's base link in world frame."""
        quat = self.get_quat(envs_idx=envs_idx)
        forward = self.forward_vec.expand(quat.shape[0], -1)
        forward_w = quat_apply(quat, forward)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    def get_T(self, envs_idx=None):
        """Return the entity's base-link pose as a 4x4 homogeneous transform matrix.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are considered. Defaults to None.
        """
        pos = self.get_pos(envs_idx=envs_idx)
        quat = self.get_quat(envs_idx=envs_idx)
        return gu.trans_quat_to_T(pos, quat)

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return the linear velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        frame : Literal["world", "body"], optional
            Coordinate frame for the returned linear velocity. Valid options:
            - "world": linear velocity expressed in world coordinates (default)
            - "body": linear velocity expressed in the body (base link) frame
        """
        vel_w = self._entity.get_vel(envs_idx=envs_idx)
        if frame == "world":
            return vel_w
        elif frame == "body":
            quat_wb = self.get_quat(envs_idx=envs_idx)
            return quat_apply_inverse(quat_wb, vel_w)
        else:
            raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return the angular velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        frame : Literal["world", "body"], optional
            Coordinate frame for the returned angular velocity. Valid options:
            - "world": angular velocity expressed in world coordinates (default)
            - "body": angular velocity expressed in the body (base link) frame
        """
        ang_w = self._entity.get_ang(envs_idx=envs_idx)

        if frame == "world":
            return ang_w
        elif frame == "body":
            quat_wb = self.get_quat(envs_idx=envs_idx)
            return quat_apply_inverse(quat_wb, ang_w)
        else:
            raise ValueError(f"Invalid frame '{frame}'. Expected 'world' or 'body'.")

    def set_pos(self, pos, envs_idx=None) -> None:
        """Set the entity's position."""
        self._entity.set_pos(pos, envs_idx=envs_idx)

    def set_quat(self, quat, envs_idx=None, relative=True) -> None:
        """
        Set the entity's orientation.

        Parameters
        ----------
        quat : array_like
            The orientation to set in the format of (w, x, y, z).
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        relative : bool, optional
            If True, the quaternion is applied relative to the entity's default orientation. Defaults to True.
        """
        self._entity.set_quat(quat, envs_idx=envs_idx, relative=relative)

    def get_mass(self):
        """Return the total mass of the entity."""
        return self._entity.get_mass()

    def set_mass(self, mass):
        """Set the total mass of the entity.

        Parameters
        ----------
        mass : float
            The target total mass.
        """
        self._entity.set_mass(mass)
