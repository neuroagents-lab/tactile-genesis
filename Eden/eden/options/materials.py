"""Material options for rigid, MPM, SPH, FEM, and PBD entities."""

from __future__ import annotations
from typing import ClassVar, Literal, TypeAlias

import genesis as gs

from eden.options.options import ConfigurableOptions


class MaterialOptions(ConfigurableOptions):
    """
    Options for Genesis materials.

    Subclasses set ``_genesis_material_cls`` to the corresponding ``gs.materials.*``
    class; ``to_genesis_material`` then forwards every declared field through.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        If True, the entity's visual mesh is included in the raycaster BVH used by
        Lidar / DepthCamera sensors. Required for non-colliding background meshes
        (e.g. photogrammetry GLBs with ``collision=False``) to be hit by lidar rays.
        Must be set before ``scene.build()``. Default is False.
    """

    use_visual_raycasting: bool = False

    _genesis_material_cls: ClassVar[type | None] = None

    def to_genesis_material(self):
        cls = self._genesis_material_cls
        if cls is None:
            raise NotImplementedError(f"{type(self).__name__} does not implement to_genesis_material()")
        return cls(**self.dict())


class KinematicMaterialOptions(MaterialOptions):
    """Options for Genesis kinematic materials."""

    _genesis_material_cls: ClassVar[type] = gs.materials.Kinematic


class RigidMaterialOptions(MaterialOptions):
    """
    Options for Genesis rigid materials.

    Parameters
    ----------
    rho : float, optional
        The density of the material used to compute mass. Default is 200.0.
    friction : float, optional
        Friction coefficient within the rigid solver. If None, a default of 1.0 may be used or parsed from file.
    needs_coup : bool, optional
        Whether the material participates in coupling with other solvers. Default is True.
    coup_friction : float, optional
        Friction used during coupling. Must be non-negative. Default is 0.1.
    coup_softness : float, optional
        Softness of coupling interaction. Must be non-negative. Default is 0.002.
    coup_restitution : float, optional
        Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0.
    sdf_cell_size : float, optional
        Cell size in SDF grid in meters. Defines grid resolution. Default is 0.005.
    sdf_min_res : int, optional
        Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
    sdf_max_res : int, optional
        Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
    gravity_compensation : float, optional
        Compensation factor for gravity. 1.0 cancels gravity. Default is 0.
    coup_type : str or None, optional
        Coupling mode for this entity. Only used by the IPC coupler. Requires ``needs_coup=True``.
        If None, auto-selected based on entity type: ``'external_articulation'`` for fixed-base
        articulated robots, ``'two_way_soft_constraint'`` for floating-base robots, and
        ``'ipc_only'`` for non-articulated objects. Valid values:
            - 'two_way_soft_constraint': Two-way soft coupling.
            - 'external_articulation': Joint-level coupling for articulated bodies. Joint positions will be coupled at
            the DOF level.
            - 'ipc_only': IPC controls entity, transforms copied to Genesis (one-way). Only supported by rigid
            non-articulated objects.
        Default is None.
    coup_links : list of str or None, optional
        Link names to include in coupling. When set, only the named links participate
        in coupling; other links are excluded. Only supported with needs_coup=True and
        ``two_way_soft_constraint`` type in IPC. Default is None.
    enable_coup_collision : bool, optional
        Whether coupler collision is enabled for this entity's links. Only used by the IPC coupler.
        Unlike ``needs_coup=False`` (which removes the entity from the coupler entirely), setting this to
        False keeps the entity in the coupler for coupling forces but disables contact response. Default is True.
    coup_collision_links : list of str or None, optional
        Link names whose geoms participate in coupler collision. Only used by the IPC coupler.
        Only effective when ``enable_coup_collision=True``. If None, all coupled links have collision.
        When set, only the named links get coupler collision; other links are marked no-collision.
        Default is None.
    contact_resistance : float or None, optional
        IPC coupling contact resistance/stiffness override for this entity. ``None`` means use
        ``IPCCouplerOptions.contact_resistance``. Default is None.
    """

    rho: float = 200.0
    friction: float | None = None
    needs_coup: bool = True
    coup_friction: float = 0.1
    coup_softness: float = 0.002
    coup_restitution: float = 0.0
    sdf_cell_size: float = 0.005
    sdf_min_res: int = 32
    sdf_max_res: int = 128
    gravity_compensation: float = 0.0
    coup_type: Literal["two_way_soft_constraint", "external_articulation", "ipc_only"] | None = None
    coup_links: list[str] | None = None
    enable_coup_collision: bool = True
    coup_collision_links: list[str] | None = None
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.Rigid


class MPMElasticMaterialOptions(MaterialOptions):
    """
    Options for Genesis MPM elastic materials.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    model: str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
    """

    E: float = 3e5
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    sampler: str | None = None
    model: Literal["corotation", "neohooken"] = "corotation"

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.Elastic


class MPMElastoPlasticMaterialOptions(MaterialOptions):
    """Options for an MPM elasto-plastic material.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    yield_lower: float, optional
        Lower bound for the yield clamp (ignored if using von Mises). Default is 2.5e-2.
    yield_higher: float, optional
        Upper bound for the yield clamp (ignored if using von Mises). Default is 4.5e-2.
    use_von_mises: bool, optional
        Whether to use von Mises yield criterion. Default is True.
    von_mises_yield_stress: float, optional
        Yield stress for von Mises criterion. Default is 10000.
    """

    E: float = 1e6  # Young's modulus
    nu: float = 0.2  # Poisson's ratio
    rho: float = 1000.0  # density (kg/m^3)
    lam: float | None = None
    mu: float | None = None
    sampler: str | None = None
    yield_lower: float = 2.5e-2
    yield_higher: float = 4.5e-2
    use_von_mises: bool = True  # von Mises yield criterion
    von_mises_yield_stress: float = 10000.0

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.ElastoPlastic


class MPMLiquidMaterialOptions(MaterialOptions):
    """Options for an MPM liquid material.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    viscous: bool, optional
        Whether the liquid is viscous. Simply set mu to zero when non-viscous. Default is False.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    viscous: bool = False
    sampler: str | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.Liquid


class SPHLiquidMaterialOptions(MaterialOptions):
    """Options for an SPH liquid material.

    Parameters
    ----------
    rho: float, optional
        The density (kg/m^3) the material tends to maintain in equilibrium (i.e., the "rest" or undeformed state). Default is 1000.
    stiffness: float, optional
        State stiffness (N/m^2). A material constant controlling how pressure increases with compression. Default is 50000.0.
    exponent: float, optional
        State exponent. Controls how nonlinearly pressure scales with density. Larger values mean stiffer response to compression. Default is 7.0.
    mu: float, optional
        The viscosity of the liquid. A measure of the internal friction of the fluid or material. Default is 0.005
    gamma: float, optional
        The surface tension of the liquid. Controls how strongly the material "clumps" together at boundaries. Default is 0.01
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults to 'regular' because:
        SPH is sensitive to the initial particle distribution, as it directly determines the initial density and pressure fields.
        To ensure numerical stability, particles must be initialized using a regular sampler that enforces near-uniform spacing.
        Irregular samplers (e.g. pbs, random) introduce local density fluctuations at initialization,
        which lead to large spurious pressure forces and can cause the simulation to become unstable or diverge.
    """

    rho: float = 1000.0
    stiffness: float = 50_000.0
    exponent: float = 7.0
    mu: float = 0.005
    gamma: float = 0.01
    sampler: str = "regular"

    _genesis_material_cls: ClassVar[type] = gs.materials.SPH.Liquid


class PBDLiquidMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) liquid materials.

    Particle-based fluid simulated with the PBD solver. The PBD solver's
    simulation domain (``lower_bound``/``upper_bound``) and iteration counts are
    configured via ``EnvOptions.pbd_options`` (a ``gs.options.PBDOptions``). If
    left unset, genesis uses its PBD defaults (a ``(-100, -100, 0)``..``(100,
    100, 100)`` domain) — set it to bound the domain to the working volume.
    Position/orientation are baked into the morph at creation time
    (``set_pos``/``set_quat`` raise on the resulting entity).

    Parameters
    ----------
    rho : float, optional
        Rest density of the fluid. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'random', 'regular'). 'pbs' is only supported on
        Linux x86. Default is 'pbs'.
    density_relaxation : float, optional
        Relaxation factor for the density (incompressibility) constraint. Larger
        values enforce incompressibility more strongly per iteration. Default is 0.2.
    viscosity_relaxation : float, optional
        Relaxation factor for the XSPH viscosity constraint. Larger values make the
        fluid more viscous. Default is 0.01.
    """

    rho: float = 1000.0
    sampler: Literal["pbs", "random", "regular"] = "pbs"
    density_relaxation: float = 0.2
    viscosity_relaxation: float = 0.01

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Liquid


class PBDClothMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) cloth materials.

    Thin-shell cloth simulated as a particle/edge network with the PBD solver.
    Built from a triangle mesh morph (e.g. ``EntityOptions(file="meshes/cloth.obj")``).
    The PBD solver iteration counts are configured via ``EnvOptions.pbd_options``
    (genesis defaults are used if unset). Individual particles can be pinned via the
    underlying Genesis entity's ``fix_particles`` / ``find_closest_particle``
    (accessible through ``entity.entity``).

    Parameters
    ----------
    rho : float, optional
        Surface density of the cloth. Default is 4.0.
    static_friction : float, optional
        Static friction coefficient for particle contact. Default is 0.15.
    kinetic_friction : float, optional
        Kinetic friction coefficient for particle contact. Default is 0.15.
    stretch_compliance : float, optional
        Compliance (inverse stiffness) of the stretch constraint. Smaller values
        make the cloth less stretchy. Default is 1e-7.
    bending_compliance : float, optional
        Compliance (inverse stiffness) of the bending constraint. Smaller values
        make the cloth resist folding more. Default is 1e-5.
    stretch_relaxation : float, optional
        Relaxation factor for the stretch constraint per iteration. Default is 0.3.
    bending_relaxation : float, optional
        Relaxation factor for the bending constraint per iteration. Default is 0.1.
    air_resistance : float, optional
        Air drag applied to the cloth particles. Default is 0.001.
    """

    rho: float = 4.0
    static_friction: float = 0.15
    kinetic_friction: float = 0.15
    stretch_compliance: float = 1e-7
    bending_compliance: float = 1e-5
    stretch_relaxation: float = 0.3
    bending_relaxation: float = 0.1
    air_resistance: float = 0.001

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Cloth


class PBDElasticMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) elastic (soft-body) materials.

    Volumetric soft body simulated with the PBD solver, governed by stretch,
    bending, and volume-preservation constraints. The PBD solver is configured
    via ``EnvOptions.pbd_options`` (genesis defaults are used if unset).

    Parameters
    ----------
    rho : float, optional
        Material density. Default is 1000.0.
    static_friction : float, optional
        Static friction coefficient for particle contact. Default is 0.15.
    kinetic_friction : float, optional
        Kinetic friction coefficient for particle contact. Default is 0.15.
    stretch_compliance : float, optional
        Compliance (inverse stiffness) of the stretch constraint. Default is 0.0.
    bending_compliance : float, optional
        Compliance (inverse stiffness) of the bending constraint. Default is 0.0.
    volume_compliance : float, optional
        Compliance (inverse stiffness) of the volume-preservation constraint.
        Default is 0.0.
    stretch_relaxation : float, optional
        Relaxation factor for the stretch constraint per iteration. Default is 0.1.
    bending_relaxation : float, optional
        Relaxation factor for the bending constraint per iteration. Default is 0.1.
    volume_relaxation : float, optional
        Relaxation factor for the volume constraint per iteration. Default is 0.1.
    """

    rho: float = 1000.0
    static_friction: float = 0.15
    kinetic_friction: float = 0.15
    stretch_compliance: float = 0.0
    bending_compliance: float = 0.0
    volume_compliance: float = 0.0
    stretch_relaxation: float = 0.1
    bending_relaxation: float = 0.1
    volume_relaxation: float = 0.1

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Elastic


class PBDParticleMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) free-particle materials.

    Bare particles driven by the PBD solver with no internal constraints (collide
    and respond to gravity only). The PBD solver is configured via
    ``EnvOptions.pbd_options`` (genesis defaults are used if unset).

    Parameters
    ----------
    rho : float, optional
        Material density. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'random', 'regular'). 'pbs' is only supported on
        Linux x86. Default is 'pbs'.
    """

    rho: float = 1000.0
    sampler: Literal["pbs", "random", "regular"] = "pbs"

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Particle


class FEMElasticMaterialOptions(MaterialOptions):
    """
    Options for Genesis FEM elastic materials.

    Used for volumetric deformable bodies (tetrahedralized meshes) simulated
    with the FEM solver. When an IPC coupler is active, these entities
    participate in penetration-free frictional contact.

    Parameters
    ----------
    E : float, optional
        Young's modulus controlling stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio describing volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    model : str, optional
        Constitutive model ('linear', 'stable_neohookean', 'linear_corotated'). Default is 'linear'.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float or None, optional
        IPC contact resistance/stiffness override. None uses the coupler global default. Default is None.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"
    friction_mu: float = 0.1
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.FEM.Elastic


class FEMClothMaterialOptions(MaterialOptions):
    """
    Options for Genesis FEM cloth (thin shell) materials.

    Used for cloth, fabric, and thin flexible surfaces simulated as 2D shells
    in the IPC backend. Requires an IPC coupler and GPU backend.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa) controlling stiffness. Default is 1e4.
    nu : float, optional
        Poisson ratio. Default is 0.49 (nearly incompressible).
    rho : float, optional
        Material density (kg/m^3). Default is 200.
    thickness : float, optional
        Shell thickness in meters. Default is 0.001 (1mm).
    bending_stiffness : float or None, optional
        Bending resistance. None disables bending. Default is None.
    model : str, optional
        FEM material model. Default is 'stable_neohookean'.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float or None, optional
        IPC contact resistance/stiffness override. None uses the coupler global default. Default is None.
    """

    E: float = 1e4
    nu: float = 0.49
    rho: float = 200.0
    thickness: float = 0.001
    bending_stiffness: float | None = None
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"
    friction_mu: float = 0.1
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.FEM.Cloth


class FEMMuscleMaterialOptions(MaterialOptions):
    """
    Options for Genesis FEM muscle materials.

    Extends FEM elastic material with muscle activation support for
    soft-body actuation.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    model : str, optional
        Constitutive model. Default is 'linear'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float or None, optional
        IPC contact resistance/stiffness override. Default is None.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"
    n_groups: int = 1
    friction_mu: float = 0.1
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.FEM.Muscle


MaterialLike: TypeAlias = (
    RigidMaterialOptions
    | KinematicMaterialOptions
    | MPMElasticMaterialOptions
    | MPMElastoPlasticMaterialOptions
    | MPMLiquidMaterialOptions
    | SPHLiquidMaterialOptions
    | PBDLiquidMaterialOptions
    | PBDClothMaterialOptions
    | PBDElasticMaterialOptions
    | PBDParticleMaterialOptions
    | FEMElasticMaterialOptions
    | FEMClothMaterialOptions
    | FEMMuscleMaterialOptions
)
"""Union of every concrete material options class."""
