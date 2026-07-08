"""Surface appearance options (plastic, metal, glass, ...)."""

from __future__ import annotations

from typing import ClassVar, Literal, TypeAlias

import genesis as gs
from genesis.options.surfaces import Surface
from genesis.typing import FArrayType, UnitInterval

from eden.options.options import ConfigurableOptions


class SurfaceOptions(ConfigurableOptions):
    """
    Deferred, serializable description of a Genesis surface.

    Genesis surfaces (``gs.surfaces.*``) cannot be instantiated before ``gs.init()``
    (they reference ``gs.EPS`` internally), so they cannot be embedded directly in an
    ``EdenConfig`` that is constructed at import time. ``SurfaceOptions`` captures the
    surface specification as plain config data and materializes the real Genesis surface
    lazily via :meth:`to_genesis_surface` at build time.

    Subclasses set ``_genesis_surface_cls`` to the corresponding ``gs.surfaces.*`` class;
    every field that was explicitly set is then forwarded through.

    Parameters
    ----------
    color : array-like[float] | None, optional
        Surface color. Shortcut for the primary texture with a single color (RGB or RGBA).
    opacity : float | None, optional
        Opacity of the surface in [0, 1].
    roughness : float | None, optional
        Roughness of the surface in [0, 1].
    metallic : float | None, optional
        Metalness of the surface in [0, 1].
    emissive : array-like[float] | None, optional
        Emissive color of the surface.
    ior : float | None, optional
        Index of refraction.
    vis_mode : str | None, optional
        How the entity should be visualized ('visual', 'collision', 'particle', 'sdf', 'recon').
    double_sided : bool | None, optional
        Whether to render both sides of the surface.
    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals. Default True.
    """

    color: FArrayType | None = None
    opacity: UnitInterval | None = None
    roughness: UnitInterval | None = None
    metallic: UnitInterval | None = None
    emissive: FArrayType | None = None
    ior: float | None = None
    vis_mode: Literal["visual", "collision", "particle", "sdf", "recon"] | None = None
    double_sided: bool | None = None
    smooth: bool = True

    _genesis_surface_cls: ClassVar[type[Surface]] = gs.surfaces.Default

    def to_genesis_surface(self) -> Surface:
        """Materialize the Genesis surface. Must be called after ``gs.init()``."""
        # Forward every set value (declared fields + dynamic extras), dropping unset
        # ``None`` shortcuts so the Genesis surface keeps its own defaults.
        kwargs = {key: value for key, value in self.dict().items() if value is not None}
        return self._genesis_surface_cls(**kwargs)


class DefaultSurfaceOptions(SurfaceOptions):
    """Options for ``gs.surfaces.Default``."""

    _genesis_surface_cls: ClassVar[type[Surface]] = gs.surfaces.Default


class PlasticSurfaceOptions(SurfaceOptions):
    """Options for ``gs.surfaces.Plastic``."""

    _genesis_surface_cls: ClassVar[type[Surface]] = gs.surfaces.Plastic


class MetalSurfaceOptions(SurfaceOptions):
    """Options for ``gs.surfaces.Metal``."""

    _genesis_surface_cls: ClassVar[type[Surface]] = gs.surfaces.Metal


class GlassSurfaceOptions(SurfaceOptions):
    """Options for ``gs.surfaces.Glass``."""

    _genesis_surface_cls: ClassVar[type[Surface]] = gs.surfaces.Glass


class EmissionSurfaceOptions(SurfaceOptions):
    """Options for ``gs.surfaces.Emission``."""

    _genesis_surface_cls: ClassVar[type[Surface]] = gs.surfaces.Emission


SurfaceLike: TypeAlias = SurfaceOptions | Surface
"""Either an Eden :class:`SurfaceOptions` (deferred) or a built Genesis :class:`Surface`."""


def resolve_surface(surface: SurfaceLike | None) -> Surface | None:
    """Resolve a surface spec to a Genesis surface, building deferred ``SurfaceOptions``."""
    if isinstance(surface, SurfaceOptions):
        return surface.to_genesis_surface()
    return surface
