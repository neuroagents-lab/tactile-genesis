"""Empty scene options (plane-only, single and bimanual)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions
from eden.options.surfaces import SurfaceLike


class _PlaneInjectingScene(SceneOptions):
    """Internal mixin that injects a ground :class:`PlaneOptions` into the scene.

    Subclass this for scenes that always include a ground plane and accept
    ``is_support_enabled`` / ``surface`` knobs to configure it. The plane is
    forwarded to the parent ``SceneOptions`` constructor under the ``plane`` key.
    """

    if TYPE_CHECKING:
        plane: PlaneOptions

    def __init__(
        self,
        is_support_enabled: bool = False,
        surface: SurfaceLike | None = None,
        **data,
    ):
        super().__init__(
            plane=PlaneOptions(
                surface=surface,
                is_support_enabled=is_support_enabled,
            ),
            **data,
        )


class EmptySceneOptions(_PlaneInjectingScene):
    """
    Options for a basic scene with a robot and a ground plane.

    Parameters
    ----------
    robot: EntityOptions
        The entity configuration to be used for the robot.
    is_support_enabled: bool
        Whether to enable support for the plane.
    surface: SurfaceOptions | Surface | None
        The surface to be used for the plane.
        `surface=PlasticSurfaceOptions(color=(1.0, 1.0, 1.0))` to create a white plane.
    **data: Any
        Additional data to be passed to the scene options.

    Entities
    --------
    robot: EntityOptions
        The entity configuration to be used for the robot.
    plane: PrimitiveOptions
        The plane entity configuration.
    """

    robot: EntityOptions


class BimanualEmptySceneOptions(_PlaneInjectingScene):
    """
    Options for a basic bimanual scene with two robots and a ground plane.

    Parameters
    ----------
    right_robot: EntityOptions
        The entity configuration to be used for the right robot.
    left_robot: EntityOptions
        The entity configuration to be used for the left robot.
    is_support_enabled: bool
        Whether to enable support for the plane.
    surface: SurfaceOptions | Surface | None
        The surface to be used for the plane.
        `surface=PlasticSurfaceOptions(color=(1.0, 1.0, 1.0))` to create a white plane.
    **data: Any
        Additional data to be passed to the scene options.

    Entities
    --------
    right_robot: EntityOptions
        The entity configuration to be used for the right robot.
    left_robot: EntityOptions
        The entity configuration to be used for the left robot.
    plane: PrimitiveOptions
        The plane entity configuration.
    """

    right_robot: EntityOptions
    left_robot: EntityOptions
