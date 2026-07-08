"""Renderer configuration options (ray tracer)."""

from typing import Any

import genesis as gs
from genesis.typing import Vec3FType
from genesis.options.surfaces import Surface
from genesis.options.renderers import RayTracer


class RayTracerOptions(RayTracer):
    """Options for the ray-tracing renderer.

    Parameters
    ----------
    env_surface: Surface | None, optional
        Environment surface.
    env_radius: float, optional
        Environment radius.
    env_pos: array-like[float, float, float], optional
        Environment position.
    env_euler: array-like[float, float, float]
        The euler angles of the environment.
    lights : list of dict, optional
        List of lights. Each light is a dictionary with keys: ['pos', 'color', 'intensity', 'radius'].
        - 'pos': array-like[float, float, float]
        - 'color': array-like[float, float, float]
        - 'intensity': float
        - 'radius': float
    """

    state_limit: int = 2**25
    tracing_depth: int = 32
    rr_depth: int = 0
    rr_threshold: float = 0.95

    env_surface: Surface = gs.surfaces.Emission(
        emissive_texture=gs.textures.ImageTexture(
            image_path="textures/indoor_bright.png",
        ),
    )
    env_radius: float = 15.0
    env_pos: Vec3FType = (0.0, 0.0, 0.0)
    env_euler: Vec3FType = (0, 0, 180)
    lights: list[dict[str, Any]] = [
        {
            "pos": (0.0, 0.0, 10.0),
            "radius": 2.0,
            "color": (15.0, 15.0, 15.0),
            "intensity": 10.0,
        },
    ]
