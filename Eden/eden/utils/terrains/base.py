"""Base terrain generator and flat terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import genesis as gs
from genesis.typing import Vec2FType, Vec2IType
from genesis.ext.isaacgym import terrain_utils as tu

from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.options.entities import TerrainTermOptions


TERRAIN_GENERATOR_REGISTRY = Registry("TERRAIN_GENERATOR")


def generate_height_field(
    terrain_generators: list[list[TerrainTermOptions]],
    subterrain_size: Vec2FType,
    n_subterrains: Vec2IType,
    horizontal_scale: float,
    vertical_scale: float,
    randomize: bool,
) -> np.ndarray:
    """Generate the height field for the terrain."""
    subterrain_rows = int(subterrain_size[0] / horizontal_scale + gs.EPS) + 1
    subterrain_cols = int(subterrain_size[1] / horizontal_scale + gs.EPS) + 1
    heightfield = np.full(
        (
            n_subterrains[0] * (subterrain_rows - 1) + 1,
            n_subterrains[1] * (subterrain_cols - 1) + 1,
        ),
        fill_value=float("-inf"),
        dtype=gs.np_float,
    )

    for i, j in zip(*map(np.ravel, np.meshgrid(*map(range, n_subterrains), indexing="ij"))):
        subterrain = tu.SubTerrain(
            width=subterrain_rows,
            length=subterrain_cols,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
        )
        if not randomize:
            saved_state = np.random.get_state()
            np.random.seed(0)

        term_option: TerrainTermOptions = terrain_generators[i][j]
        term = TERRAIN_GENERATOR_REGISTRY.get(term_option.name)(options=term_option)
        subterrain = term.compute(subterrain)

        if not randomize:
            np.random.set_state(saved_state)

        data = subterrain.height_field_raw
        subterrain_heightfield = heightfield[
            i * (subterrain_rows - 1) : (i + 1) * (subterrain_rows - 1) + 1,
            j * (subterrain_cols - 1) : (j + 1) * (subterrain_cols - 1) + 1,
        ]
        subterrain_heightfield[:] = np.maximum(subterrain_heightfield, data)

    return heightfield


class TerrainGenerator(ConfigurableMixin):
    """
    Terrain generator for Eden, wrapping Genesis' terrain utils.

    Note: Uses unparameterized ConfigurableMixin to avoid a circular import
    with eden.options.entities. _options_class_ is resolved lazily.
    """

    @classmethod
    def configure(cls, **kwargs):
        if cls._options_class_ is None:
            from eden.options.entities import TerrainTermOptions

            cls._options_class_ = TerrainTermOptions
        return super().configure(**kwargs)

    def __init__(self, options: TerrainTermOptions):
        ConfigurableMixin.__init__(self, options=options)

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class FlatTerrain(TerrainGenerator):
    """Flat terrain generator."""

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        subterrain.height_field_raw *= 0.0
        return subterrain
