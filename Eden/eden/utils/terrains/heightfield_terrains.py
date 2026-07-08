"""Height-field terrain generators (fractal, sloped, obstacles, ...)."""

from __future__ import annotations

from genesis.ext.isaacgym import terrain_utils as tu

from eden.utils.terrains.base import TERRAIN_GENERATOR_REGISTRY, TerrainGenerator


@TERRAIN_GENERATOR_REGISTRY.register()
class FractalTerrain(TerrainGenerator):
    """
    Fractal terrain generator.

    Parameters
    ----------
    levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
    scale (float, optional): scales vertical variation. Defaults to 1.0.
    """

    levels: int = 8
    scale: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.fractal_terrain(
            subterrain,
            levels=self.levels,
            scale=self.scale,
        )


@TERRAIN_GENERATOR_REGISTRY.register()
class RandomUniformTerrain(TerrainGenerator):
    min_height: float = -0.1
    max_height: float = 0.1
    step: float = 0.1
    downsampled_scale: float = 0.5

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.random_uniform_terrain(
            subterrain,
            min_height=self.min_height,
            max_height=self.max_height,
            step=self.step,
            downsampled_scale=self.downsampled_scale,
        )


@TERRAIN_GENERATOR_REGISTRY.register()
class SlopedTerrain(TerrainGenerator):
    slope: float = -0.5

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.sloped_terrain(subterrain, slope=self.slope)


@TERRAIN_GENERATOR_REGISTRY.register()
class PyramidSlopedTerrain(TerrainGenerator):
    slope: float = -0.1

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.pyramid_sloped_terrain(subterrain, slope=self.slope)


@TERRAIN_GENERATOR_REGISTRY.register()
class DiscreteObstaclesTerrain(TerrainGenerator):
    max_height: float = 0.05
    min_size: float = 1.0
    max_size: float = 5.0
    num_rects: int = 20

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.discrete_obstacles_terrain(
            subterrain,
            max_height=self.max_height,
            min_size=self.min_size,
            max_size=self.max_size,
            num_rects=self.num_rects,
        )


@TERRAIN_GENERATOR_REGISTRY.register()
class WaveTerrain(TerrainGenerator):
    num_waves: float = 2.0
    amplitude: float = 0.1

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.wave_terrain(
            subterrain,
            num_waves=self.num_waves,
            amplitude=self.amplitude,
        )


@TERRAIN_GENERATOR_REGISTRY.register()
class StairsTerrain(TerrainGenerator):
    step_width: float = 0.75
    step_height: float = -0.1

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.stairs_terrain(
            subterrain,
            step_width=self.step_width,
            step_height=self.step_height,
        )


@TERRAIN_GENERATOR_REGISTRY.register()
class PyramidStairsTerrain(TerrainGenerator):
    step_width: float = 0.75
    step_height: float = -0.1

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.pyramid_stairs_terrain(
            subterrain,
            step_width=self.step_width,
            step_height=self.step_height,
        )


@TERRAIN_GENERATOR_REGISTRY.register()
class SteppingStonesTerrain(TerrainGenerator):
    stone_size: float = 1.0
    stone_distance: float = 0.25
    max_height: float = 0.2
    platform_size: float = 0.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return tu.stepping_stones_terrain(
            subterrain,
            stone_size=self.stone_size,
            stone_distance=self.stone_distance,
            max_height=self.max_height,
            platform_size=self.platform_size,
        )
