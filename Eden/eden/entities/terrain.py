"""Terrain entity wrapper for height-field and procedural terrain morphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import genesis as gs

from eden.entities.rigid import RigidEntity


class Terrain(RigidEntity):
    """A special Entity that represents the terrain."""

    def __init__(self, env, options):
        super().__init__(env, options)
        if self._options.use_checker_surface:
            self.surface = self._create_checker_surface()

    def _create_morph_from_options(self):
        """Create terrain morph from TerrainOptions fields."""
        from eden.utils.terrains.base import generate_height_field

        opts = self._options
        n_subterrains = (
            len(opts.terrain_generators),
            len(opts.terrain_generators[0]),
        )

        height_field = generate_height_field(
            terrain_generators=opts.terrain_generators,
            subterrain_size=opts.subterrain_size,
            n_subterrains=n_subterrains,
            horizontal_scale=opts.horizontal_scale,
            vertical_scale=opts.vertical_scale,
            randomize=opts.randomize,
        )

        self.morph = gs.morphs.Terrain(
            subterrain_size=opts.subterrain_size,
            n_subterrains=n_subterrains,
            horizontal_scale=opts.horizontal_scale,
            vertical_scale=opts.vertical_scale,
            uv_scale=opts.uv_scale,
            randomize=opts.randomize,
            height_field=height_field,
            visualization=opts.visualization,
        )

    def _create_checker_surface(self):
        """Create a tiled checker surface based on terrain dimensions."""
        import os

        import numpy as np
        from genesis.utils.misc import get_assets_dir
        from PIL import Image

        opts = self._options
        n_subterrains = (
            len(opts.terrain_generators),
            len(opts.terrain_generators[0]),
        )
        x_scale = n_subterrains[0] * opts.subterrain_size[0] / 2
        y_scale = n_subterrains[1] * opts.subterrain_size[1] / 2
        checker_image = np.array(Image.open(os.path.join(get_assets_dir(), "textures/checker.png")))
        tiled_image = np.tile(checker_image, (int(x_scale), int(y_scale), 1))
        return gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_array=tiled_image,
            )
        )

    def pre_build(self) -> None:
        super().pre_build()
        (terrain_geom,) = self._entity.geoms

        assert "height_field" in terrain_geom.metadata
        height_field = terrain_geom.metadata["height_field"]
        self._height_field = torch.as_tensor(height_field, device=gs.device, dtype=gs.tc_float)
        assert self._height_field.ndim == 2, f"Height field must be 2D, but got {self._height_field.shape}"
        self._height_field *= self._entity.morph.vertical_scale

        # NOTE: reshape from (width, height) to (height, width) for grid_sample calculation
        # NOTE: we only need one copy since all environments share the same terrain
        self._height_field = self._height_field.T
        # TODO: heterogeneous terrain support
        self._height_field = self._height_field[None, None, :, :].expand(self._env.num_envs, -1, -1, -1)

    def post_build(self) -> None:
        super().post_build()

        self._origin = self._entity.get_pos()[0]

        # Calculate total terrain size from subterrain configuration
        subterrain_size = self._entity.morph.subterrain_size
        n_subterrains = self._entity.morph.n_subterrains

        total_x_size = subterrain_size[0] * n_subterrains[0]
        total_y_size = subterrain_size[1] * n_subterrains[1]
        self._size = (total_x_size, total_y_size)

        # Calculate bounds from origin and size
        x_min = self._origin[0]
        y_min = self._origin[1]
        x_max = x_min + total_x_size
        y_max = y_min + total_y_size
        self._bounds = (x_min, x_max, y_min, y_max)

        self._bounds_min_buffer = torch.tensor([x_min, y_min], device=gs.device, dtype=gs.tc_float)
        self._bounds_inv_range_buffer = torch.tensor(
            [1.0 / (x_max - x_min), 1.0 / (y_max - y_min)],
            device=gs.device,
            dtype=gs.tc_float,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the terrain limits as an ``(x_min, y_min, x_max, y_max)`` tuple.

        Returns
        -------
        tuple[float, float, float, float]
            The terrain bounds in ``xyxy`` format.
        """
        return self._bounds

    def get_height(self, grid: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Get the height of the terrain at the given xy coordinates.

        Parameters
        ----------
        grid : torch.Tensor, shape (B, h_out, w_out, 2)
            The grid of xy coordinates to sample the height from. The grid is expected to be in the range [-1, 1] if `normalize` is False.
        normalize : bool, optional
            Whether to normalize the coordinates to [-1, 1] inplace. Defaults to True.

        Returns
        -------
        height : torch.Tensor, shape (B, h_out, w_out)
            The height of the terrain at the given xy coordinates.
        """
        if normalize:
            # Normalize coordinates to [-1, 1] range expected by grid_sample
            # norm_xy = 2 * (xy - mins) / (maxs - mins) - 1
            grid = 2 * (grid - self._bounds_min_buffer) * self._bounds_inv_range_buffer - 1.0

        # Border padding mode isn't supported on Mac GPU (mps)
        # https://github.com/pytorch/pytorch/issues/125098
        interpolated = F.grid_sample(
            self._height_field,  # B, 1, height, width
            grid,  # B, h_out, w_out, 2
            mode="bilinear",
            padding_mode="border" if gs.device.type != "mps" else "zeros",
            align_corners=True,
        )  # B, 1, h_out, w_out

        return interpolated.squeeze(1)  # B, h_out, w_out
