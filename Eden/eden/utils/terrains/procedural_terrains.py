"""Procedural box/Perlin terrain generators."""

from __future__ import annotations

import numpy as np
from genesis.ext.isaacgym import terrain_utils as tu
from genesis.typing import Vec2FType, Vec2IType

from eden.utils.terrains.base import TERRAIN_GENERATOR_REGISTRY, TerrainGenerator


def _perlin_2d(shape: Vec2IType, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a single octave of 2D Perlin-like gradient noise on a grid."""
    rows, cols = shape
    # Number of grid cells
    grid_r = max(int(np.ceil(rows / scale)), 1)
    grid_c = max(int(np.ceil(cols / scale)), 1)

    # Random unit gradient vectors at each grid vertex
    angles = rng.uniform(0, 2 * np.pi, (grid_r + 1, grid_c + 1))
    grad_x = np.cos(angles)
    grad_y = np.sin(angles)

    # Pixel coordinates mapped to grid space
    yr = np.linspace(0, grid_r, rows, endpoint=False)
    xc = np.linspace(0, grid_c, cols, endpoint=False)
    xc, yr = np.meshgrid(xc, yr)

    # Integer grid cell indices
    y0 = yr.astype(int)
    x0 = xc.astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    # Fractional offsets within each cell
    fy = yr - y0
    fx = xc - x0

    # Smoothstep fade
    uy = fy * fy * (3 - 2 * fy)
    ux = fx * fx * (3 - 2 * fx)

    # Dot products between gradient and distance vectors at four corners
    d00 = grad_x[y0, x0] * fx + grad_y[y0, x0] * fy
    d10 = grad_x[y1, x0] * (fx) + grad_y[y1, x0] * (fy - 1)
    d01 = grad_x[y0, x1] * (fx - 1) + grad_y[y0, x1] * fy
    d11 = grad_x[y1, x1] * (fx - 1) + grad_y[y1, x1] * (fy - 1)

    # Bilinear interpolation
    v0 = d00 * (1 - ux) + d01 * ux
    v1 = d10 * (1 - ux) + d11 * ux
    return v0 * (1 - uy) + v1 * uy


@TERRAIN_GENERATOR_REGISTRY.register()
class PerlinNoiseTerrain(TerrainGenerator):
    """
    Fractal Perlin noise terrain with layered octaves for natural-looking surfaces.

    Parameters
    ----------
    octaves : int
        Number of noise octaves to layer. More octaves add finer detail.
    persistence : float
        Amplitude multiplier per octave (controls roughness).
    lacunarity : float
        Frequency multiplier per octave (controls detail density).
    scale : float
        Base scale of the noise (larger = smoother features).
    amplitude : float
        Overall height amplitude in vertical_scale units.
    seed : int
        Random seed for reproducible noise. Use -1 for random.
    """

    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    scale: float = 20.0
    amplitude: float = 1.0
    seed: int = -1

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        rows, cols = subterrain.height_field_raw.shape
        seed = self.seed if self.seed >= 0 else np.random.randint(0, 2**31)
        rng = np.random.default_rng(seed)

        noise = np.zeros((rows, cols), dtype=np.float64)
        freq = 1.0
        amp = 1.0
        max_amp = 0.0

        for _ in range(self.octaves):
            noise += amp * _perlin_2d((rows, cols), self.scale / freq, rng)
            max_amp += amp
            amp *= self.persistence
            freq *= self.lacunarity

        # Normalize to [-1, 1] then scale
        noise /= max_amp
        subterrain.height_field_raw[:] = (noise * self.amplitude / subterrain.vertical_scale).astype(
            subterrain.height_field_raw.dtype
        )
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class GapTerrain(TerrainGenerator):
    """
    Terrain with parallel gaps (chasms) across the surface.

    Parameters
    ----------
    gap_width : float
        Width of each gap in meters.
    gap_depth : float
        Depth of each gap in meters.
    platform_width : float
        Width of flat platform between gaps in meters.
    num_gaps : int
        Number of gaps to place across the terrain.
    """

    gap_width: float = 0.3
    gap_depth: float = 1.0
    platform_width: float = 1.0
    num_gaps: int = 4

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows = hf.shape[0]
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        # Start with flat terrain
        hf[:] = 0

        gap_w_cells = max(int(self.gap_width / hs), 1)
        platform_w_cells = max(int(self.platform_width / hs), 1)
        depth_units = int(self.gap_depth / vs)

        # Distribute gaps evenly along rows
        stride = gap_w_cells + platform_w_cells
        total_width = self.num_gaps * stride
        start = max((rows - total_width) // 2, 0)

        for g in range(self.num_gaps):
            gap_start = start + g * stride + platform_w_cells
            gap_end = min(gap_start + gap_w_cells, rows)
            if gap_start < rows:
                hf[gap_start:gap_end, :] = -depth_units

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class PitTerrain(TerrainGenerator):
    """
    Terrain with rectangular pits at random positions.

    Parameters
    ----------
    pit_depth : float
        Depth of each pit in meters.
    pit_size_range : tuple of float
        Min and max side length of pits in meters.
    num_pits : int
        Number of pits to place on the terrain.
    platform_size : float
        Size of a safe flat zone in the center in meters. Set 0 to disable.
    """

    pit_depth: float = 0.5
    pit_size_range: Vec2FType = (0.5, 1.5)
    num_pits: int = 8
    platform_size: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        depth_units = int(self.pit_depth / vs)

        # Safe platform in center
        platform_cells = int(self.platform_size / hs)
        center_r, center_c = rows // 2, cols // 2
        safe_r_min = center_r - platform_cells // 2
        safe_r_max = center_r + platform_cells // 2
        safe_c_min = center_c - platform_cells // 2
        safe_c_max = center_c + platform_cells // 2

        for _ in range(self.num_pits):
            size_cells = np.random.randint(
                int(self.pit_size_range[0] / hs),
                max(
                    int(self.pit_size_range[1] / hs),
                    int(self.pit_size_range[0] / hs) + 1,
                )
                + 1,
            )
            r0 = np.random.randint(0, max(rows - size_cells, 1))
            c0 = np.random.randint(0, max(cols - size_cells, 1))
            r1 = min(r0 + size_cells, rows)
            c1 = min(c0 + size_cells, cols)

            # Skip if overlaps safe platform
            if self.platform_size > 0:
                if r1 > safe_r_min and r0 < safe_r_max and c1 > safe_c_min and c0 < safe_c_max:
                    continue

            hf[r0:r1, c0:c1] = -depth_units

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class RidgeTerrain(TerrainGenerator):
    """
    Terrain with parallel sharp ridges.

    Parameters
    ----------
    ridge_height : float
        Height of each ridge in meters.
    ridge_width : float
        Width of each ridge in meters.
    ridge_spacing : float
        Spacing between ridges in meters.
    """

    ridge_height: float = 0.15
    ridge_width: float = 0.1
    ridge_spacing: float = 0.5

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows = hf.shape[0]
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        height_units = int(self.ridge_height / vs)
        ridge_w_cells = max(int(self.ridge_width / hs), 1)
        spacing_cells = max(int(self.ridge_spacing / hs), 1)
        stride = ridge_w_cells + spacing_cells

        row = spacing_cells // 2
        while row < rows:
            end = min(row + ridge_w_cells, rows)
            # Triangular cross-section: peak in center of ridge
            for r in range(row, end):
                mid = (row + end) / 2.0
                dist = abs(r - mid)
                half_w = (end - row) / 2.0
                if half_w > 0:
                    frac = 1.0 - dist / half_w
                else:
                    frac = 1.0
                hf[r, :] = int(height_units * frac)
            row += stride

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class CraterTerrain(TerrainGenerator):
    """
    Terrain with smooth circular crater depressions.

    Parameters
    ----------
    crater_depth : float
        Maximum depth of craters in meters.
    crater_radius_range : tuple of float
        Min and max radius of craters in meters.
    num_craters : int
        Number of craters to place on the terrain.
    """

    crater_depth: float = 0.3
    crater_radius_range: Vec2FType = (0.5, 2.0)
    num_craters: int = 6

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        depth_units = int(self.crater_depth / vs)

        # Pre-compute row and col coordinate arrays
        row_coords = np.arange(rows)
        col_coords = np.arange(cols)

        for _ in range(self.num_craters):
            radius_cells = np.random.uniform(
                self.crater_radius_range[0] / hs,
                self.crater_radius_range[1] / hs,
            )
            cr = np.random.randint(0, rows)
            cc = np.random.randint(0, cols)

            # Compute distance from crater center for all cells
            dr = row_coords - cr
            dc = col_coords - cc
            dist_sq = dr[:, None] ** 2 + dc[None, :] ** 2
            r_sq = radius_cells**2

            # Smooth cosine depression inside radius
            mask = dist_sq < r_sq
            dist = np.sqrt(dist_sq)
            depression = np.zeros_like(hf, dtype=np.float64)
            depression[mask] = -depth_units * 0.5 * (1 + np.cos(np.pi * dist[mask] / radius_cells))

            # Accumulate (craters can overlap)
            hf[:] = np.minimum(hf, hf + depression.astype(hf.dtype))

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class NarrowBeamsTerrain(TerrainGenerator):
    """Radial beams extending from a central platform across a pit.

    Inspired by mjlab's BoxNarrowBeamsTerrainCfg.

    Parameters
    ----------
    num_beams : int
        Number of beams radiating from the center.
    beam_width : float
        Width of each beam in meters.
    beam_height : float
        Height of beams above the pit floor in meters.
    platform_width : float
        Side length of the central platform in meters.
    floor_depth : float
        Depth of the surrounding pit in meters.
    """

    num_beams: int = 16
    beam_width: float = 0.3
    beam_height: float = 0.2
    platform_width: float = 1.5
    floor_depth: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        floor_units = -int(self.floor_depth / vs)
        beam_units = int(self.beam_height / vs)
        beam_w_cells = max(int(self.beam_width / hs / 2), 1)
        platform_cells = int(self.platform_width / hs / 2)

        # Start with pit floor
        hf[:] = floor_units

        cr, cc = rows // 2, cols // 2

        # Central platform
        hf[
            cr - platform_cells : cr + platform_cells,
            cc - platform_cells : cc + platform_cells,
        ] = beam_units

        # Radial beams
        row_coords = np.arange(rows) - cr
        col_coords = np.arange(cols) - cc

        for i in range(self.num_beams):
            angle = 2 * np.pi * i / self.num_beams
            dx = np.cos(angle)
            dy = np.sin(angle)

            # For each cell, compute perpendicular distance to the beam line
            # and distance along beam direction
            perp_dist = np.abs(row_coords[:, None] * dx - col_coords[None, :] * dy)
            along_dist = row_coords[:, None] * dy + col_coords[None, :] * dx

            mask = (perp_dist <= beam_w_cells) & (along_dist >= 0)
            hf[mask] = beam_units

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class RandomGridTerrain(TerrainGenerator):
    """Grid of cells with random heights, creating a blocky uneven surface.

    Inspired by mjlab's BoxRandomGridTerrainCfg.

    Parameters
    ----------
    grid_width : float
        Side length of each grid cell in meters.
    grid_height_range : tuple of float
        Min and max height of grid cells in meters.
    platform_width : float
        Side length of the central flat platform in meters.
    """

    grid_width: float = 0.5
    grid_height_range: Vec2FType = (0.0, 0.3)
    platform_width: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        cell_size = max(int(self.grid_width / hs), 1)
        platform_cells = int(self.platform_width / hs / 2)
        cr, cc = rows // 2, cols // 2

        n_cells_r = rows // cell_size
        n_cells_c = cols // cell_size

        for gi in range(n_cells_r):
            for gj in range(n_cells_c):
                r0 = gi * cell_size
                c0 = gj * cell_size
                r1 = min(r0 + cell_size, rows)
                c1 = min(c0 + cell_size, cols)

                height_m = np.random.uniform(*self.grid_height_range)
                hf[r0:r1, c0:c1] = int(height_m / vs)

        # Flat central platform
        hf[
            cr - platform_cells : cr + platform_cells,
            cc - platform_cells : cc + platform_cells,
        ] = 0

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class RandomBoxesTerrain(TerrainGenerator):
    """Randomly scattered raised boxes on a flat surface.

    Inspired by mjlab's BoxRandomSpreadTerrainCfg.

    Parameters
    ----------
    num_boxes : int
        Number of boxes to scatter.
    box_size_range : tuple of float
        Min and max side length of boxes in meters.
    box_height_range : tuple of float
        Min and max height of boxes in meters.
    platform_width : float
        Side length of the safe central platform in meters.
    """

    num_boxes: int = 40
    box_size_range: Vec2FType = (0.3, 1.0)
    box_height_range: Vec2FType = (0.05, 0.5)
    platform_width: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        platform_cells = int(self.platform_width / hs / 2)
        cr, cc = rows // 2, cols // 2
        safe_r_min = cr - platform_cells
        safe_r_max = cr + platform_cells
        safe_c_min = cc - platform_cells
        safe_c_max = cc + platform_cells

        for _ in range(self.num_boxes):
            w = np.random.uniform(*self.box_size_range)
            h_m = np.random.uniform(*self.box_height_range)
            w_cells = max(int(w / hs), 1)
            h_units = int(h_m / vs)

            r0 = np.random.randint(0, max(rows - w_cells, 1))
            c0 = np.random.randint(0, max(cols - w_cells, 1))
            r1 = min(r0 + w_cells, rows)
            c1 = min(c0 + w_cells, cols)

            # Skip if overlaps safe platform
            if r1 > safe_r_min and r0 < safe_r_max and c1 > safe_c_min and c0 < safe_c_max:
                continue

            # Boxes stack (take max height)
            hf[r0:r1, c0:c1] = np.maximum(hf[r0:r1, c0:c1], h_units)

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class TiltedGridTerrain(TerrainGenerator):
    """Grid of cells with tilted surfaces creating uneven, sloped patches.

    Inspired by mjlab's BoxTiltedGridTerrainCfg.

    Parameters
    ----------
    grid_width : float
        Side length of each grid cell in meters.
    tilt_range : float
        Maximum tilt angle in degrees.
    height_range : float
        Additional height variation in meters.
    platform_width : float
        Side length of the central flat platform in meters.
    """

    grid_width: float = 1.0
    tilt_range: float = 15.0
    height_range: float = 0.1
    platform_width: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        cell_size = max(int(self.grid_width / hs), 1)
        platform_cells = int(self.platform_width / hs / 2)
        cr, cc = rows // 2, cols // 2

        n_cells_r = rows // cell_size
        n_cells_c = cols // cell_size

        # Local coordinates within a cell [0, 1]
        local_r = np.linspace(0, 1, cell_size)
        local_c = np.linspace(0, 1, cell_size)
        lr, lc = np.meshgrid(local_r, local_c, indexing="ij")

        for gi in range(n_cells_r):
            for gj in range(n_cells_c):
                r0 = gi * cell_size
                c0 = gj * cell_size
                r1 = min(r0 + cell_size, rows)
                c1 = min(c0 + cell_size, cols)
                actual_r = r1 - r0
                actual_c = c1 - c0

                # Random tilt direction and magnitude
                tilt_deg = np.random.uniform(0, self.tilt_range)
                tilt_rad = np.radians(tilt_deg)
                tilt_dir = np.random.uniform(0, 2 * np.pi)

                # Height offset from base
                base_h = np.random.uniform(0, self.height_range)

                # Tilted plane: height = base + tan(tilt) * (dx*cos + dy*sin) * cell_width
                slope = np.tan(tilt_rad) * self.grid_width
                tile = lr[:actual_r, :actual_c] * np.cos(tilt_dir) + lc[:actual_r, :actual_c] * np.sin(tilt_dir)
                heights = (base_h + slope * tile) / vs

                hf[r0:r1, c0:c1] = heights.astype(hf.dtype)

        # Flat central platform
        hf[
            cr - platform_cells : cr + platform_cells,
            cc - platform_cells : cc + platform_cells,
        ] = 0

        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class NestedRingsTerrain(TerrainGenerator):
    """Concentric rectangular rings forming a nested obstacle pattern.

    Inspired by mjlab's BoxNestedRingsTerrainCfg.

    Parameters
    ----------
    num_rings : int
        Number of concentric rings.
    ring_width : float
        Thickness of each ring in meters.
    gap_width : float
        Space between rings in meters.
    height_range : tuple of float
        Min and max height of rings in meters.
    platform_width : float
        Side length of the central platform in meters.
    """

    num_rings: int = 5
    ring_width: float = 0.4
    gap_width: float = 0.2
    height_range: Vec2FType = (0.1, 0.4)
    platform_width: float = 1.0

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        hf = subterrain.height_field_raw
        rows, cols = hf.shape
        hs = subterrain.horizontal_scale
        vs = subterrain.vertical_scale

        hf[:] = 0
        cr, cc = rows // 2, cols // 2
        platform_cells = int(self.platform_width / hs / 2)
        ring_w = max(int(self.ring_width / hs), 1)
        gap_w = max(int(self.gap_width / hs), 1)
        stride = ring_w + gap_w

        # Precompute Chebyshev distances from center
        dr = np.abs(np.arange(rows) - cr)
        dc = np.abs(np.arange(cols) - cc)
        dist_r = dr[:, None]  # (rows, 1)
        dist_c = dc[None, :]  # (1, cols)

        for i in range(self.num_rings):
            h_m = np.random.uniform(*self.height_range)
            h_units = int(h_m / vs)

            outer = platform_cells + (i + 1) * stride
            inner = outer - ring_w

            # Rectangular ring: inside outer box AND touching at least one edge of inner box
            in_outer = (dist_r < outer) & (dist_c < outer)
            in_inner = (dist_r < inner) & (dist_c < inner)
            mask = in_outer & ~in_inner

            hf[mask] = h_units

        return subterrain
