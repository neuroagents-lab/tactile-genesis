"""Mesh helpers: link meshes, polygon sampling, support-surface extraction."""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from dataclasses import dataclass

import math
from genesis.utils.geom import trans_quat_to_T
import numpy as np
import shapely
import torch
import trimesh
from trimesh.transformations import unit_vector

if TYPE_CHECKING:
    # Importing RigidLink eagerly pulls genesis.utils.array_class, which
    # requires gs.init() to have run. Defer to type-check only.
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


def _polygon_with_holes(
    polygon: shapely.geometry.Polygon,
    holes: Sequence[shapely.geometry.Polygon] | None,
) -> shapely.geometry.Polygon:
    """Return ``polygon`` with optional inner ring(s) carved out of it."""
    polygon_holes = list(holes) if holes is not None else []
    if polygon_holes:
        return shapely.geometry.Polygon(
            polygon.exterior.coords,
            holes=[hole.exterior.coords for hole in polygon_holes],
        )
    return shapely.geometry.Polygon(polygon.exterior.coords)


@dataclass
class SupportData:
    polygon: shapely.geometry.Polygon
    transform: np.ndarray


@dataclass
class ExtendedSupportData(SupportData):
    clearance: float = math.inf
    transform: torch.Tensor | None = None
    normal: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    sample_points: torch.Tensor | None = None  # [n_points, 2]
    num_sample_points: int = 0  # n_points


def get_link_mesh(link: RigidLink, use_visual_mesh: bool = False) -> trimesh.Trimesh:
    """
    Get the mesh of a link.

    Parameters
    ----------
    link : RigidLink
        Link to get the mesh of
    use_visual_mesh : bool
        Whether to use the visual mesh or the collision mesh

    Returns
    -------
    trimesh.Trimesh:
        Combined mesh of the link
    """
    meshes = []
    if use_visual_mesh:
        geoms = link.vgeoms
    else:
        geoms = link.geoms
    for i, geom in enumerate(geoms):
        # Use init_pos and init_quat for canonical pose (for articulated objects)
        # These represent the geom's pose in the link's local frame
        geom_pos = torch.from_numpy(geom.init_pos).to(torch.float32)
        geom_quat = torch.from_numpy(geom.init_quat).to(torch.float32)
        T = trans_quat_to_T(geom_pos.unsqueeze(0), geom_quat.unsqueeze(0))
        if T.ndim == 3:
            T = T[0]  # NOTE: we use the canonical space so batch can be ignored
        mesh = geom.get_trimesh().copy()  # NOTE: avoid in-place write
        mesh.apply_transform(T)
        meshes.append(mesh)
    combined_mesh = trimesh.util.concatenate(meshes)

    return combined_mesh


def rejection_sample_uniform(
    polygon: shapely.geometry.Polygon,
    num_points: int,
    scale: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Random sample points within the polygon.

    NOTE: resulting points might vary

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to sample from
    num_points : int
        Number of points to sample
    scale : np.ndarray
        Scale for the randomly generated points
    bias : np.ndarray
        Bias for the randomly generated points

    Returns
    -------
    tuple[np.ndarray, int]:
        Points sampled from the polygon and the number of points
    """
    points = scale * np.random.rand(num_points, 2) + bias
    mask = shapely.contains_xy(polygon, *points.T)
    hit = points[mask]
    hit_count = len(hit)
    return hit, hit_count


def rejection_sample_gaussian(
    polygon: shapely.geometry.Polygon,
    num_points: int,
    offset: float | np.ndarray = 0.0,
    std: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Random sample points within the polygon.

    NOTE: resulting points might vary

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to sample from
    num_points : int
        Number of points to sample
    offset : float
        Offset for the randomly generated points
    std : float
        Standard deviation for the randomly generated points

    Returns
    -------
    tuple[np.ndarray, int]:
        Points sampled from the polygon and the number of points
    """
    points = np.random.normal(np.array(polygon.centroid.coords) + offset, std, size=(num_points, 2))
    mask = shapely.contains_xy(polygon, *points.T)
    hit = points[mask]
    hit_count = len(hit)
    return hit, hit_count


def sample_polygon_grid(
    polygon: shapely.geometry.Polygon,
    grid_size: float,
    holes: Sequence[shapely.geometry.Polygon] | None = None,
) -> np.ndarray:
    """
    Sample points from a polygon using a grid.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to sample from
    grid_size : float
        Size of the grid
    holes : Sequence[shapely.geometry.Polygon], optional
        Holes in the polygon

    Returns
    -------
    np.ndarray:
        Points sampled from the polygon
    """
    # get size of bounding box
    bounds = np.reshape(polygon.bounds, (2, 2))
    scale = np.ptp(bounds, axis=0)
    bias = bounds[0]
    nx = int(scale[0] / grid_size)
    ny = int(scale[1] / grid_size)

    polygon = _polygon_with_holes(polygon, holes)

    # generate grid points
    x = np.linspace(0, 1, nx) * scale[0] + bias[0]
    y = np.linspace(0, 1, ny) * scale[1] + bias[1]
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    mask = shapely.contains_xy(polygon, *points.T)
    hit = points[mask]
    return hit


def sample_polygon_uniform(
    polygon: shapely.geometry.Polygon,
    count: int,
    holes: Sequence[shapely.geometry.Polygon] | None = None,
    factor: float = 1.5,
    max_iter: int = 10,
) -> np.ndarray:
    """Use rejection sampling to generate random points inside a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon that will contain points
    count : int
        Number of points to return
    holes : Sequence[shapely.geometry.Polygon], optional
        Holes to carve out of the polygon before sampling.
    factor : float
        Increase the initial count by this factor for rejection sampling
    max_iter : int
        Maximum number of intersection checks is: > count * factor * max_iter

    Returns
    -------
    np.ndarray:
        Random points inside polygon where n <= count
    """
    # get size of bounding box
    bounds = np.reshape(polygon.bounds, (2, 2))
    extents = np.ptp(bounds, axis=0)

    # how many points to check per loop iteration
    per_loop = int(count * factor)
    polygon = _polygon_with_holes(polygon, holes)

    hit, hit_count = rejection_sample_uniform(polygon, num_points=per_loop, scale=extents, bias=bounds[0])

    if hit_count >= count:
        return hit[:count]

    hits = [hit]
    # if we have to do iterations loop here slowly
    for _ in range(max_iter):
        hit, hit_count_ = rejection_sample_uniform(polygon, num_points=per_loop, scale=extents, bias=bounds[0])

        hits.append(hit)
        hit_count += hit_count_

        if hit_count > count:
            break

    # stack the hits into an (n,2) array and truncate
    return np.vstack(hits)[:count]


def sample_polygon_gaussian(
    polygon: shapely.geometry.Polygon,
    count: int,
    holes: Sequence[shapely.geometry.Polygon] | None = None,
    factor: float = 1.5,
    max_iter: int = 10,
    offset: float | np.ndarray = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """Use rejection sampling to generate Gaussian-distributed points inside a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon that will contain points
    count : int
        Number of points to return
    holes : Sequence[shapely.geometry.Polygon], optional
        Holes to carve out of the polygon before sampling.
    factor : float
        Increase the initial count by this factor for rejection sampling
    max_iter : int
        Maximum number of intersection checks is: > count * factor * max_iter
    offset : float or np.ndarray
        Offset added to the polygon centroid to center the Gaussian.
    std : float
        Standard deviation of the Gaussian used for sampling.

    Returns
    -------
    np.ndarray:
        Random points inside polygon where n <= count
    """
    # how many points to check per loop iteration
    per_loop = int(count * factor)
    polygon = _polygon_with_holes(polygon, holes)

    hit, hit_count = rejection_sample_gaussian(polygon, num_points=per_loop, offset=offset, std=std)

    if hit_count >= count:
        return hit[:count]

    hits = [hit]
    # if we have to do iterations loop here slowly
    for _ in range(max_iter):
        hit, hit_count_ = rejection_sample_gaussian(polygon, num_points=per_loop, offset=offset, std=std)

        hits.append(hit)
        hit_count += hit_count_

        if hit_count > count:
            break

    # stack the hits into an (n,2) array and truncate
    return np.vstack(hits)[:count]


def _raycasts(
    mesh: trimesh.Trimesh, origins: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast rays against a mesh and return hit locations, ray indices, and face indices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to use for ray casts.
    origins : np.ndarray
        Origins of rays.
    directions : np.ndarray
        Directions of rays.

    Returns
    -------
    np.ndarray:
        Locations of intersections, where rays hit the mesh.
    np.ndarray:
        Ray indices, mapping each returned location to a ray.
    np.ndarray:
        Array of triangle (face) indexes.
    """
    assert len(origins) == len(directions)
    return mesh.ray.intersects_location(origins, directions, multiple_hits=False)


def extrude_surface(
    polygon: shapely.geometry.Polygon,
    polygon_transform: np.ndarray,
    mesh: trimesh.Trimesh,
    rays_per_area: int = 100,
    extrude_direction: np.ndarray = np.array([0, 0, 1]),
    distance_above_support: float = 1e-3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extrudes a support polygon until collision.

    Parameters
    ----------
    polygon : trimesh.path.polygons.Polygon
        polygon to extrude
    polygon_transform : np.array
        transformation matrix of the polygon
    mesh : trimesh.Trimesh, optional
        Defaults to the scene's mesh.
    rays_per_area : int, optional
        For testing collisions to extrude support polygons.
    extrude_direction : np.ndarray, optional
        Direction to extrude
    distance_above_support : float, optional
        Support polyhedra are above the support polygon by this amount.

    Returns
    -------
        list[np.ndarray]: List of ray origins on the surface.
        list[np.ndarray]: List of ray intersections on the mesh.
    """
    # for each support polygon, sample raycasts to determine maximum height of extrusion in direction of gravity
    num_rays = int(rays_per_area * polygon.area) + 1
    pts = sample_polygon_uniform(polygon, count=num_rays)

    if len(pts) == 0:
        return [], []

    points = np.c_[pts, np.zeros(len(pts)) + distance_above_support]
    xyz_coords = trimesh.transform_points(points=points, matrix=polygon_transform)

    intersections, ray_ids, _ = _raycasts(
        mesh=mesh,
        origins=xyz_coords,
        directions=np.array(len(pts) * [list(unit_vector(extrude_direction))]),
    )
    if len(intersections) == 0:
        return [], []

    origins = xyz_coords[ray_ids]
    return origins, intersections


def get_exposed_support_surfaces(
    mesh: trimesh.Trimesh,
    local_gravity: np.ndarray = np.array([0, 0, -1]),
    precision: int = 3,
    gravity_tolerance: float = 0.1,
    min_area: float = 0.01,
    erosion_distance: float = 0.0,
) -> list[SupportData]:
    """
    Get the polgons representing the support surface in the local gravity direction.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to get the support surfaces of
    local_gravity : np.ndarray
        Local gravity direction
    precision : int
        Precision for the polygon conversion
    gravity_tolerance : float
        Tolerance for the gravity direction
    min_area : float
        Minimum area for the support surfaces
    erosion_distance : float
        Distance to erode the support surfaces

    Returns
    -------
    list[SupportData]:
        List of support surfaces
    """
    support_facet_indices = []
    for idx in np.argsort(mesh.facets_area)[::-1]:
        if mesh.facets_area[idx] <= min_area:
            break
        if np.isclose(
            np.dot(unit_vector(mesh.facets_normal[idx]), (-unit_vector(local_gravity))),
            1.0,
            atol=gravity_tolerance,
        ):
            support_facet_indices.append(idx)

    support_surfaces = []
    for index in support_facet_indices:
        normal = mesh.facets_normal[index]
        origin = mesh.facets_origin[index]

        facet_T = trimesh.geometry.plane_transform(origin, normal)
        facet_T_inv = trimesh.transformations.inverse_matrix(facet_T)

        # find boundary edges for the facet
        edges = mesh.edges_sorted.reshape((-1, 6))[mesh.facets[index]].reshape((-1, 2))
        group = trimesh.grouping.group_rows(edges, require_count=1, digits=precision)
        vertices = trimesh.transform_points(mesh.vertices, facet_T)[:, :2]

        # run the polygon conversion
        polygons = trimesh.path.polygons.edges_to_polygons(edges=edges[group], vertices=vertices)

        for polygon in polygons:
            if polygon.geom_type == "MultiPolygon":
                # This can be recursive!!
                polys = list(polygon.geoms)
            else:
                polys = [polygon]

            for uneroded_poly in polys:
                # erode to avoid object on edges
                eroded_poly = uneroded_poly.buffer(-erosion_distance)

                if eroded_poly.geom_type == "MultiPolygon":
                    eroded_polys = list(eroded_poly.geoms)
                else:
                    eroded_polys = [eroded_poly]

                for poly in eroded_polys:
                    if not poly.is_empty and poly.area > min_area:
                        support_surfaces.append(
                            SupportData(
                                polygon=poly,
                                transform=facet_T_inv,
                            )
                        )
    return support_surfaces


def AABB_to_polygon(AABB: np.ndarray) -> shapely.geometry.Polygon:
    """
    Convert an AABB to a polygon.

    Parameters
    ----------
    AABB : np.ndarray
        AABB to convert

    Returns
    -------
    shapely.geometry.Polygon:
        Polygon representing the AABB
    """
    if AABB.ndim == 3:
        AABB = AABB[0]
    min_x = AABB[0, 0]
    min_y = AABB[0, 1]
    max_x = AABB[1, 0]
    max_y = AABB[1, 1]

    coords = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    ]

    return shapely.geometry.Polygon(coords)
