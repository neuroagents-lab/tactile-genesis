"""Support-surface extraction mixin for placement queries (Taichi/Quadrants kernels).

:class:`SupportSurfaceMixin` extracts horizontal support polygons from an entity's
collision meshes and answers placement queries (see ``RigidEntity.place_on_to`` and
the samplers in :mod:`eden.managers.terms.events.placement`).

Gotcha: this module uses ``@qd.kernel`` (Quadrants/Taichi), so
``from __future__ import annotations`` **cannot** be used here — it would turn the
kernel's type annotations into strings and break kernel compilation.
"""

# NOTE: DO NOT use forward references (e.g., `from __future__ import annotations`) with quadrants (taichi) kernel
import math

import genesis as gs
import genesis.utils.geom as gu
import quadrants as qd

import numpy as np
import torch
import trimesh
from genesis.utils.geom import trans_quat_to_T

from genesis.engine.entities.base_entity import Entity
from eden.utils.geom import transform_by_R
from eden.utils.mesh import (
    SupportData,
    ExtendedSupportData,
    AABB_to_polygon,
    extrude_surface,
    get_exposed_support_surfaces,
    get_link_mesh,
    sample_polygon_uniform,
    sample_polygon_grid,
)


@qd.data_oriented
class SupportSurfaceMixin:
    """Mixin for support surface preparation. Used for penetration free object placement."""

    _entity: Entity

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def solver(self):
        return self._entity._solver

    @staticmethod
    def _select_support_links(entity_links, support_links_name):
        """Filter ``entity_links`` to the subset named in ``support_links_name``.

        ``support_links_name`` of ``None`` or empty means every link contributes
        (legacy behavior). Unknown names raise ``ValueError`` so a typo in the
        option surfaces at build time instead of silently falling back to "all
        links".
        """
        if not support_links_name:
            return list(entity_links)
        wanted = set(support_links_name)
        available = {link.name for link in entity_links}
        missing = wanted - available
        if missing:
            raise ValueError(
                f"support_links_name {sorted(missing)} not found on entity; available links: {sorted(available)}"
            )
        return [link for link in entity_links if link.name in wanted]

    @staticmethod
    def _attach_sample_points(
        d: ExtendedSupportData,
        *,
        pre_shrink: float,
        pre_sample_method: str,
        pre_num_sample_points: int,
        pre_grid_size: float,
        occlusion_mesh: trimesh.Trimesh | None = None,
        occlusion_gravity: np.ndarray | None = None,
    ) -> None:
        """Populate ``d.sample_points`` / ``d.num_sample_points`` from ``d.polygon``.

        Shared between fixed-base and floating-base branches of ``_prepare_support``.

        When ``occlusion_mesh`` is provided, sample points are additionally
        filtered to drop any whose anti-gravity ray-cast first hits something
        other than the support plane itself — i.e. points that sit under
        overhanging geometry. This is needed for multi-link assets like the
        Riverway franka station where a wide horizontal surface (the arm-cart
        bottom plate) is partially covered by a separate collision box bolted
        on top (the franka control electronics box). Without the filter, the
        polygon-level clearance check accepts the whole bottom plate as a
        support, and sample points inside the bolted-on box's footprint cause
        placed objects to penetrate.

        The filter only refines ``d.sample_points``. ``d.polygon`` keeps
        describing the full facet (including any occluded sub-region), so
        ``polygon.area`` over-reports the actually-placeable area when an
        overhang is present. Use ``num_sample_points`` (or
        ``len(sample_points)``) as the proxy for usable area rather than
        ``polygon.area``.
        """
        num_points = int(d.polygon.area * pre_num_sample_points)
        if num_points <= 0:
            return
        polygon = d.polygon.buffer(-pre_shrink) if pre_shrink > 0 else d.polygon
        if pre_sample_method == "uniform":
            sampled = sample_polygon_uniform(polygon=polygon, count=num_points)
        elif pre_sample_method == "grid":
            sampled = sample_polygon_grid(polygon=polygon, grid_size=pre_grid_size)
        else:
            raise ValueError(f"pre_sample_method must be 'uniform' or 'grid'; got {pre_sample_method!r}")

        if len(sampled) == 0:
            return

        if occlusion_mesh is not None and occlusion_gravity is not None:
            sampled = SupportSurfaceMixin._drop_occluded_samples(
                sampled,
                polygon_transform=d.transform,
                occlusion_mesh=occlusion_mesh,
                occlusion_gravity=occlusion_gravity,
            )
            if len(sampled) == 0:
                return

        d.sample_points = torch.from_numpy(sampled).to(dtype=gs.tc_float, device=gs.device)
        d.num_sample_points = len(d.sample_points)

    @staticmethod
    def _drop_occluded_samples(
        sampled_2d: np.ndarray,
        *,
        polygon_transform: np.ndarray | torch.Tensor,
        occlusion_mesh: trimesh.Trimesh,
        occlusion_gravity: np.ndarray,
        ray_origin_offset: float = 10.0,
        plane_tolerance: float = 1e-3,
    ) -> np.ndarray:
        """Return ``sampled_2d`` with overhang-occluded points removed.

        Strategy: convert each 2D polygon-local point to a 3D world point on
        the support plane, then cast a ray ANTI-PARALLEL to gravity starting
        ``ray_origin_offset`` metres further along ``-gravity``. The first
        face the ray hits identifies the topmost surface at that xy. A point
        is *kept* iff its ray hits at exactly ``ray_origin_offset`` (the
        polygon's own plane) within ``plane_tolerance``. Points whose ray
        hits closer (an overhang above the polygon) **or** further (a face
        below the polygon — happens when the polygon spans a hole) **or**
        misses entirely (mesh degeneracy / numerical edge case) are dropped.
        The fail-safe is "occluded": if we can't confirm the polygon plane
        is the topmost surface at that xy, we don't place objects there.

        This is correct even when the support polygon and an overhanging
        collision box share a coplanar boundary (the franka-station bug):
        the ray comes from far above, so it hits the overhanging box's TOP
        face — not the support plane underneath.

        NOTE on ``d.polygon``: this filter only refines the SAMPLE POINTS;
        it does **not** subtract the occluded region from the polygon
        geometry stored on ``ExtendedSupportData``. ``polygon.area`` keeps
        reporting the full detected facet area (which can be larger than
        the actually-placeable area). Downstream code that needs a
        "usable area" must derive it from ``len(sample_points)`` rather
        than ``polygon.area``.
        """
        gravity = np.asarray(occlusion_gravity, dtype=np.float64)
        gravity = gravity / np.linalg.norm(gravity)

        # Polygon-local 2D -> 3D (z=0 in polygon frame) -> world.
        local_3d = np.hstack([sampled_2d, np.zeros((len(sampled_2d), 1))])
        if isinstance(polygon_transform, torch.Tensor):
            T = polygon_transform.detach().cpu().numpy()
            if T.ndim == 3:
                T = T[0]
        else:
            T = np.asarray(polygon_transform)
        world_3d = trimesh.transform_points(local_3d, T)

        # Cast rays from far along -gravity, going +gravity (i.e. DOWN onto
        # the surface from above). The first hit tells us the highest face
        # at that xy.
        ray_origins = world_3d + (-gravity)[None, :] * ray_origin_offset
        ray_dirs = np.tile(gravity[None, :], (len(world_3d), 1))

        locations, ray_ids, _ = occlusion_mesh.ray.intersects_location(ray_origins, ray_dirs, multiple_hits=False)

        # Per-sample distance from the ray origin to its first hit (inf if
        # the ray missed everything — shouldn't happen for a point drawn
        # from a real polygon, but treat misses as occluded so a mesh
        # degeneracy at the polygon edge can't sneak a bad placement
        # through).
        hit_distance = np.full(len(world_3d), np.inf)
        if len(locations) > 0:
            distances = np.linalg.norm(locations - ray_origins[ray_ids], axis=1)
            hit_distance[ray_ids] = distances

        # Keep iff the first hit lands within `plane_tolerance` of the
        # expected polygon-plane distance:
        #   - exact match            -> the polygon IS the topmost face here  -> keep
        #   - hit too close (above)  -> an overhang sits above the polygon    -> drop
        #   - hit too far (below)    -> ray punched through a hole in polygon -> drop
        #   - inf (no hit at all)    -> mesh edge degeneracy                  -> drop
        keep_mask = np.isclose(hit_distance, ray_origin_offset, atol=plane_tolerance, rtol=0.0)
        return sampled_2d[keep_mask]

    def _prepare_support(
        self,
        distance_above_support: float = 1e-3,
        rays_per_area: int = 100,
        minimum_clearance: float = 0.05,
        *,
        pre_sample_points: bool = True,
        pre_shrink: float = 0.1,
        pre_sample_method: str = "uniform",  # "uniform" or "grid"
        pre_num_sample_points: int = 100,
        pre_grid_size: float = 0.01,
        support_links_name: list[str] | None = None,
    ):
        """Extract and cache the entity's support surface for later placement queries.

        Parameters
        ----------
        distance_above_support: float
            distance above the support surface to sample points
        rays_per_area: int
            number of rays to cast per area (rays/m^2)
        minimum_clearance: float
            minimum clearance to consider for support surface
        pre_sample_points: bool
            whether to pre-sample points from support surface
        pre_shrink: float
            fraction by which to shrink each support polygon inward before sampling,
            keeping sampled points away from surface edges
        pre_sample_method: str
            method used to pre-sample points, either ``"uniform"`` or ``"grid"``
        pre_num_sample_points: int
            number of points to pre-sample from support surface per area (points/m^2)
        pre_grid_size: float
            grid cell size (in meters) used when ``pre_sample_method`` is ``"grid"``
        support_links_name: list[str] | None
            Exact link names (URDF link / MJCF body names) whose geometry
            contributes to support extraction. ``None`` or empty means every
            link on the entity contributes (legacy behavior). Unknown names
            raise ``ValueError`` so a typo doesn't silently fall back to
            "all links".
        """
        self.distance_above_support = distance_above_support
        self.rays_per_area = rays_per_area
        self._support_data_list: list[ExtendedSupportData] = []

        if isinstance(self._entity._morph, gs.morphs.Plane):
            self._support_data_list.append(
                ExtendedSupportData(
                    polygon=AABB_to_polygon(self._entity.get_AABB()),
                    transform=torch.eye(4, device=gs.device, dtype=gs.tc_float).unsqueeze(0),
                    clearance=math.inf,
                    normal=torch.tensor([0, 0, 1], device=gs.device, dtype=gs.tc_float).unsqueeze(0),
                )
            )
            _T_reference = torch.eye(4, device=gs.device, dtype=gs.tc_float).unsqueeze(0)
            self._T_reference_inv = torch.linalg.inv(_T_reference)
            return

        _pos_save = self._entity.get_pos()
        _quat_save = self._entity.get_quat()
        _T_reference = trans_quat_to_T(_pos_save, _quat_save)
        self._T_reference_inv = torch.linalg.inv(_T_reference)

        links = self._select_support_links(self._entity.links, support_links_name)

        meshes = []
        for link in links:
            pos = link.get_pos(envs_idx=0).cpu().numpy()
            quat = link.get_quat(envs_idx=0).cpu().numpy()
            if pos.ndim == 2:
                pos = pos[0]
                quat = quat[0]
            T = trans_quat_to_T(pos, quat)
            combined = get_link_mesh(link)
            combined.apply_transform(T)
            meshes.append(combined)
        combined_mesh = trimesh.util.concatenate(meshes)

        if self.is_fixed_base:
            fixed_gravity = np.array([0, 0, -1])
            support_data = self._calculate_support(
                combined_mesh,
                minimum_clearance,
                require_roof=False,
                gravity=fixed_gravity,
            )
            for d in support_data:
                d.normal = torch.tensor([[0, 0, 1]], device=gs.device, dtype=gs.tc_float)
                if pre_sample_points:
                    self._attach_sample_points(
                        d,
                        pre_shrink=pre_shrink,
                        pre_sample_method=pre_sample_method,
                        pre_num_sample_points=pre_num_sample_points,
                        pre_grid_size=pre_grid_size,
                        occlusion_mesh=combined_mesh,
                        occlusion_gravity=fixed_gravity,
                    )
                self._support_data_list.append(d)
        else:
            # NOTE: calculate support from all 6 directions
            for gravity in torch.eye(3, device=gs.device):
                for direction in (1, -1):
                    gravity_np = (gravity * direction).cpu().numpy()
                    support_data = self._calculate_support(
                        combined_mesh,
                        minimum_clearance,
                        require_roof=False,
                        gravity=gravity_np,
                    )
                    for d in support_data:
                        d.normal = (-gravity * direction).unsqueeze(0)
                        if pre_sample_points:
                            self._attach_sample_points(
                                d,
                                pre_shrink=pre_shrink,
                                pre_sample_method=pre_sample_method,
                                pre_num_sample_points=pre_num_sample_points,
                                pre_grid_size=pre_grid_size,
                                occlusion_mesh=combined_mesh,
                                occlusion_gravity=gravity_np,
                            )
                        self._support_data_list.append(d)

    def _calculate_support(
        self,
        combined_mesh,
        minimum_clearance: float = 0.1,
        require_roof: bool = False,
        gravity: np.ndarray = np.array([0, 0, -1]),
        erosion_distance: float = 0.0,
    ) -> list[ExtendedSupportData]:
        """Perform support surface preparation in environment 0 (if batched).

        minimum_clearance
            minimum height available above the support
        require_roof
            if the support has to be covered (e.g., inside shelf, microwave, etc.)
        gravity:
            gravity direction in world coordinate system that support surface is holding against
        """
        # NOTE: we should do this in canonical space so that batch can be ignored
        support_data: list[SupportData] = get_exposed_support_surfaces(
            combined_mesh, gravity, erosion_distance=erosion_distance
        )

        _support_data_list: list[ExtendedSupportData] = []
        for d in support_data:
            origins, intersections = extrude_surface(
                d.polygon,
                d.transform,
                combined_mesh,
                rays_per_area=self.rays_per_area,
                extrude_direction=-gravity,
                distance_above_support=self.distance_above_support,
            )

            if len(intersections) == 0:
                # NOTE: open support
                if not require_roof:
                    _support_data_list.append(
                        ExtendedSupportData(
                            polygon=d.polygon,
                            transform=torch.from_numpy(d.transform)
                            .to(dtype=gs.tc_float, device=gs.device)
                            .unsqueeze(0),
                            clearance=math.inf,
                        )
                    )
            else:
                # NOTE: closed support
                clearance = np.min(np.linalg.norm(intersections - origins, axis=-1))
                if clearance > minimum_clearance:
                    _support_data_list.append(
                        ExtendedSupportData(
                            polygon=d.polygon,
                            transform=torch.from_numpy(d.transform)
                            .to(dtype=gs.tc_float, device=gs.device)
                            .unsqueeze(0),
                            clearance=clearance,
                        )
                    )

        return _support_data_list

    def current_support(
        self,
        minimum_clearance: float = 0.1,
        require_roof: bool = False,
        gravity: torch.Tensor = torch.tensor([0, 0, -1]),
        gravity_tolerance: float = 0.1,
        envs_idx: (slice | torch.Tensor) | None = None,
    ) -> list[ExtendedSupportData]:
        """
        Get the valid current support data at the entity's current pose.

        Parameters
        ----------
        minimum_clearance: float
            minimum clearance to consider for support surface
        require_roof: bool
            if the support has to be covered (e.g., inside shelf, microwave, etc.)
        gravity: torch.Tensor
            a normalized gravity vector indicating the gravity direction in world coordinate system
            that the support surface is holding against
        gravity_tolerance: float
            tolerance for gravity direction alignment
        envs_idx: None | slice | torch.Tensor, optional
            the indices of the environments to query. If None, all environments are considered. Defaults to None.
        """
        _pos_cur = self._entity.get_pos(envs_idx=envs_idx)
        _quat_cur = self._entity.get_quat(envs_idx=envs_idx)
        _T_cur = trans_quat_to_T(_pos_cur, _quat_cur)

        if envs_idx is not None and self._T_reference_inv.shape[0] != 1:
            T_revert = _T_cur @ self._T_reference_inv[envs_idx]
        else:
            T_revert = _T_cur @ self._T_reference_inv

        res: list[ExtendedSupportData] = []
        for d in self._support_data_list:
            if minimum_clearance > 0 or require_roof:
                if require_roof and d.clearance == math.inf:
                    continue
                if d.clearance < minimum_clearance:
                    continue

            normal = transform_by_R(d.normal, T_revert[..., :3, :3])
            normal = normal / torch.norm(normal, dim=-1, keepdim=True)

            # check if the normal is aligned with the opposite gravity direction
            normalized_gravity = gravity / torch.norm(gravity, dim=-1, keepdim=True)
            valid_mask = torch.isclose(
                (normal * -normalized_gravity).sum(dim=-1),
                torch.ones(normal.shape[0], device=gs.device, dtype=gs.tc_float),
                atol=gravity_tolerance,
            )
            res.append(
                ExtendedSupportData(
                    polygon=d.polygon,
                    transform=(
                        T_revert @ d.transform[envs_idx]
                        if envs_idx is not None and d.transform.shape[0] > 1
                        else T_revert @ d.transform
                    ),  # (B, 4, 4)
                    normal=normal,  # (B, 3)
                    valid_mask=valid_mask,  # (B, )
                    sample_points=d.sample_points,
                    num_sample_points=d.num_sample_points,
                )
            )
        return res

    @qd.kernel
    def _kernel_update_aabbs(
        self,
        geoms_pos: qd.Tensor,
        geoms_quat: qd.Tensor,
        geoms_init_AABB: qd.Tensor,
        geoms_aabb_min: qd.Tensor,
        geoms_aabb_max: qd.Tensor,
    ):
        """Update the AABB with an offset before the broad phase (custom override)."""
        qd.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in qd.ndrange(self.solver.n_geoms, self.solver._B):
            pos = geoms_pos[i_g, i_b]
            quat = geoms_quat[i_g, i_b]

            lower = gu.qd_vec3(qd.math.inf)
            upper = gu.qd_vec3(-qd.math.inf)
            for i_corner in range(8):
                corner_pos = gu.qd_transform_by_trans_quat(
                    geoms_init_AABB[i_g, i_corner],
                    pos,
                    quat,
                )
                lower = qd.min(lower, corner_pos)
                upper = qd.max(upper, corner_pos)

            geoms_aabb_min[i_g, i_b] = lower - 0.01
            geoms_aabb_max[i_g, i_b] = upper + 0.01

    @qd.kernel
    def _kernel_filter_detection(
        self,
        tensor: qd.types.ndarray(),
        mask: qd.types.ndarray(),
        n_broad_pairs: qd.Tensor,
        broad_collision_pairs: qd.Tensor,
        link_idx: qd.Tensor,
        entity_idx: qd.Tensor,
        geom_start: qd.Tensor,
        geom_end: qd.Tensor,
        geoms_aabb_min: qd.Tensor,
        geoms_aabb_max: qd.Tensor,
    ):
        qd.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_pair in range(n_broad_pairs[i_b]):
                i_ga = broad_collision_pairs[i_pair, i_b][0]
                i_gb = broad_collision_pairs[i_pair, i_b][1]

                valid = -1
                if self.entity.geom_start <= i_ga and i_ga < self.entity.geom_end:
                    if i_gb < self.entity.geom_start or self.entity.geom_end <= i_gb:
                        valid = 0
                if self.entity.geom_start <= i_gb and i_gb < self.entity.geom_end:
                    if i_ga < self.entity.geom_start or self.entity.geom_end <= i_ga:
                        valid = 1

                if valid > -1:
                    l_idx = link_idx[i_gb]
                    if valid == 1:
                        l_idx = link_idx[i_ga]
                    i_l = [l_idx, i_b] if qd.static(self.solver._options.batch_links_info) else l_idx
                    i_e = entity_idx[i_l]
                    if mask[i_b, i_e] == 0:
                        g_start = geom_start[i_e]
                        g_end = geom_end[i_e]
                        lower = gu.qd_vec3(qd.math.inf)
                        upper = gu.qd_vec3(-qd.math.inf)
                        for i_g in range(self.solver.n_geoms):
                            if g_start <= i_g < g_end:
                                lower = qd.min(
                                    lower,
                                    geoms_aabb_min[i_g, i_b],
                                )
                                upper = qd.max(
                                    upper,
                                    geoms_aabb_max[i_g, i_b],
                                )
                        for j in range(3):
                            tensor[i_b, i_e, 0, j] = lower[j]
                            tensor[i_b, i_e, 1, j] = upper[j]
                        mask[i_b, i_e] = 1
