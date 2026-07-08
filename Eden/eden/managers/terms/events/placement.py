"""Object-placement event terms and support-surface samplers."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Callable

import genesis as gs
import torch
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from eden.envs.base import EnvBase
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.utils.geom import transform_by_T
from eden.utils.mesh import ExtendedSupportData
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.options.managers.events import EventTermOptions


@torch.jit.script
def _fused_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    mu_x: float,
    mu_y: float,
    inv_var_x_2: float,
    inv_var_y_2: float,
    norm_const: float,
) -> torch.Tensor:
    # Math: exp( - (x-mu)^2 / 2var )
    # Optimized: exp( - (x-mu)^2 * (0.5 * inv_var) )
    term_x = (x - mu_x).square() * inv_var_x_2
    term_y = (y - mu_y).square() * inv_var_y_2
    return torch.exp(-(term_x + term_y)) * norm_const


class Gaussian2D:
    def __init__(self, mu_x: float, mu_y: float, var_x: float, var_y: float):
        self.mu_x = float(mu_x)
        self.mu_y = float(mu_y)
        # Pre-calculate the normalization constant: 1 / (2 * pi * sigma_x * sigma_y)
        sigma_x = math.sqrt(var_x)
        sigma_y = math.sqrt(var_y)
        self.norm_const = 1.0 / (2 * math.pi * sigma_x * sigma_y)

        # Pre-calculate the inverse denominator for the exponent: 1 / (2 * var)
        # We multiply by this later instead of dividing for speed
        self.inv_var_x_2 = 0.5 / var_x
        self.inv_var_y_2 = 0.5 / var_y

    @torch.inference_mode()
    def get_probs(self, points):
        """Compute the Gaussian weights for the given points.

        Parameters
        ----------
        points: torch.Tensor
            The points to compute the weights for, of shape (N, 2)

        Returns
        -------
        torch.Tensor
            The weights for the points, of shape (N,)
        """
        # View avoids memory copy, unlike slicing sometimes
        x, y = points.unbind(dim=-1)

        return _fused_gaussian_kernel(
            x,
            y,
            self.mu_x,
            self.mu_y,
            self.inv_var_x_2,
            self.inv_var_y_2,
            self.norm_const,
        )


# ------------------------------------------------------------------------------------
# ---------------------------- Placement Samplers ------------------------------------
# ------------------------------------------------------------------------------------


def constant_xy_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    batch_size: int = 1,
    *,
    pos_xy: tuple[float, float],
    support_idx: int = -1,
    **kwargs,
):
    """Sample a constant xy placement position on the support surface.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place
    batch_size: int
        the number of placement positions to sample (one per environment).
    pos_xy: tuple[float, float]
        the xy position of the object to place.
    support_idx: int
        index into the valid support surfaces to place on; ``-1`` selects the last one.
    **kwargs
        additional keyword arguments, ignored by this sampler.

    NOTE: This sampler aligns z-level to the support surface but *no guarantee* on the given pos_xy is on the surface.
    NOTE: This sampler does *not* care the collision states.
          It is a user's responsibility to ensure the specified pos_xy is free from collision.
    """
    del asset_AABB  # unused
    # find the support with normal +z
    support_data = [d for d in support_data if d.valid_mask.any()]
    d = support_data[max(support_idx, 0)]
    vertices_3d_on_plane = torch.zeros((batch_size, 1, 3), device=gs.device, dtype=gs.tc_float)  # (B, N, 3)

    if isinstance(pos_xy, tuple):
        vertices_3d_on_plane[:, :, 0] = pos_xy[0]
        vertices_3d_on_plane[:, :, 1] = pos_xy[1]
    else:
        vertices_3d_on_plane[:, :, :2] = pos_xy

    res_pos = transform_by_T(vertices_3d_on_plane, d.transform).squeeze(1)
    res_dir = d.normal
    return res_pos, res_dir, None


def range_xy_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    batch_size: int = 1,
    *,
    range_x: tuple[float, float] | torch.Tensor,
    range_y: tuple[float, float] | torch.Tensor,
    support_idx: int = -1,
    **kwargs,
):
    """Sample an xy placement position uniformly from a rectangular range on the support surface.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place
    batch_size: int
        the number of placement positions to sample (one per environment).
    range_x: tuple[float, float] | torch.Tensor
        the range of the x-coordinate of the object to place.
    range_y: tuple[float, float] | torch.Tensor
        the range of the y-coordinate of the object to place.
    support_idx: int
        index into the valid support surfaces to place on; ``-1`` selects the last one.
    **kwargs
        additional keyword arguments, ignored by this sampler.

    NOTE: This sampler aligns z-level to the support surface but *no guarantee* on the given pos_xy is on the surface.
    NOTE: This sampler does *not* care the collision states.
          It is a user's responsibility to ensure the specified pos_xy is free from collision.
    """
    del asset_AABB  # unused
    # find the support with normal +z
    support_data = [d for d in support_data if d.valid_mask.any()]
    d = support_data[max(support_idx, 0)]
    lower = torch.zeros(batch_size, 3, device=gs.device, dtype=gs.tc_float)
    upper = torch.zeros(batch_size, 3, device=gs.device, dtype=gs.tc_float)
    lower[:, 0] = range_x[0]
    lower[:, 1] = range_y[0]
    upper[:, 0] = range_x[1]
    upper[:, 1] = range_y[1]
    vertices_3d_on_plane = sample_uniform(lower, upper, (batch_size, 3), device=gs.device)  # (B, 3)

    res_pos = transform_by_T(vertices_3d_on_plane, d.transform).squeeze(1)
    res_dir = d.normal
    return res_pos, res_dir, None


def uniform_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    holes: torch.Tensor | None = None,
    **kwargs,
):
    """Sample a placement position uniformly from the pre-sampled support points.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place (B, 2, 3)
    holes: torch.Tensor | None
        the AABBs of existing objects concatenated in dim=1, e.g., (B, L, 2, 3) for L holes
    **kwargs
        additional keyword arguments, ignored by this sampler.
    """
    offset = (asset_AABB[:, 1] - asset_AABB[:, 0]) / 2  # (B, 3)
    # consider only supports that are valid for current gravity
    support_data = [d for d in support_data if d.valid_mask.any()]

    res_pos = []
    res_dir = []
    res_valid_mask = []

    for d in support_data:
        if d.sample_points is not None:
            vertices_3d_on_plane = torch.zeros((1, d.num_sample_points, 3), device=gs.device, dtype=gs.tc_float)
            vertices_3d_on_plane[0, :, :2] = d.sample_points
            res_pos.append(transform_by_T(vertices_3d_on_plane, d.transform))  # B, N, 3
            res_dir.append(d.normal.unsqueeze(1).expand(-1, d.num_sample_points, 3))
            res_valid_mask.append(d.valid_mask.unsqueeze(1).expand(-1, d.num_sample_points))  # B, N

    if len(res_pos) > 0:
        res_pos = torch.cat(res_pos, dim=1)  # (B, sum(N), 3)
        res_dir = torch.cat(res_dir, dim=1)  # (B, sum(N), 3)
        res_valid_mask = torch.cat(res_valid_mask, dim=1)  # (B, sum(N),)

        # NOTE: filter out points inside holes
        if holes is not None:
            assert holes.ndim == 4, f"Unexpected holes shape {holes.shape}"
            if holes.shape[1] > 0:
                min_occ = holes[..., 0, :2] - offset[:, None, :2]
                max_occ = holes[..., 1, :2] + offset[:, None, :2]
                is_inside_tmp = torch.all(
                    (min_occ[:, None] <= res_pos[..., None, :2]) & (res_pos[..., None, :2] < max_occ[:, None]),
                    dim=-1,
                ).any(dim=-1)  # (B, sum(N))
                res_valid_mask = res_valid_mask & torch.logical_not(is_inside_tmp)

        valid_mask = res_valid_mask.any(dim=1)  # (B, )
        random_true_indices = torch.multinomial(res_valid_mask.float(), 1).flatten()
        res_pos = res_pos[torch.arange(res_pos.shape[0]), random_true_indices]  # (B', 3)
        res_dir = res_dir[torch.arange(res_dir.shape[0]), random_true_indices]  # (B', 3)
        return res_pos, res_dir, valid_mask
    else:
        raise ValueError("No pre-sampled points found for the support data")


def gaussian_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    holes: torch.Tensor | None = None,
    gaussian_weights: Callable | None = None,
    **kwargs,
):
    """Sample a placement position from the support points weighted by a Gaussian.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place
    holes: torch.Tensor | None
        the AABBs of existing objects concatenated in dim=1, e.g., (B, L, 2, 3) for L holes
    gaussian_weights: Callable | None
        callable mapping candidate points to per-point sampling weights. If None, a default is used.
    **kwargs
        additional keyword arguments, ignored by this sampler.
    """
    offset = (asset_AABB[:, 1] - asset_AABB[:, 0]) / 2  # (B, 3)
    # consider only supports that are valid for current gravity
    support_data = [d for d in support_data if d.valid_mask.any()]

    res_pos = []
    res_dir = []
    res_valid_mask = []
    res_weights = []

    for d in support_data:
        if d.sample_points is not None:
            vertices_3d_on_plane = torch.zeros((1, d.num_sample_points, 3), device=gs.device, dtype=gs.tc_float)
            vertices_3d_on_plane[0, :, :2] = d.sample_points
            pos = transform_by_T(vertices_3d_on_plane, d.transform)
            res_pos.append(pos)  # B, N, 3
            res_dir.append(d.normal.unsqueeze(1).expand(-1, d.num_sample_points, 3))

            res_weights.append(gaussian_weights(pos[..., :2]))  # B, N
            res_valid_mask.append(d.valid_mask.unsqueeze(1).expand(-1, d.num_sample_points))  # B, N

    if len(res_pos) > 0:
        res_pos = torch.cat(res_pos, dim=1)  # (B, sum(N), 3)
        res_dir = torch.cat(res_dir, dim=1)  # (B, sum(N), 3)
        res_valid_mask = torch.cat(res_valid_mask, dim=1)  # (B, sum(N),)
        res_weights = torch.cat(res_weights, dim=1)  # (B, sum(N),)

        # NOTE: filter out points inside holes
        if holes is not None:
            assert holes.ndim == 4, f"Unexpected holes shape {holes.shape}"
            if holes.shape[1] > 0:
                min_occ = holes[..., 0, :2] - offset[:, None, :2]
                max_occ = holes[..., 1, :2] + offset[:, None, :2]
                is_inside_tmp = torch.all(
                    (min_occ[:, None] <= res_pos[..., None, :2]) & (res_pos[..., None, :2] < max_occ[:, None]),
                    dim=-1,
                ).any(dim=-1)  # (B, sum(N))
                res_valid_mask = res_valid_mask & torch.logical_not(is_inside_tmp)

        valid_mask = res_valid_mask.any(dim=1)  # (B, )
        random_true_indices = torch.multinomial(res_weights * res_valid_mask.float(), 1).flatten()
        res_pos = res_pos[torch.arange(res_pos.shape[0]), random_true_indices]  # (B', 3)
        res_dir = res_dir[torch.arange(res_dir.shape[0]), random_true_indices]  # (B', 3)
        return res_pos, res_dir, valid_mask
    else:
        raise ValueError("No pre-sampled points found for the support data")


@EVENT_TERM_REGISTRY.register()
def place_constant_xy(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    support_entity_name: str,
    pos_xy: tuple[float, float],
):
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)
    entity: Entity = env.entities[entity_name]
    support_entity: Entity = env.entities[support_entity_name]
    # NOTE: reset the orientation to the default root quat
    entity.set_quat(entity.default_root_quat.repeat(env.num_envs, 1)[envs_idx], envs_idx=envs_idx)
    sampler = partial(constant_xy_sampler, pos_xy=pos_xy)
    entity.place_on_to(support_entity, sampler=sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
def place_range_xy(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    support_entity_name: str,
    range_x: tuple[float, float],
    range_y: tuple[float, float],
    range_roll: tuple[float, float] = (0, 0),
    range_pitch: tuple[float, float] = (0, 0),
    range_yaw: tuple[float, float] = (0, 0),
):
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity: Entity = env.entities[entity_name]
    support_entity: Entity = env.entities[support_entity_name]

    lower = torch.tensor(
        [range_roll[0], range_pitch[0], range_yaw[0]],
        device=env.device,
        dtype=torch.float32,
    )
    upper = torch.tensor(
        [range_roll[1], range_pitch[1], range_yaw[1]],
        device=env.device,
        dtype=torch.float32,
    )
    sampled = sample_uniform(lower, upper, (env.num_envs, 3), device=env.device)[envs_idx]
    quat = transform_quat_by_quat(
        entity.default_root_quat.repeat(env.num_envs, 1)[envs_idx], xyz_to_quat(sampled, rpy=True)
    )
    entity.set_quat(quat, envs_idx=envs_idx)
    sampler = partial(range_xy_sampler, range_x=range_x, range_y=range_y)
    entity.place_on_to(support_entity, sampler=sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
def place_uniform(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    support_entity_name: str,
    samples_per_area: int = 100,
):
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)
    entity: Entity = env.entities[entity_name]
    support_entity: Entity = env.entities[support_entity_name]
    # NOTE: reset the orientation to the default root quat
    entity.set_quat(entity.default_root_quat.repeat(env.num_envs, 1)[envs_idx], envs_idx=envs_idx)
    sampler = partial(uniform_sampler, samples_per_area=samples_per_area)
    entity.place_on_to(support_entity, sampler=sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class PlaceGaussian(EventTerm):
    """Event term that places an entity on a support surface via Gaussian-weighted sampling.

    Parameters
    ----------
    entity_name: str
        The name of the entity to place
    support_entity_name: str
        The name of the support entity
    mu_x: float
        The mean of the x-coordinate in world frame
    mu_y: float
        The mean of the y-coordinate in world frame
    """

    entity_name: str = ""
    support_entity_name: str = ""
    mu_x: float = 0.0
    mu_y: float = 0.0
    var_x: float = 0.1
    var_y: float = 0.1

    def __init__(
        self,
        env: EnvBase,
        options: EventTermOptions,
    ):
        super().__init__(env=env, options=options)
        self.entity: Entity | None = None
        self.support_entity: Entity | None = None
        self.sampler = None

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        self.support_entity = self._env.entities[self.support_entity_name]
        self.sampler = partial(
            gaussian_sampler,
            gaussian_weights=Gaussian2D(
                mu_x=self.mu_x,
                mu_y=self.mu_y,
                var_x=self.var_x,
                var_y=self.var_y,
            ).get_probs,
        )

    def compute(self, envs_idx: slice | torch.Tensor | None):
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)

        # NOTE: reset the orientation to the default root quat
        self.entity.set_quat(self.entity.default_root_quat.repeat(self._env.num_envs, 1)[envs_idx], envs_idx=envs_idx)
        self.entity.place_on_to(self.support_entity, sampler=self.sampler, envs_idx=envs_idx)
