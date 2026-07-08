"""Reset event terms for base-state and DOF initialization."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from genesis.typing import Vec2FType
from genesis.utils.geom import axis_angle_to_quat

from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


@functools.lru_cache(maxsize=4)
def _z_axis_up(device: torch.device) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 1.0], device=device)


@EVENT_TERM_REGISTRY.register()
def reset_base_state_uniform(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    pos_x_range: Vec2FType = (-0.5, 0.5),
    pos_y_range: Vec2FType = (-0.5, 0.5),
    pos_z_range: Vec2FType = (0.0, 0.0),
    yaw_range: Vec2FType = (-3.14, 3.14),
):
    """Reset robot base position with uniform random offsets and random yaw.

    Applies small random offsets to the default root position (x, y, z) and
    randomizes the yaw angle. This is similar to mjlab's ``reset_root_state_uniform``.

    Rigid-only: relies on ``set_pos``/``set_quat``, which raise on
    :class:`ParticleEntity` (MPM/SPH). For deformable entities, bake pose into
    the morph at creation time.
    """
    from eden.entities.rigid import RigidEntity

    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity = env.entities[entity_name]
    if not isinstance(entity, RigidEntity):
        raise TypeError(
            f"reset_base_state_uniform requires a RigidEntity for '{entity_name}', got {type(entity).__name__}."
        )

    # Get default base position
    default_pos = entity.default_root_pos  # (num_envs, 3) or (1, 3) or (3,)
    if default_pos.dim() == 1:
        default_pos = default_pos.unsqueeze(0)
    # expand + contiguous clone so in-place ops work
    new_pos = default_pos.expand(env.num_envs, -1).clone()

    # Sample random offsets
    dx = sample_uniform(pos_x_range[0], pos_x_range[1], (env.num_envs,), device=env.device)
    dy = sample_uniform(pos_y_range[0], pos_y_range[1], (env.num_envs,), device=env.device)
    dz = sample_uniform(pos_z_range[0], pos_z_range[1], (env.num_envs,), device=env.device)
    new_pos[:, 0] += dx
    new_pos[:, 1] += dy
    new_pos[:, 2] += dz

    yaw = sample_uniform(yaw_range[0], yaw_range[1], (env.num_envs,), device=env.device)
    quat = axis_angle_to_quat(yaw, _z_axis_up(env.device))

    # set_pos and set_quat have asymmetric shape contracts when ``envs_idx``
    # is a bool mask:
    #   * ``set_pos`` — Eden's wrapper forwards ``relative=False`` (Genesis
    #     default), so Genesis takes the zero-copy fast path
    #     ``torch.where(mask, new, current)`` over the full ``(num_envs, 3)``
    #     buffer. Pre-indexing into a ``(n_true, 3)`` subset crashes the
    #     ``broadcast_tensor`` call inside that path.
    #   * ``set_quat`` — Eden's wrapper defaults ``relative=True`` (compose
    #     with the entity's initial root quat). The fast path requires
    #     ``relative=False``, so set_quat *always* falls through to the slow
    #     path, which sanitizes a bool mask to int indices and expects a
    #     ``(n_selected, 4)`` subset.
    # ``tensor[envs_idx]`` for a bool mask is a subset gather, matching the
    # slow-path contract; the integer-index branch is identical for both.
    is_bool_mask = isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool
    new_pos_arg = new_pos if is_bool_mask else new_pos[envs_idx]
    entity.set_pos(new_pos_arg, envs_idx=envs_idx)
    entity.set_quat(quat[envs_idx], envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
def reset_dofs_by_offset(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    dofs_pos_range: Vec2FType = (-0.1, 0.1),
    dofs_vel_range: Vec2FType | None = None,
):
    """Reset DOFs by adding uniform random offsets to the default DOF positions.

    Rigid-only: requires a :class:`RigidEntity` (DOFs are not defined on
    :class:`ParticleEntity`).
    """
    from eden.entities.rigid import RigidEntity

    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity = env.entities[entity_name]
    if not isinstance(entity, RigidEntity):
        raise TypeError(
            f"reset_dofs_by_offset requires a RigidEntity for '{entity_name}', got {type(entity).__name__}."
        )
    dofs_pos = entity.default_dofs_pos.clone() + sample_uniform(
        dofs_pos_range[0],
        dofs_pos_range[1],
        (env.num_envs, entity.num_dofs),
        device=env.device,
    )
    dofs_pos = dofs_pos.clamp(entity.soft_dofs_pos_limits[:, :, 0], entity.soft_dofs_pos_limits[:, :, 1])
    # TODO: use set_qpos instead
    entity.set_dofs_pos(dofs_pos[envs_idx], envs_idx=envs_idx)

    if dofs_vel_range is not None:
        dofs_vel = sample_uniform(
            dofs_vel_range[0],
            dofs_vel_range[1],
            (env.num_envs, entity.num_dofs),
            device=env.device,
        )[envs_idx]
        entity.set_dofs_vel(dofs_vel, envs_idx=envs_idx)
