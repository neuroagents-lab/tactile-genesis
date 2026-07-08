"""Proprioceptive observation terms (base pose/velocity, joint pos/vel)."""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import torch
from genesis.utils.geom import quat_to_xyz

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@OBSERVATION_TERM_REGISTRY.register()
def base_pos(env: EnvBase, *, entity_name: str, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
    entity: Entity = env.entities[entity_name]
    return entity.get_pos(envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def base_quat(env: EnvBase, *, entity_name: str, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
    """Return the quaternion of the entity's base link.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    envs_idx : torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    """
    entity: Entity = env.entities[entity_name]
    return entity.get_quat(envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def base_rpy(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    terms_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the roll, pitch, and yaw of the entity's base link.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    envs_idx : torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    terms_idx : torch.Tensor | None, optional
        The indices of the terms to return. If None, all terms will be returned. Defaults to None.
    """
    entity: Entity = env.entities[entity_name]
    if terms_idx is not None:
        return quat_to_xyz(entity.get_quat(envs_idx=envs_idx), rpy=True)[..., terms_idx]
    return quat_to_xyz(entity.get_quat(envs_idx=envs_idx), rpy=True)


@OBSERVATION_TERM_REGISTRY.register()
def base_lin_vel(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    frame: Literal["world", "body"] = "world",
) -> torch.Tensor:
    entity: Entity = env.entities[entity_name]
    return entity.get_vel(envs_idx=envs_idx, frame=frame)


@OBSERVATION_TERM_REGISTRY.register()
def base_ang_vel(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    frame: Literal["world", "body"] = "world",
) -> torch.Tensor:
    asset: Entity = env.entities[entity_name]
    return asset.get_ang(envs_idx=envs_idx, frame=frame)


@OBSERVATION_TERM_REGISTRY.register()
def dofs_pos(
    env: EnvBase,
    *,
    entity_name: str,
    offset_from_default: bool = False,
    envs_idx: slice | torch.Tensor | None = None,
    terms_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return joint angles (positions for linear actuators) of the entity.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    offset_from_default : bool, optional
        Whether to offset the joint angles from the default positions. Defaults to False.
    envs_idx : torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    terms_idx : torch.Tensor | None, optional
        The indices of the terms to return. If None, all terms will be returned. Defaults to None.
    """
    robot: Entity = env.entities[entity_name]
    if offset_from_default:
        dofs_pos = robot.get_dofs_pos(envs_idx=envs_idx) - robot.default_dofs_pos
    else:
        dofs_pos = robot.get_dofs_pos(envs_idx=envs_idx)

    if terms_idx is not None:
        return dofs_pos[..., terms_idx]
    return dofs_pos


@OBSERVATION_TERM_REGISTRY.register()
def dofs_vel(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
    terms_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return joint velocities of the entity.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The name of the entity.
    envs_idx : torch.Tensor | None, optional
        The indices of the environments. If None, all environments will be considered. Defaults to None.
    terms_idx : torch.Tensor | None, optional
        The indices of the terms to return. If None, all terms will be returned. Defaults to None.
    """
    robot: Entity = env.entities[entity_name]
    dofs_vel = robot.get_dofs_vel(envs_idx=envs_idx)
    if terms_idx is not None:
        return dofs_vel[..., terms_idx]
    return dofs_vel


@OBSERVATION_TERM_REGISTRY.register()
def dofs_force(env: EnvBase, *, entity_name: str, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    return robot.get_dofs_force(envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def dofs_control_force(env: EnvBase, *, entity_name: str, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    return robot.get_dofs_control_force(envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_pos(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    return robot.get_links_pos(ls_idx_local=ls_idx_local, envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_quat(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    return robot.get_links_quat(ls_idx_local=ls_idx_local, envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_vel(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    return robot.get_links_vel(ls_idx_local=ls_idx_local, envs_idx=envs_idx)


@OBSERVATION_TERM_REGISTRY.register()
def links_ang(
    env: EnvBase,
    *,
    entity_name: str,
    ls_idx_local: torch.Tensor | None = None,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    return robot.get_links_ang(ls_idx_local=ls_idx_local, envs_idx=envs_idx)
