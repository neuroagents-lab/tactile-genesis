"""Camera observation terms (RGB, depth, segmentation, normals, point cloud)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers import OBSERVATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@OBSERVATION_TERM_REGISTRY.register()
def rgb_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_rgb()


@OBSERVATION_TERM_REGISTRY.register()
def depth_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_depth()


@OBSERVATION_TERM_REGISTRY.register()
def segmentation_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_segm()


@OBSERVATION_TERM_REGISTRY.register()
def normal_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_normal()


@OBSERVATION_TERM_REGISTRY.register()
def pointcloud_image(
    env: RLEnvBase,
    *,
    camera_name: str,
    world_frame: bool = False,
) -> torch.Tensor:
    pc, valid_mask = env.cameras[camera_name].render_pointcloud(world_frame=world_frame)
    return pc
