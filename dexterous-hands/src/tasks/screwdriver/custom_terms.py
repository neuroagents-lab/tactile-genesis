"""Screwdriver task helpers, rewards, observations, and terminations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import genesis.utils.geom as gu
import torch
from eden.constants import MetricDirection, MetricMode
from eden.managers import (
    METRIC_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    ObservationTerm,
    RewardTerm,
    TerminationTerm,
)
from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.options import ObservationTermOptions
from eden.utils.isaac_math import axis_angle_from_quat, quat_conjugate, quat_mul

from shared_terms import _env_cache_masked_set
from utils import get_entity_metadata, resolve_entity_link

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


FINGER_NAME_ALIASES = {
    "thumb": ("thumb", "th"),
    "index": ("index", "ff"),
    "middle": ("middle", "mid", "mf"),
    "ring": ("ring", "rf"),
    "pinky": ("pinky", "little", "lf"),
}


def _name_matches_alias(name: str, alias: str) -> bool:
    normalized = name.lower()
    token = alias.lower()
    if len(token) >= 3:
        return token in normalized
    if normalized.startswith(token):
        return True
    return any(f"{delimiter}{token}" in normalized for delimiter in ("_", "-", ".", "/"))


def _get_robot_metadata(robot):
    metadata = getattr(getattr(robot, "options", None), "metadata", None)
    return getattr(robot, "metadata", None) if metadata is None else metadata


def _iter_hand_links(
    robot, *, include_fingertips: bool, include_finger_links: bool, include_palm_link: bool
) -> tuple[str, ...]:
    metadata = _get_robot_metadata(robot)
    if metadata is None:
        return ()
    links = []
    if include_fingertips:
        links.extend(getattr(metadata, "fingertip_links", ()) or ())
    if include_finger_links:
        links.extend(getattr(metadata, "finger_links", ()) or ())
    if include_palm_link and hasattr(metadata, "palm_link"):
        links.append(metadata.palm_link)
    return tuple(links)


def _link_matches_finger(link_name: str, finger: str) -> bool:
    aliases = FINGER_NAME_ALIASES.get(finger)
    if aliases is None:
        raise ValueError(f"Unsupported finger: {finger}")
    return any(_name_matches_alias(link_name, alias) for alias in aliases)


def resolve_hand_dof_indices(robot, *, finger: str) -> tuple[int, ...]:
    aliases = FINGER_NAME_ALIASES.get(finger)
    if aliases is None:
        raise ValueError(f"Unsupported finger: {finger}")
    dof_names = list(getattr(getattr(robot, "options", None), "dofs_name", ()) or ())
    if not dof_names:
        dof_names = list(getattr(robot, "dofs_name", ()) or ())
    return tuple(i for i, name in enumerate(dof_names) if any(_name_matches_alias(name, alias) for alias in aliases))


def filter_hand_dof_names(
    dof_names: list[str] | tuple[str, ...],
    *,
    excluded_fingers: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    excluded_aliases = []
    for finger in excluded_fingers:
        aliases = FINGER_NAME_ALIASES.get(finger)
        if aliases is None:
            raise ValueError(f"Unsupported finger: {finger}")
        excluded_aliases.extend(aliases)
    if not excluded_aliases:
        return tuple(dof_names)
    return tuple(name for name in dof_names if not any(_name_matches_alias(name, alias) for alias in excluded_aliases))


def resolve_hand_link(
    robot,
    *,
    finger: str,
    link_name: str | None = None,
    role: str = "fingertip",
) -> str:
    if link_name:
        try:
            robot.get_link(link_name)
            return link_name
        except Exception:
            pass
    if role == "fingertip":
        candidates = _iter_hand_links(
            robot, include_fingertips=True, include_finger_links=True, include_palm_link=False
        )
    elif role == "finger":
        candidates = _iter_hand_links(
            robot, include_fingertips=False, include_finger_links=True, include_palm_link=False
        )
    elif role == "palm":
        candidates = _iter_hand_links(
            robot, include_fingertips=False, include_finger_links=False, include_palm_link=True
        )
    else:
        raise ValueError(f"Unsupported hand role: {role}")
    for candidate in candidates:
        if role == "palm" or _link_matches_finger(candidate, finger):
            return candidate
    raise ValueError(f"Could not resolve a '{role}' link for finger '{finger}'. Requested link was '{link_name}'.")


def resolve_grasp_target_link_name(entity, *, default_link_name: str | None = None) -> str | None:
    metadata = get_entity_metadata(entity)
    if metadata is None:
        return default_link_name
    link_name = getattr(metadata, "grasp_target_link", None)
    return link_name if link_name is not None else default_link_name


def resolve_grasp_target_link(entity, *, link_name: str | None = None, default_link_name: str | None = None):
    resolved_link_name = (
        link_name
        if link_name is not None
        else resolve_grasp_target_link_name(entity, default_link_name=default_link_name)
    )
    if resolved_link_name is None:
        return entity
    return resolve_entity_link(entity, resolved_link_name)


def get_grasp_target_pos(
    entity,
    *,
    link_name: str | None = None,
    use_aabb_center: bool | None = None,
    local_offset: tuple[float, float, float] | list[float] | torch.Tensor | None = None,
    default_link_name: str | None = None,
):
    metadata = get_entity_metadata(entity)
    target = resolve_grasp_target_link(entity, link_name=link_name, default_link_name=default_link_name)
    if use_aabb_center is None:
        use_aabb_center = (
            bool(getattr(metadata, "grasp_target_use_aabb_center", False)) if metadata is not None else False
        )
    if local_offset is None and metadata is not None and hasattr(metadata, "grasp_target_local_offset"):
        local_offset = getattr(metadata, "grasp_target_local_offset")
    target_pos = target.get_AABB().mean(dim=-2) if use_aabb_center else target.get_pos()
    if local_offset is not None:
        offset = torch.as_tensor(local_offset, device=target_pos.device, dtype=target_pos.dtype).view(1, 3)
        target_pos = target_pos + gu.transform_by_quat(offset.expand(target_pos.shape[0], -1), target.get_quat())
    return target_pos


def get_grasp_target_distance(
    entity,
    points: torch.Tensor,
    *,
    link_name: str | None = None,
    use_aabb: bool | None = None,
    local_offset: tuple[float, float, float] | list[float] | torch.Tensor | None = None,
    default_link_name: str | None = None,
) -> torch.Tensor:
    metadata = get_entity_metadata(entity)
    target = resolve_grasp_target_link(entity, link_name=link_name, default_link_name=default_link_name)
    if use_aabb is None:
        use_aabb = bool(getattr(metadata, "grasp_target_use_aabb_center", False)) if metadata is not None else False
    if use_aabb and local_offset is None:
        target_aabb = target.get_AABB()
        aabb_min = target_aabb[:, 0, :]
        aabb_max = target_aabb[:, 1, :]
        while aabb_min.ndim < points.ndim:
            aabb_min = aabb_min.unsqueeze(-2)
            aabb_max = aabb_max.unsqueeze(-2)
        closest_points = torch.maximum(torch.minimum(points, aabb_max), aabb_min)
        return torch.norm(points - closest_points, dim=-1)
    target_pos = get_grasp_target_pos(
        entity,
        link_name=link_name,
        use_aabb_center=use_aabb,
        local_offset=local_offset,
        default_link_name=default_link_name,
    )
    while target_pos.ndim < points.ndim:
        target_pos = target_pos.unsqueeze(-2)
    return torch.norm(points - target_pos, dim=-1)


def _build_thumb_index_grasp_term(
    term,
    *,
    obj_name: str,
    robot_name: str,
    link_name: str | None,
    thumb_link: str | None,
    index_link: str | None,
) -> None:
    term._obj = term._env.entities[obj_name]
    term._robot = term._env.entities[robot_name]
    term._target = resolve_grasp_target_link(term._obj, link_name=link_name)
    term._thumb_link = term._robot.get_link(
        resolve_hand_link(term._robot, finger="thumb", link_name=thumb_link, role="fingertip")
    )
    term._index_link = term._robot.get_link(
        resolve_hand_link(term._robot, finger="index", link_name=index_link, role="fingertip")
    )


def _thumb_index_positions(term) -> torch.Tensor:
    return torch.stack((term._thumb_link.get_pos(), term._index_link.get_pos()), dim=1)


def _expand_vec(
    values: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_envs: int,
    normalize: bool = False,
) -> torch.Tensor:
    vec = torch.as_tensor(values, device=device, dtype=dtype).view(1, 3)
    if normalize:
        vec = vec / torch.linalg.vector_norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
    return vec.expand(n_envs, -1)


def _build_alignment_term(
    term,
    *,
    obj_name: str,
    link_name: str | None,
    local_up_axis: tuple[float, float, float],
    world_up_axis: tuple[float, float, float],
) -> None:
    term._obj = term._env.entities[obj_name]
    term._target = resolve_grasp_target_link(term._obj, link_name=link_name)
    dtype = term._target.get_quat().dtype
    term._local_axis = _expand_vec(local_up_axis, device=term.device, dtype=dtype, n_envs=term.num_envs)
    term._world_axis = _expand_vec(
        world_up_axis,
        device=term.device,
        dtype=dtype,
        n_envs=term.num_envs,
        normalize=True,
    )


def _compute_alignment_term(term) -> torch.Tensor:
    target_quat = term._target.get_quat()
    target_up = gu.transform_by_quat(term._local_axis, target_quat)
    target_up = target_up / torch.linalg.vector_norm(target_up, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.clamp((target_up * term._world_axis).sum(dim=-1), min=-1.0, max=1.0)


def _get_rotation_axis(env, *, command_name: str = "rotation_axis", dtype: torch.dtype | None = None) -> torch.Tensor:
    try:
        axis = env.command_manager.get_command(command_name)
    except Exception:
        axis = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=dtype or torch.float32).expand(env.num_envs, -1)
    axis = axis.to(device=env.device, dtype=dtype or axis.dtype)
    return axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)


def _get_entity_dof_names(entity) -> tuple[str, ...]:
    dof_names = list(getattr(getattr(entity, "options", None), "dofs_name", ()) or ())
    runtime_dof_names = list(getattr(entity, "dofs_name", ()) or ())
    for name in runtime_dof_names:
        if name not in dof_names:
            dof_names.append(name)
    return tuple(dof_names)


def _resolve_spin_dof_index(entity, *, dofs_idx_local=None) -> int | None:
    configured_dof_names = tuple(getattr(getattr(entity, "options", None), "dofs_name", ()) or ())
    runtime_dof_names = tuple(getattr(entity, "dofs_name", ()) or ())
    if "handle_ball_joint" in configured_dof_names or runtime_dof_names.count("handle_ball_joint") >= 3:
        n_dofs = len(dofs_idx_local) if dofs_idx_local is not None else len(runtime_dof_names)
        return 2 if n_dofs >= 3 else None
    dof_names = _get_entity_dof_names(entity)
    for preferred_name in ("handle_to_shaft", "cylindrical_1_2", "nut", "screw"):
        try:
            return dof_names.index(preferred_name)
        except ValueError:
            continue
    return 0 if len(dof_names) == 1 else None


def _get_explicit_screw_dof_state(env, *, obj_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    obj = env.entities[obj_name]
    dofs_idx_local = getattr(obj, "dofs_idx_local", None)
    spin_dof_idx = _resolve_spin_dof_index(obj, dofs_idx_local=dofs_idx_local)
    if dofs_idx_local is None or spin_dof_idx is None or len(dofs_idx_local) <= spin_dof_idx:
        return None
    try:
        dof_pos = obj.get_dofs_pos(dofs_idx_local)[:, spin_dof_idx]
        dof_vel = obj.get_dofs_vel(dofs_idx_local)[:, spin_dof_idx]
    except Exception:
        return None
    if not hasattr(env, "_init_screw_dof_pos"):
        env._init_screw_dof_pos = dof_pos.clone()
    is_first_step = env.episode_length_buf <= 1
    if is_first_step.any():
        idx = torch.where(is_first_step)[0]
        _env_cache_masked_set(env, "_init_screw_dof_pos", idx, dof_pos[is_first_step])
        dof_vel = dof_vel.clone()
        dof_vel[is_first_step] = 0.0
    return dof_pos - env._init_screw_dof_pos, dof_vel


def get_screw_axis_state(
    env,
    *,
    obj_name: str = "obj",
    link_name: str | None = None,
    command_name: str = "rotation_axis",
) -> tuple[torch.Tensor, torch.Tensor]:
    step_token = env.episode_length_buf
    cached_step_token = getattr(env, "_screw_axis_state_step_token", None)
    if cached_step_token is not None and torch.equal(cached_step_token, step_token):
        return env._cached_screw_axis_pos, env._cached_screw_dof_vel
    obj = env.entities[obj_name]
    explicit_dof_state = _get_explicit_screw_dof_state(env, obj_name=obj_name)
    if explicit_dof_state is not None:
        env._cached_screw_axis_pos, env._cached_screw_dof_vel = explicit_dof_state
        env._screw_axis_state_step_token = step_token.clone()
        return env._cached_screw_axis_pos, env._cached_screw_dof_vel
    target = resolve_grasp_target_link(obj, link_name=link_name)
    target_quat = target.get_quat()
    axis = _get_rotation_axis(env, command_name=command_name, dtype=target_quat.dtype)
    if not hasattr(env, "_prev_screw_target_quat"):
        env._prev_screw_target_quat = target_quat.clone()
        env._cached_screw_axis_pos = torch.zeros(env.num_envs, device=env.device, dtype=target_quat.dtype)
        env._cached_screw_dof_vel = torch.zeros(env.num_envs, device=env.device, dtype=target_quat.dtype)
        env._screw_axis_state_step_token = step_token.clone()
        return env._cached_screw_axis_pos, env._cached_screw_dof_vel
    is_first_step = env.episode_length_buf <= 1
    if is_first_step.any():
        idx = torch.where(is_first_step)[0]
        _env_cache_masked_set(env, "_prev_screw_target_quat", idx, target_quat[is_first_step])
        _env_cache_masked_set(env, "_cached_screw_axis_pos", idx, 0.0)
        _env_cache_masked_set(env, "_cached_screw_dof_vel", idx, 0.0)
    quat_diff = quat_mul(target_quat, quat_conjugate(env._prev_screw_target_quat))
    axis_delta = (axis_angle_from_quat(quat_diff) * axis).sum(dim=-1)
    dt = env.env_options.sim_dt * env.env_options.decimation
    env._cached_screw_axis_pos = env._cached_screw_axis_pos + axis_delta
    env._cached_screw_dof_vel = axis_delta / dt
    env._prev_screw_target_quat = target_quat.clone()
    env._screw_axis_state_step_token = step_token.clone()
    return env._cached_screw_axis_pos, env._cached_screw_dof_vel


@OBSERVATION_TERM_REGISTRY.register()
def screw_rotation_state_obs(
    env,
    *,
    obj_name: str = "obj",
    link_name: str | None = None,
    command_name: str = "rotation_axis",
) -> torch.Tensor:
    screw_axis_pos, _ = get_screw_axis_state(env, obj_name=obj_name, link_name=link_name, command_name=command_name)
    return screw_axis_pos.unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
def screw_rotation_velocity_obs(
    env,
    *,
    obj_name: str = "obj",
    link_name: str | None = None,
    command_name: str = "rotation_axis",
) -> torch.Tensor:
    _, screw_axis_vel = get_screw_axis_state(env, obj_name=obj_name, link_name=link_name, command_name=command_name)
    return screw_axis_vel.unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
class ScrewObjTiltObs(ObservationTerm):
    """Off-axis tilt of the screwdriver as a single scalar (radians).

    The angle between the object's local up axis (the shaft, rotated into the
    world frame) and the world vertical -- i.e. how far the screwdriver has
    fallen over. ``0`` means perfectly upright; larger means more tilted. This
    is the privileged quantity an auxiliary distillation head can be asked to
    predict from the student latent.
    """

    obj_name: str = "obj"
    link_name: str | None = None
    local_up_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self._buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

    def build(self) -> None:
        super().build()
        _build_alignment_term(
            self,
            obj_name=self.obj_name,
            link_name=self.link_name,
            local_up_axis=self.local_up_axis,
            world_up_axis=self.world_up_axis,
        )

    def compute(self, *args, **kwargs) -> torch.Tensor:
        alignment = _compute_alignment_term(self)
        self._buf[:, 0] = torch.arccos(alignment).to(self._buf.dtype)
        return self._buf


@OBSERVATION_TERM_REGISTRY.register()
class ScrewAxisRotationProgressObs(ObservationTerm):
    """Per-step rotation progress about the screwdriver's shaft (radians).

    The signed angle the object rotated about ``local_axis`` -- an axis fixed in
    the object's local frame -- since the previous step, mirroring the per-step
    contribution of :func:`shared_terms.rotation_reward`. Positive means rotation
    along ``+local_axis`` (the rewarded direction). Used as an auxiliary
    distillation target so the student latent encodes how much target-axis
    rotation it just achieved.
    """

    obj_name: str = "obj"
    local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self._buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._prev_quat: torch.Tensor | None = None
        self._cached_token: torch.Tensor | None = None

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.obj_name]
        self._prev_quat = self._obj.get_quat().clone()

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Per-episode reinit is handled by the episode_length_buf <= 1 guard in
        # compute(); just drop the per-step cache so the next call recomputes.
        self._cached_token = None

    def compute(self, *args, **kwargs) -> torch.Tensor:
        step_token = self._env.episode_length_buf
        if self._cached_token is not None and torch.equal(self._cached_token, step_token):
            return self._buf

        quat = self._obj.get_quat().detach()
        # On the first step of each episode the stored prev quat is stale; use the
        # current quat there so the delta is zero. ``torch.where`` keeps this
        # functional (no in-place writes on possibly-inference tensors).
        is_first_step = (step_token <= 1).unsqueeze(-1)
        prev_quat = torch.where(is_first_step, quat, self._prev_quat)

        quat_diff = quat_mul(quat, quat_conjugate(prev_quat))
        local = torch.tensor(self.local_axis, device=self.device, dtype=quat.dtype).expand(self.num_envs, 3)
        target_axis = gu.transform_by_quat(local, quat)
        axis_delta = (axis_angle_from_quat(quat_diff) * target_axis).sum(dim=-1)

        self._buf[:, 0] = axis_delta.to(self._buf.dtype)
        self._prev_quat = quat
        self._cached_token = step_token.clone()
        return self._buf


@METRIC_TERM_REGISTRY.register(
    direction=MetricDirection.HIB,
    metric_mode=MetricMode.RESET,
)
def screw_total_rotation_metric(
    env,
    *,
    obj_name: str = "obj",
    local_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> torch.Tensor:
    """Total signed rotation (radians) of the object over the episode.

    Each step, measures the object's incremental rotation and projects it onto
    ``local_axis`` -- an axis fixed in the object's local frame (the screwdriver
    shaft), rotated by the current quaternion each step -- mirroring
    :func:`shared_terms.rotation_reward`. The per-step deltas are accumulated, so
    the value reported on episode reset is the net rotation about the shaft for
    the whole episode. Positive means rotation along ``+local_axis``.

    Parameters
    ----------
    obj_name : str
        Name of the object entity. Default: "obj"
    local_axis : tuple[float, float, float]
        Object-frame axis the rotation is projected onto. Default: (0.0, -1.0, 0.0)
    """
    obj = env.entities[obj_name]
    quat = obj.get_quat()

    # First-ever call: initialise the per-episode accumulators.
    if not hasattr(env, "_screw_total_rotation"):
        env._screw_total_rotation = torch.zeros(env.num_envs, device=env.device, dtype=quat.dtype)
        env._screw_total_rotation_prev_quat = quat.clone()
        return env._screw_total_rotation

    # Reset the accumulator for envs whose episode just (re)started.
    is_first_step = env.episode_length_buf <= 1
    if is_first_step.any():
        idx = torch.where(is_first_step)[0]
        _env_cache_masked_set(env, "_screw_total_rotation", idx, 0.0)
        _env_cache_masked_set(env, "_screw_total_rotation_prev_quat", idx, quat[is_first_step])

    # Incremental rotation since the previous step, projected onto the shaft axis.
    quat_diff = quat_mul(quat, quat_conjugate(env._screw_total_rotation_prev_quat))
    local = torch.tensor(local_axis, device=env.device, dtype=quat.dtype).expand(env.num_envs, 3)
    target_axis = gu.transform_by_quat(local, quat)
    axis_delta = (axis_angle_from_quat(quat_diff) * target_axis).sum(dim=-1)

    env._screw_total_rotation = env._screw_total_rotation + axis_delta
    env._screw_total_rotation_prev_quat = quat.clone()
    return env._screw_total_rotation


@REWARD_TERM_REGISTRY.register()
def screw_rotation_reward(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    link_name: str | None = None,
    command_name: str = "rotation_axis",
    angvel_clip_min: float = -4.0,
    angvel_clip_max: float = 4.0,
    rotation_sign: float = 1.0,
) -> torch.Tensor:
    _, screw_dof_vel = get_screw_axis_state(env, obj_name=entity_name, link_name=link_name, command_name=command_name)
    return torch.clip(rotation_sign * screw_dof_vel, min=angvel_clip_min, max=angvel_clip_max)


@REWARD_TERM_REGISTRY.register()
def screw_angvel_penalty(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    link_name: str | None = None,
    command_name: str = "rotation_axis",
    threshold: float = 15.0,
    rotation_sign: float = 1.0,
) -> torch.Tensor:
    _, screw_dof_vel = get_screw_axis_state(env, obj_name=entity_name, link_name=link_name, command_name=command_name)
    return torch.clamp(rotation_sign * screw_dof_vel - threshold, min=0.0)


@REWARD_TERM_REGISTRY.register()
def screw_pc_z_dist_penalty(
    env: EnvBase,
    *,
    min_z_spread: float = 0.03,
    point_cloud_attr: str = "point_cloud_buf",
    point_cloud_sensor_name: str = "",
) -> torch.Tensor:
    point_cloud = getattr(env, point_cloud_attr, None)
    if not isinstance(point_cloud, torch.Tensor) and point_cloud_sensor_name and hasattr(env, "sensors"):
        sensor = env.sensors.get(point_cloud_sensor_name)
        if sensor is not None:
            data = sensor.read()
            tensors = data if isinstance(data, tuple) else (data,)
            point_cloud = next(
                (t for t in tensors if isinstance(t, torch.Tensor) and t.ndim == 3 and t.shape[-1] == 3),
                None,
            )
    if not isinstance(point_cloud, torch.Tensor) or point_cloud.ndim != 3 or point_cloud.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)
    z_spread = point_cloud[..., 2].amax(dim=1) - point_cloud[..., 2].amin(dim=1)
    return torch.where(z_spread > min_z_spread, z_spread, torch.zeros_like(z_spread))


@REWARD_TERM_REGISTRY.register()
def screw_object_dist_penalty(
    env: EnvBase,
    *,
    entity_name: str = "obj",
    margin: float = 0.0,
) -> torch.Tensor:
    """Per-env distance penalty against the object placement cached by :class:`LoadGraspPose`.

    Reads ``env._sampled_grasp_obj_pos`` (shape ``[num_envs, 3]``) so each env's target tracks
    its own screwdriver geometry/grasp. With ``margin > 0``, distances within ``margin`` produce
    zero penalty; beyond that, the penalty grows linearly as ``dist - margin``. Envs whose
    grasp record lacks an object pose (``_sampled_grasp_has_obj_pose`` False) get zero penalty.
    """
    obj = env.entities[entity_name]
    obj_pos = obj.get_pos()

    target = getattr(env, "_sampled_grasp_obj_pos", None)
    if target is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=obj_pos.dtype)

    dist = torch.norm(obj_pos - target.to(dtype=obj_pos.dtype), dim=-1)
    penalty = torch.clamp(dist - margin, min=0.0)

    has_pose = getattr(env, "_sampled_grasp_has_obj_pose", None)
    if has_pose is not None:
        penalty = penalty * has_pose.to(dtype=penalty.dtype)
    return penalty


@REWARD_TERM_REGISTRY.register()
class ScrewVerticalAlignmentPenalty(RewardTerm):
    """Penalize tilt of the object's ``local_up_axis`` away from ``world_up_axis``.

    Parameters
    ----------
    tilt_margin : float
        The angle in degrees beyond which the penalty starts to apply.
    """

    obj_name: str = "obj"
    link_name: str | None = None
    tilt_margin: float = 12.0
    local_up_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def build(self) -> None:
        super().build()
        _build_alignment_term(
            self,
            obj_name=self.obj_name,
            link_name=self.link_name,
            local_up_axis=self.local_up_axis,
            world_up_axis=self.world_up_axis,
        )
        self._min_alignment = math.cos(math.radians(self.tilt_margin))

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        alignment = _compute_alignment_term(self)
        return torch.clamp(self._min_alignment - alignment, min=0.0)


@REWARD_TERM_REGISTRY.register()
class ScrewSurfaceDistanceReward(RewardTerm):
    obj_name: str = "obj"
    robot_name: str = "robot"
    link_name: str | None = None
    thumb_link: str | None = None
    index_link: str | None = None
    distance_threshold: float = 0.05

    def build(self) -> None:
        super().build()
        _build_thumb_index_grasp_term(
            self,
            obj_name=self.obj_name,
            robot_name=self.robot_name,
            link_name=self.link_name,
            thumb_link=self.thumb_link,
            index_link=self.index_link,
        )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        fingertip_dist = get_grasp_target_distance(self._obj, _thumb_index_positions(self), link_name=self.link_name)
        mean_dist = fingertip_dist.mean(dim=-1)
        return torch.clamp(1.0 - mean_dist / self.distance_threshold, max=1.0)


@REWARD_TERM_REGISTRY.register()
class ScrewPoseDiffPenalty(RewardTerm):
    entity_name: str = "robot"
    excluded_dof_pattern: str = "thumb"

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.entity_name]
        dof_names = list(getattr(self._robot.options, "dofs_name", ()) or ())
        self._pose_diff_mask = torch.ones(len(dof_names), device=self.device)
        excluded_indices = set(resolve_hand_dof_indices(self._robot, finger=self.excluded_dof_pattern))
        if not excluded_indices:
            pattern = self.excluded_dof_pattern.lower()
            excluded_indices = {i for i, name in enumerate(dof_names) if pattern in name.lower()}
        for idx in excluded_indices:
            if 0 <= idx < len(dof_names):
                self._pose_diff_mask[idx] = 0.0
        self._init_hand_pose = self._robot.get_dofs_pos(self._robot.dofs_idx_local).clone()

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        current_pose = self._robot.get_dofs_pos(self._robot.dofs_idx_local)
        self._init_hand_pose[envs_idx] = current_pose[envs_idx].clone()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        current_pose = self._robot.get_dofs_pos(self._robot.dofs_idx_local)
        is_first_step = self._env.episode_length_buf <= 1
        if is_first_step.any():
            self._init_hand_pose[is_first_step] = current_pose[is_first_step].clone()
        pose_diff = current_pose - self._init_hand_pose
        return ((pose_diff**2) * self._pose_diff_mask).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
class ScrewTorquePenalty(RewardTerm):
    entity_name: str = "robot"

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.entity_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        control_forces = self._robot.get_dofs_control_force(self._robot.dofs_idx_local)
        return (control_forces**2).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
class ScrewWorkPenalty(RewardTerm):
    entity_name: str = "robot"

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.entity_name]
        self._dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        self._prev_dof_pos = self._robot.get_dofs_pos(self._robot.dofs_idx_local).clone()

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        current_pose = self._robot.get_dofs_pos(self._robot.dofs_idx_local)
        self._prev_dof_pos[envs_idx] = current_pose[envs_idx].clone()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        control_forces = self._robot.get_dofs_control_force(self._robot.dofs_idx_local)
        current_pose = self._robot.get_dofs_pos(self._robot.dofs_idx_local)
        dof_vel = (current_pose - self._prev_dof_pos) / self._dt
        is_first_step = self._env.episode_length_buf <= 1
        if is_first_step.any():
            dof_vel[is_first_step] = 0.0
        self._prev_dof_pos.copy_(current_pose)
        abs_power = (torch.abs(control_forces) * torch.abs(dof_vel)).sum(dim=-1)
        return abs_power**2


@TERMINATION_TERM_REGISTRY.register()
class ScrewNutStagnation(TerminationTerm):
    entity_name: str = "obj"
    command_name: str = "rotation_axis"
    history_len: int = 60
    stagnation_eps: float = 0.003

    def build(self) -> None:
        super().build()
        self._history = torch.zeros(self.num_envs, self.history_len, device=self.device)
        self._write_idx = -1

    def compute(self) -> torch.Tensor:
        screw_axis_pos, _ = get_screw_axis_state(self._env, obj_name=self.entity_name, command_name=self.command_name)
        self._write_idx = (self._write_idx + 1) % self.history_len
        self._history[:, self._write_idx] = screw_axis_pos
        is_first_step = self._env.episode_length_buf <= 1
        if is_first_step.any():
            self._history[is_first_step] = screw_axis_pos[is_first_step].unsqueeze(1).expand(-1, self.history_len)
        history_filled = self._env.episode_length_buf >= self.history_len
        pos_variance = torch.var(self._history, dim=1)
        return (pos_variance < self.stagnation_eps) & history_filled


@TERMINATION_TERM_REGISTRY.register()
class ScrewJointLimit(TerminationTerm):
    entity_name: str = "obj"
    dof_name: str | None = None
    dof_idx: int = 0
    enabled: bool = True
    upper_limit: float = 628.3185
    reset_margin: float = 5.0

    def build(self) -> None:
        super().build()
        self._zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._obj = self._env.entities[self.entity_name]
        self._dofs_idx_local = getattr(self._obj, "dofs_idx_local", None)
        self._resolved_dof_idx = self.dof_idx
        if self.dof_name is not None:
            try:
                self._resolved_dof_idx = _get_entity_dof_names(self._obj).index(self.dof_name)
            except ValueError:
                self._resolved_dof_idx = None
        self._enabled = (
            self.enabled
            and self._resolved_dof_idx is not None
            and self._dofs_idx_local is not None
            and len(self._dofs_idx_local) > self._resolved_dof_idx
        )

    def compute(self) -> torch.Tensor:
        if not self._enabled:
            return self._zeros
        try:
            screw_dof_pos = self._obj.get_dofs_pos(self._dofs_idx_local)[:, self._resolved_dof_idx]
        except Exception:
            return self._zeros
        return screw_dof_pos > (self.upper_limit - self.reset_margin)


@TERMINATION_TERM_REGISTRY.register()
class ScrewHandleFall(TerminationTerm):
    obj_name: str = "obj"
    link_name: str | None = None
    min_alignment: float = 0.2
    local_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    world_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def build(self) -> None:
        super().build()
        _build_alignment_term(
            self,
            obj_name=self.obj_name,
            link_name=self.link_name,
            local_up_axis=self.local_up_axis,
            world_up_axis=self.world_up_axis,
        )

    def compute(self) -> torch.Tensor:
        return _compute_alignment_term(self) < self.min_alignment


@TERMINATION_TERM_REGISTRY.register()
class ScrewFingerDistance(TerminationTerm):
    obj_name: str = "obj"
    robot_name: str = "robot"
    link_name: str | None = None
    thumb_link: str | None = None
    index_link: str | None = None
    distance_threshold: float = 0.05

    def build(self) -> None:
        super().build()
        _build_thumb_index_grasp_term(
            self,
            obj_name=self.obj_name,
            robot_name=self.robot_name,
            link_name=self.link_name,
            thumb_link=self.thumb_link,
            index_link=self.index_link,
        )

    def compute(self) -> torch.Tensor:
        fingertip_dist = get_grasp_target_distance(self._obj, _thumb_index_positions(self), link_name=self.link_name)
        return (fingertip_dist[:, 0] > self.distance_threshold) | (fingertip_dist[:, 1] > self.distance_threshold)


@TERMINATION_TERM_REGISTRY.register()
class ScrewLowContactStagnation(TerminationTerm):
    obj_name: str = "obj"
    link_name: str | None = None
    history_len: int = 60
    force_threshold: float = 1e-3

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.obj_name]
        self._zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        try:
            target = resolve_grasp_target_link(self._obj, link_name=self.link_name)
            resolved_idx = _resolve_link_index(target)
        except Exception:
            resolved_idx = None
        self._target_link_idx = -1 if resolved_idx is None else resolved_idx
        self._history = torch.zeros(self.num_envs, self.history_len, device=self.device, dtype=torch.bool)
        self._write_idx = -1

    def compute(self) -> torch.Tensor:
        try:
            contact_forces = self._obj.get_links_net_contact_force()
            if contact_forces.ndim == 3 and 0 <= self._target_link_idx < contact_forces.shape[1]:
                agg_force_mag = torch.norm(contact_forces[:, self._target_link_idx, :], dim=-1)
            else:
                agg_force_mag = torch.norm(contact_forces, dim=-1).amax(dim=-1)
        except (AttributeError, NotImplementedError):
            return self._zeros
        low_contact = agg_force_mag < self.force_threshold
        self._write_idx = (self._write_idx + 1) % self.history_len
        self._history[:, self._write_idx] = low_contact
        is_first_step = self._env.episode_length_buf <= 1
        if is_first_step.any():
            self._history[is_first_step] = low_contact[is_first_step].unsqueeze(1).expand(-1, self.history_len)
        history_filled = self._env.episode_length_buf >= self.history_len
        return torch.all(self._history, dim=1) & history_filled


def _resolve_link_index(link) -> int | None:
    for attr in ("idx_local", "link_idx_local", "link_idx", "rb_idx_local", "rb_idx", "index", "idx"):
        value = getattr(link, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return int(value.item())
    return None


__all__ = [
    "ScrewFingerDistance",
    "ScrewHandleFall",
    "ScrewJointLimit",
    "ScrewLowContactStagnation",
    "ScrewNutStagnation",
    "ScrewPoseDiffPenalty",
    "ScrewSurfaceDistanceReward",
    "ScrewTorquePenalty",
    "ScrewVerticalAlignmentPenalty",
    "ScrewWorkPenalty",
    "filter_hand_dof_names",
    "get_grasp_target_distance",
    "get_grasp_target_pos",
    "get_screw_axis_state",
    "resolve_grasp_target_link",
    "resolve_grasp_target_link_name",
    "resolve_hand_dof_indices",
    "resolve_hand_link",
    "screw_angvel_penalty",
    "screw_object_dist_penalty",
    "screw_pc_z_dist_penalty",
    "screw_rotation_reward",
    "screw_rotation_state_obs",
    "screw_rotation_velocity_obs",
]
