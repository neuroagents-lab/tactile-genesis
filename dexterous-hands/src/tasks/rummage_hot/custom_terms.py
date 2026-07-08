from __future__ import annotations

import math
from typing import TYPE_CHECKING

import eden as en
import genesis.utils.geom as gu
import torch
from eden.constants import MetricDirection, MetricMode
from eden.managers import (
    COMMAND_TERM_REGISTRY,
    EVENT_TERM_REGISTRY,
    METRIC_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    TERMINATION_TERM_REGISTRY,
    CommandTerm,
    EventTerm,
    ObservationTerm,
    RewardTerm,
    TerminationTerm,
)
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


@OBSERVATION_TERM_REGISTRY.register()
class TemperatureDiffReading(en.observations.SensorRead):
    """Per-step delta of concatenated sensor readings (times ``scale``); ``_last_absolute`` mirrors the latest absolute read."""

    scale: float = 1.0

    def build(self) -> None:
        super().build()
        self._prev: torch.Tensor | None = None

    def _read_absolute(self) -> torch.Tensor:
        """Same concatenation as :meth:`SensorRead.compute` without mutating ``_cached`` / ``_prev``."""
        sensor_readings: list[torch.Tensor] = []
        for sensor in self._sensors:
            data = sensor.read_ground_truth() if self.read_ground_truth else sensor.read()
            if not isinstance(data, tuple):
                data = (data,)
            for tensor in data:
                if tensor.ndim > 2:
                    tensor = tensor.flatten(start_dim=1)
                sensor_readings.append(tensor.float())
        return torch.cat(sensor_readings, dim=-1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if self._prev is None:
            return
        current = self._read_absolute()
        self._last_absolute = current.detach().clone()
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev[envs_idx] = current[envs_idx].detach().clone()

    def compute(self, *args, **kwargs) -> torch.Tensor:
        current = self._read_absolute()
        self._last_absolute = current.detach().clone()
        if self._prev is None:
            self._prev = current.detach().clone()
            out = torch.zeros_like(current)
            self._cached = out
            return out
        diff = (current - self._prev) * self.scale
        self._prev.copy_(current.detach())
        self._cached = diff
        return diff


def _axis_cell_centers(lo: float, hi: float, spacing: float, device: torch.device) -> torch.Tensor:
    """1D lattice cell centers from ``lo`` to ``hi`` with step ``spacing`` (first center at ``lo + spacing/2``)."""
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    span = hi - lo
    if span <= 1e-9:
        return torch.tensor([(lo + hi) * 0.5], device=device, dtype=torch.float32)
    centers: list[float] = []
    x = lo + spacing * 0.5
    while x <= hi + 1e-8:
        centers.append(x)
        x += spacing
    if not centers:
        centers.append((lo + hi) * 0.5)
    return torch.tensor(centers, device=device, dtype=torch.float32)


@EVENT_TERM_REGISTRY.register()
class RandomlyPlaceInGrid(EventTerm):
    """Place entities in a 3D grid expressed in a (possibly tilted) bin's local frame.

    Positions ``range_{x,y,z}`` and the yaw rotation are sampled in the bin's local frame,
    then composed with ``bin_quat`` (in ``(w, x, y, z)`` order) to produce world-frame
    poses. Defaults to identity, which reproduces the original axis-aligned behavior.
    """

    entity_names: tuple[str, ...] = tuple()
    range_x: tuple[float, float] = (-0.2, 0.2)
    range_y: tuple[float, float] = (-0.2, 0.2)
    range_z: tuple[float, float] = (-0.2, 0.2)
    spacing: float = 0.1
    jiggle_std: float = 0.01
    range_yaw: tuple[float, float] = (-math.pi, math.pi)
    bin_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def build(self) -> None:
        super().build()
        self.entities: list[Entity] = self._env.get_entities(list(self.entity_names))
        if not self.entities:
            raise ValueError(f"RandomlyPlaceInGrid: no entities found for names: {self.entity_names}")

        self._bin_quat = torch.tensor(self.bin_quat, device=self.device, dtype=torch.float32)

        default_quats: list[torch.Tensor] = []
        for ent in self.entities:
            _, quat = ent.get_default_root_pose()
            ent.set_quat(quat)
            default_quats.append(quat)
        self._entity_quats = torch.stack(default_quats, dim=0)

        rx0, rx1 = self.range_x
        ry0, ry1 = self.range_y
        rz0, rz1 = self.range_z
        dev = self.device
        xs = _axis_cell_centers(rx0, rx1, self.spacing, dev)
        ys = _axis_cell_centers(ry0, ry1, self.spacing, dev)
        zs = _axis_cell_centers(rz0, rz1, self.spacing, dev)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        slots = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

        n_ent = len(self.entities)
        num_slots = slots.shape[0]
        if num_slots < n_ent:
            raise ValueError(
                f"RandomlyPlaceInGrid: grid has {num_slots} slots but {n_ent} entities; "
                f"increase ranges or spacing (x={self.range_x} y={self.range_y} z={self.range_z}, spacing={self.spacing})."
            )
        self._slots = slots

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True, prefer_slice=False)
        if n_envs == 0:
            return
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            envs_idx = torch.where(envs_idx)[0]
        if isinstance(envs_idx, slice):
            envs_idx = torch.arange(self.num_envs, device=self.device)[envs_idx]

        device = self.device
        slots = self._slots
        num_slots = slots.shape[0]
        n_ent = len(self.entities)
        bin_quat = self._bin_quat

        pick = torch.argsort(torch.rand(n_envs, num_slots, device=device), dim=-1)[:, :n_ent]
        pos_local = slots[pick]
        if self.jiggle_std != 0.0:
            pos_local = pos_local + torch.randn(n_envs, n_ent, 3, device=device, dtype=torch.float32) * self.jiggle_std
        # Rotate sampled positions from bin-local frame into world frame.
        pos_world = gu.transform_by_quat(pos_local, bin_quat)

        yaw = sample_uniform(*self.range_yaw, size=(n_envs, n_ent), device=str(device))
        yaw_eulers = torch.zeros((n_envs, n_ent, 3), device=device, dtype=torch.float32)
        yaw_eulers[:, :, 2] = yaw
        yaw_quat = gu.xyz_to_quat(yaw_eulers.reshape(-1, 3), rpy=True, degrees=False).reshape(n_envs, n_ent, 4)

        for j, entity in enumerate(self.entities):
            entity.set_pos(pos_world[:, j, :], envs_idx=envs_idx)

            base_q = self._entity_quats[j, envs_idx]
            yq = yaw_quat[:, j, :]
            # World quat = R_bin @ R_yaw @ R_base: yaw spins around the bin's local Z,
            # then the whole entity is tilted with the bin.
            local_q = gu.transform_quat_by_quat(base_q, yq)
            quat = gu.transform_quat_by_quat(local_q, bin_quat)
            entity.set_quat(quat, envs_idx=envs_idx)


@REWARD_TERM_REGISTRY.register()
def temperature_reading_reward(
    env: EnvBase,
    *,
    obs_name: str = "temp_sensors",
    target_temperature: float = 100.0,
) -> torch.Tensor:
    """Linear shaping: progress of the best-matching sensor toward ``target_temperature``.

    Uses the sensor with smallest current absolute error to the target; reward is
    ``reward_scale * (|T_prev - target| - |T_curr - target|)`` for that sensor.
    ``T_prev`` is the observation term's stored previous absolute read (updated when
    observations compute, after rewards — do not call ``compute()`` here).

    For plain ``SensorRead`` terms (no ``_read_absolute``), falls back to
    ``exp(-min_i |T_i - target|)``.
    """
    term = env.observation_manager.get_term(obs_name)
    read = getattr(term, "_read_absolute", None)
    if read is None:
        temperatures = term.compute()
        min_abs_err, _ = torch.abs(temperatures - target_temperature).min(dim=-1)
        return torch.exp(-min_abs_err)
    curr = read()
    prev = getattr(term, "_prev", None)
    n_env = curr.shape[0]
    if prev is None:
        return torch.zeros(n_env, device=curr.device, dtype=curr.dtype)
    err_c = torch.abs(curr - target_temperature)
    err_p = torch.abs(prev - target_temperature)
    k = err_c.argmin(dim=-1)
    batch = torch.arange(n_env, device=curr.device)
    closer = err_p[batch, k] - err_c[batch, k]
    return closer


@OBSERVATION_TERM_REGISTRY.register()
class RevealedObjectPosition(ObservationTerm):
    """Object position (relative to the robot base), revealed only after first touch.

    Each episode the position is hidden (returns zeros) until any fingertip link of
    ``robot_name`` makes force-bearing contact with ``object_name``. From that first
    contact until the next reset, the object's position relative to the robot base
    is returned.
    """

    robot_name: str = "robot"
    object_name: str = "obj"

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._obj = self._env.entities[self.object_name]
        _, ft_idx = self._robot.find_named_links_idx_local(
            self._robot.metadata.fingertip_links, preserve_order=True
        )
        self._fingertip_idx = torch.as_tensor(ft_idx, dtype=torch.long, device=self.device)
        self._revealed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Only ever reassign self._revealed (never update in place): an in-place
        # write would pin it as an inference tensor and crash a later step.
        if envs_idx is None:
            self._revealed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return
        keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep[envs_idx] = False
        self._revealed = self._revealed & keep

    def compute(self, *args, **kwargs) -> torch.Tensor:
        grouped = self._robot.get_grouped_contacts(self._obj, links_a_idx_local=self._fingertip_idx)
        touching = grouped["force_norm_sum"].flatten(start_dim=1).sum(dim=-1) > 0.0
        self._revealed = self._revealed | touching
        rel_pos = self._obj.get_pos() - self._robot.get_pos()
        self._cached = rel_pos * self._revealed.unsqueeze(-1).to(rel_pos.dtype)
        return self._cached


@OBSERVATION_TERM_REGISTRY.register()
class ContactDurationObs(ObservationTerm):
    """Per-fingertip continuous-contact duration with whichever ball it currently touches.

    Mirrors the bookkeeping in :class:`SameBallContactReward` so the teacher can
    observe the hidden timer the windowed ``ball_contact`` reward depends on. Per
    fingertip, tracks the ball carrying the largest aggregate contact force and
    how many consecutive steps that *same* ball has been held; switching ball or
    losing contact resets that fingertip's timer.

    Output, shape ``(num_envs, n_fingertips + 1)``: per-fingertip contact
    duration in seconds (clamped to ``max_seconds``, then multiplied by
    ``duration_scale`` to keep values O(1)), followed by a 0/1 flag for whether
    any fingertip is currently touching the hot target (index 0 of ``ball_names``).
    """

    robot_name: str = "robot"
    ball_names: tuple[str, ...] = ()
    max_seconds: float = 3.0
    duration_scale: float = 0.5
    contact_threshold: float = 0.001

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._balls = [self._env.entities[name] for name in self.ball_names]
        self._fingertip_link_names = list(self._robot.metadata.fingertip_links)
        _, ft_idx = self._robot.find_named_links_idx_local(
            self._fingertip_link_names, preserve_order=True
        )
        self._fingertip_idx = torch.as_tensor(ft_idx, dtype=torch.long, device=self.device)
        self._n_fingertips = self._fingertip_idx.numel()
        self._dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        self._dist_sensors = [
            [
                self._env.sensors[f"priv_ftball_dist__{link}__{ball}"]
                for ball in self.ball_names
            ]
            for link in self._fingertip_link_names
        ]
        # Per-env, per-fingertip: currently-held ball index (-1 == none) and step count.
        self.reset()

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Reassign-only (never in place): an in-place write would pin these as
        # inference tensors and crash a later step.
        shape = (self.num_envs, self._n_fingertips)
        if envs_idx is None:
            self._contact_ball = torch.full(shape, -1, dtype=torch.long, device=self.device)
            self._contact_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
            return
        keep = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)
        keep[envs_idx] = False
        self._contact_ball = torch.where(keep, self._contact_ball, torch.full_like(self._contact_ball, -1))
        self._contact_steps = torch.where(keep, self._contact_steps, torch.zeros_like(self._contact_steps))

    def _fingertip_ball_distance(self) -> torch.Tensor:
        """(num_envs, n_fingertips, n_balls) surface distance per fingertip-ball pair (meters)."""
        per_ft: list[torch.Tensor] = []
        for sensors_for_ft in self._dist_sensors:
            per_ball = [sensor.read().reshape(self.num_envs, 1) for sensor in sensors_for_ft]
            per_ft.append(torch.cat(per_ball, dim=-1))
        return torch.stack(per_ft, dim=1)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        dist = self._fingertip_ball_distance()  # (E, F, B)
        min_dist, nearest_ball = dist.min(dim=-1)
        touching = min_dist < self.contact_threshold  # (E, F)
        cur_ball = torch.where(touching, nearest_ball, torch.full_like(self._contact_ball, -1))

        held_same = touching & (cur_ball == self._contact_ball)
        self._contact_steps = torch.where(
            held_same,
            self._contact_steps + 1,
            touching.to(self._contact_steps.dtype),  # restart at 1 on a fresh contact, else 0
        )
        self._contact_ball = cur_ball

        duration = self._contact_steps.to(torch.float32) * self._dt
        duration = duration.clamp(max=self.max_seconds) * self.duration_scale
        # Hot target is index 0 of ball_names ("obj").
        hot_flag = (touching & (cur_ball == 0)).any(dim=-1, keepdim=True).to(duration.dtype)
        self._cached = torch.cat([duration, hot_flag], dim=-1)
        return self._cached


@REWARD_TERM_REGISTRY.register()
class SameBallContactReward(RewardTerm):
    """Reward fingertips holding continuous contact with a single ball.

    Per fingertip, tracks which ball it currently touches (the one carrying the
    largest aggregate contact force) and how many consecutive steps that contact
    with that *same* ball has lasted. While the continuous contact duration lies
    within ``[min_seconds, max_seconds]`` the fingertip contributes ``1.0`` to the
    reward; shorter or longer contacts contribute nothing. Switching to a
    different ball or losing contact resets that fingertip's timer.

    The per-step reward is the number of fingertips currently inside the rewarded
    duration window (i.e. summed across fingertips).

    If ``unbounded_ball_idx`` is set, contacts with that ball (an index into
    ``ball_names``) are not capped by ``max_seconds`` — they keep paying out as
    long as the contact persists past ``min_seconds``.
    """

    robot_name: str = "robot"
    ball_names: tuple[str, ...] = ()
    min_seconds: float = 1.0
    max_seconds: float = 2.0
    unbounded_ball_idx: int | None = None

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._balls = [self._env.entities[name] for name in self.ball_names]
        _, ft_idx = self._robot.find_named_links_idx_local(
            self._robot.metadata.fingertip_links, preserve_order=True
        )
        self._fingertip_idx = torch.as_tensor(ft_idx, dtype=torch.long, device=self.device)
        self._n_fingertips = self._fingertip_idx.numel()
        self._dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        # Per-env, per-fingertip: currently-held ball index (-1 == none) and step count.
        self.reset()

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Only ever reassign the state buffers (never update them in place): an
        # in-place write would pin them as inference tensors and crash a later step.
        shape = (self.num_envs, self._n_fingertips)
        if envs_idx is None:
            self._contact_ball = torch.full(shape, -1, dtype=torch.long, device=self.device)
            self._contact_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
            return
        keep = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)
        keep[envs_idx] = False
        self._contact_ball = torch.where(keep, self._contact_ball, torch.full_like(self._contact_ball, -1))
        self._contact_steps = torch.where(keep, self._contact_steps, torch.zeros_like(self._contact_steps))

    def _fingertip_ball_force(self) -> torch.Tensor:
        """(num_envs, n_fingertips, n_balls) aggregate contact-force norm per fingertip-ball pair."""
        per_ball: list[torch.Tensor] = []
        for ball in self._balls:
            grouped = self._robot.get_grouped_contacts(ball, links_a_idx_local=self._fingertip_idx)
            # force_norm_sum: (num_envs, n_fingertips, n_ball_links) -> sum across the ball's links.
            per_ball.append(grouped["force_norm_sum"].sum(dim=-1))
        return torch.stack(per_ball, dim=-1)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        force = self._fingertip_ball_force()  # (E, F, B)
        touching = force.amax(dim=-1) > 0.0  # (E, F)
        cur_ball = torch.where(touching, force.argmax(dim=-1), torch.full_like(self._contact_ball, -1))

        held_same = touching & (cur_ball == self._contact_ball)
        self._contact_steps = torch.where(
            held_same,
            self._contact_steps + 1,
            touching.to(self._contact_steps.dtype),  # restart at 1 on a fresh contact, else 0
        )
        self._contact_ball = cur_ball

        duration = self._contact_steps.to(torch.float32) * self._dt
        in_window = (duration >= self.min_seconds) & (duration <= self.max_seconds)
        if self.unbounded_ball_idx is not None:
            in_window = in_window | ((duration >= self.min_seconds) & (cur_ball == self.unbounded_ball_idx))
        return in_window.to(torch.float32).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
class DistinctBallCoverageReward(RewardTerm):
    """Reward touching each distinct ball once per episode.

    A ball is "covered" the first step its continuous force-bearing contact with
    any fingertip has lasted at least ``min_seconds``; that step contributes
    ``1.0`` to the reward and the ball is not credited again until the env
    resets. The per-step reward is the number of balls newly covered this step,
    so the episodic total equals the number of distinct balls rummaged. This
    makes "touch every ball" reward-maximising, countering the local optimum of
    parking on a single ball to farm :class:`SameBallContactReward`.
    """

    robot_name: str = "robot"
    ball_names: tuple[str, ...] = ()
    min_seconds: float = 0.2

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._balls = [self._env.entities[name] for name in self.ball_names]
        _, ft_idx = self._robot.find_named_links_idx_local(
            self._robot.metadata.fingertip_links, preserve_order=True
        )
        self._fingertip_idx = torch.as_tensor(ft_idx, dtype=torch.long, device=self.device)
        self._n_balls = len(self._balls)
        self._dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        # Per-env, per-ball: consecutive force-bearing-contact steps and a credited flag.
        self.reset()

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Reassign-only (never in place): an in-place write would pin these as
        # inference tensors and crash a later step.
        shape = (self.num_envs, self._n_balls)
        if envs_idx is None:
            self._contact_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
            self._credited = torch.zeros(shape, dtype=torch.bool, device=self.device)
            return
        keep = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)
        keep[envs_idx] = False
        self._contact_steps = torch.where(keep, self._contact_steps, torch.zeros_like(self._contact_steps))
        self._credited = self._credited & keep

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        per_ball: list[torch.Tensor] = []
        for ball in self._balls:
            grouped = self._robot.get_grouped_contacts(ball, links_a_idx_local=self._fingertip_idx)
            # force_norm_sum: (num_envs, n_fingertips, n_ball_links) -> any force-bearing contact.
            per_ball.append(grouped["force_norm_sum"].flatten(start_dim=1).sum(dim=-1))
        touching = torch.stack(per_ball, dim=-1) > 0.0  # (E, n_balls)

        self._contact_steps = torch.where(
            touching, self._contact_steps + 1, torch.zeros_like(self._contact_steps)
        )
        qualified = self._contact_steps.to(torch.float32) * self._dt >= self.min_seconds
        newly = qualified & ~self._credited
        self._credited = self._credited | newly
        return newly.to(torch.float32).sum(dim=-1)


@METRIC_TERM_REGISTRY.register(
    direction=MetricDirection.HIB,
    metric_mode=MetricMode.INTERVAL,
)
def max_temperature_metric(env: EnvBase, *, obs_name: str = "temp_sensors") -> torch.Tensor:
    """Hottest absolute temperature currently sensed by any temperature sensor, per env.

    ``obs_name`` is a temperature observation term; its ``_read_absolute`` is used so
    the metric tracks raw temperatures rather than the term's per-step deltas.
    """
    term = env.observation_manager.get_term(obs_name)
    read = getattr(term, "_read_absolute", None)
    temperatures = read() if read is not None else term.compute()
    return temperatures.max(dim=-1).values


@TERMINATION_TERM_REGISTRY.register()
class ObjectLiftedHold(TerminationTerm):
    """Success termination: the object has stayed lifted for ``hold_seconds``.

    An env terminates once the object's lowest point (AABB bottom) has stayed at or
    above ``lift_height`` for ``hold_seconds`` of continuous simulated time. Any step
    with the object below that height resets the env's hold timer.
    """

    object_name: str = "obj"
    lift_height: float = 0.2
    hold_seconds: float = 2.0

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.object_name]
        aabb = self._obj.get_AABB()
        self._center_to_bottom_z = (aabb[:, 1, 2] - aabb[:, 0, 2]) / 2.0
        self._dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        self.reset()

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Reassign-only (never in place) so the buffer can't become a pinned
        # inference tensor and crash a later step.
        if envs_idx is None:
            self._hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            return
        keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep[envs_idx] = False
        self._hold_steps = self._hold_steps * keep

    def compute(self, *args, **kwargs) -> torch.Tensor:
        obj_bottom_z = self._obj.get_pos()[:, 2] - self._center_to_bottom_z
        lifted = obj_bottom_z >= self.lift_height
        self._hold_steps = torch.where(lifted, self._hold_steps + 1, torch.zeros_like(self._hold_steps))
        return self._hold_steps.to(torch.float32) * self._dt >= self.hold_seconds


@COMMAND_TERM_REGISTRY.register()
class BallGraspSequenceCommand(CommandTerm):
    """Per-env deck of ball indices; the current target advances when lifted.

    On each env reset the deck is reshuffled to a fresh permutation of
    ``ball_names`` (random sampling without replacement) and each ball's
    initial z is captured. The current target ``_target_idx`` is
    ``deck[deck_pos]``. The current target is considered "lifted" when its
    root z exceeds its captured initial z by ``lift_delta`` (meters), held
    continuously for ``hold_seconds``. For non-hot targets that advances
    ``deck_pos`` (and flags ``_last_advance``); for the hot target it sets
    ``_hot_success`` (consumed by a termination + bonus reward) and the deck
    is not advanced further.

    Per-fingertip-per-ball contact bookkeeping (``_contact_ball``,
    ``_contact_steps``, ``_target_contact_time``) is still maintained for
    shaping reward terms that read it.

    ``command`` is a one-hot ``(num_envs, n_balls)`` of the current target,
    suitable for ``en.observations.generated_commands``.
    """

    robot_name: str = "robot"
    ball_names: tuple[str, ...] = ()
    hot_ball_idx: int = 0
    lift_delta: float = 0.02
    hold_seconds: float = 1.0
    contact_threshold: float = 0.001
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._balls = [self._env.entities[name] for name in self.ball_names]
        self._n_balls = len(self._balls)
        self._fingertip_link_names = list(self._robot.metadata.fingertip_links)
        _, ft_idx = self._robot.find_named_links_idx_local(
            self._fingertip_link_names, preserve_order=True
        )
        self._fingertip_idx = torch.as_tensor(ft_idx, dtype=torch.long, device=self.device)
        self._n_fingertips = self._fingertip_idx.numel()
        self._dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        self._dist_sensors = [
            [
                self._env.sensors[f"priv_ftball_dist__{link}__{ball}"]
                for ball in self.ball_names
            ]
            for link in self._fingertip_link_names
        ]

        ft_shape = (self.num_envs, self._n_fingertips)
        self._contact_ball = torch.full(ft_shape, -1, dtype=torch.long, device=self.device)
        self._contact_steps = torch.zeros(ft_shape, dtype=torch.long, device=self.device)
        # Cumulative time each fingertip has spent in contact with the *current*
        # commanded target; only resets on deck advance or env reset, so tap-and-
        # release cannot refund the per-target reward budget.
        self._target_contact_time = torch.zeros(ft_shape, dtype=torch.long, device=self.device)
        self._deck = torch.zeros((self.num_envs, self._n_balls), dtype=torch.long, device=self.device)
        self._deck_pos = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initial_z = torch.zeros((self.num_envs, self._n_balls), dtype=torch.float32, device=self.device)
        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_advance = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hot_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._command_buf = torch.zeros((self.num_envs, self._n_balls), dtype=torch.float32, device=self.device)

        self.stats["target_idx"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.stats["deck_pos"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.stats["success"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    @property
    def command(self) -> torch.Tensor:
        return self._command_buf

    def _refresh_command_buf(self) -> None:
        self._command_buf = torch.nn.functional.one_hot(
            self._target_idx, num_classes=self._n_balls
        ).to(torch.float32)

    def _envs_idx_to_bool(self, envs_idx) -> torch.Tensor | None:
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            return envs_idx
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if isinstance(envs_idx, slice):
            envs_idx = torch.arange(self.num_envs, device=self.device)[envs_idx]
        mask[envs_idx] = True
        return mask

    def _resample_command(self, envs_idx) -> None:
        """Fresh-deck reset for the given envs (called from base ``reset``).

        Always reassign tensors (never in-place) so they cannot become pinned
        inference tensors and crash a later step.
        """
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        mask = self._envs_idx_to_bool(envs_idx)

        fresh_perm = torch.argsort(
            torch.rand(self.num_envs, self._n_balls, device=self.device), dim=-1
        ).to(torch.long)
        mask_2d = mask.unsqueeze(-1)
        self._deck = torch.where(mask_2d, fresh_perm, self._deck)
        self._deck_pos = torch.where(mask, torch.zeros_like(self._deck_pos), self._deck_pos)
        self._target_idx = torch.where(mask, self._deck[:, 0], self._target_idx)

        # Capture the post-reset z of every ball as the per-episode lift baseline.
        # CommandManager.reset runs after EventManager.reset, so balls are already
        # placed in their scattered positions by RandomlyPlaceInGrid.
        ball_z = torch.stack([b.get_pos()[:, 2] for b in self._balls], dim=-1)
        self._initial_z = torch.where(mask_2d, ball_z, self._initial_z)

        ft_mask = mask.unsqueeze(-1)
        self._contact_ball = torch.where(ft_mask, torch.full_like(self._contact_ball, -1), self._contact_ball)
        self._contact_steps = torch.where(ft_mask, torch.zeros_like(self._contact_steps), self._contact_steps)
        self._target_contact_time = torch.where(
            ft_mask, torch.zeros_like(self._target_contact_time), self._target_contact_time
        )
        self._hold_steps = torch.where(mask, torch.zeros_like(self._hold_steps), self._hold_steps)
        self._last_advance = torch.where(mask, torch.zeros_like(self._last_advance), self._last_advance)
        self._hot_success = torch.where(mask, torch.zeros_like(self._hot_success), self._hot_success)

        self._refresh_command_buf()

    def _fingertip_ball_distance(self) -> torch.Tensor:
        """(num_envs, n_fingertips, n_balls) surface distance per fingertip-ball pair (meters)."""
        per_ft: list[torch.Tensor] = []
        for sensors_for_ft in self._dist_sensors:
            per_ball = [sensor.read().reshape(self.num_envs, 1) for sensor in sensors_for_ft]
            per_ft.append(torch.cat(per_ball, dim=-1))
        return torch.stack(per_ft, dim=1)

    def _update_command(self) -> None:
        dist = self._fingertip_ball_distance()  # (E, F, B)
        min_dist, nearest_ball = dist.min(dim=-1)
        touching = min_dist < self.contact_threshold  # (E, F)
        cur_ball = torch.where(touching, nearest_ball, torch.full_like(self._contact_ball, -1))

        held_same = touching & (cur_ball == self._contact_ball)
        self._contact_steps = torch.where(
            held_same,
            self._contact_steps + 1,
            touching.to(self._contact_steps.dtype),
        )
        self._contact_ball = cur_ball

        # Cumulative on-target contact time (anti-hack for the decay reward):
        # only goes up while touching the current target, never refunded.
        on_target = cur_ball == self._target_idx.unsqueeze(-1)
        self._target_contact_time = torch.where(
            on_target, self._target_contact_time + 1, self._target_contact_time
        )

        # Lift criterion: current target ball z must exceed its episode-start z
        # by `lift_delta`, held continuously for `hold_seconds`.
        ball_z = torch.stack([b.get_pos()[:, 2] for b in self._balls], dim=-1)
        target_z = ball_z.gather(1, self._target_idx.unsqueeze(-1)).squeeze(-1)
        initial_target_z = self._initial_z.gather(1, self._target_idx.unsqueeze(-1)).squeeze(-1)
        lifted = (target_z - initial_target_z) >= self.lift_delta
        self._hold_steps = torch.where(
            lifted, self._hold_steps + 1, torch.zeros_like(self._hold_steps)
        )
        held_long_enough = self._hold_steps.to(torch.float32) * self._dt >= self.hold_seconds

        is_hot = self._target_idx == self.hot_ball_idx
        self._hot_success = is_hot & held_long_enough

        advance_mask = held_long_enough & ~is_hot
        self._deck_pos = torch.where(
            advance_mask,
            (self._deck_pos + 1).clamp(max=self._n_balls - 1),
            self._deck_pos,
        )
        self._target_idx = self._deck.gather(1, self._deck_pos.unsqueeze(1)).squeeze(1)
        advance_mask_ft = advance_mask.unsqueeze(-1).expand_as(self._contact_steps)
        self._contact_steps = torch.where(
            advance_mask_ft, torch.zeros_like(self._contact_steps), self._contact_steps
        )
        self._contact_ball = torch.where(
            advance_mask_ft, torch.full_like(self._contact_ball, -1), self._contact_ball
        )
        self._target_contact_time = torch.where(
            advance_mask_ft, torch.zeros_like(self._target_contact_time), self._target_contact_time
        )
        self._hold_steps = torch.where(advance_mask, torch.zeros_like(self._hold_steps), self._hold_steps)

        self._last_advance = advance_mask
        self._refresh_command_buf()
        self.stats["target_idx"][:] = self._target_idx.to(torch.float32)
        self.stats["deck_pos"][:] = self._deck_pos.to(torch.float32)
        self.stats["success"][:] = self._hot_success.to(torch.float32)


@OBSERVATION_TERM_REGISTRY.register()
class CurrentTargetPositionObs(ObservationTerm):
    """Position of the currently commanded ball, relative to the robot base."""

    robot_name: str = "robot"
    command_name: str = "grasp_target"
    ball_names: tuple[str, ...] = ()

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._cmd = self._env.command_manager.get_term(self.command_name)
        self._balls = [self._env.entities[name] for name in self.ball_names]

    def compute(self, *args, **kwargs) -> torch.Tensor:
        ball_pos = torch.stack([b.get_pos() for b in self._balls], dim=1)  # (E, B, 3)
        idx = self._cmd._target_idx.view(-1, 1, 1).expand(-1, 1, 3)
        target_pos = ball_pos.gather(1, idx).squeeze(1)
        self._cached = target_pos - self._robot.get_pos()
        return self._cached


@REWARD_TERM_REGISTRY.register()
class TargetLiftReward(RewardTerm):
    """Reward lifting the currently commanded ball above its episode-start z.

    Reads ``_target_idx`` and ``_initial_z`` from the
    :class:`BallGraspSequenceCommand` term. Returns
    ``clamp((target_z - initial_z) / lift_delta, 0, 1)``, so it ramps up
    smoothly until the ball has been lifted by ``lift_delta`` meters.
    Whichever ball is currently the target (hot or decoy) is the one rewarded.
    """

    command_name: str = "grasp_target"
    ball_names: tuple[str, ...] = ()
    lift_delta: float = 0.02

    def build(self) -> None:
        super().build()
        self._cmd = self._env.command_manager.get_term(self.command_name)
        self._balls = [self._env.entities[name] for name in self.ball_names]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        ball_z = torch.stack([b.get_pos()[:, 2] for b in self._balls], dim=-1)
        target_idx = self._cmd._target_idx.unsqueeze(-1)
        target_z = ball_z.gather(1, target_idx).squeeze(-1)
        initial_z = self._cmd._initial_z.gather(1, target_idx).squeeze(-1)
        return torch.clamp((target_z - initial_z) / self.lift_delta, min=0.0, max=1.0)


@REWARD_TERM_REGISTRY.register()
def command_last_advance(env: "EnvBase", *, command_name: str = "grasp_target") -> torch.Tensor:
    """1.0 on the step the grasp-target command just advanced past a non-hot ball."""
    return env.command_manager.get_term(command_name)._last_advance.to(torch.float32)


@REWARD_TERM_REGISTRY.register()
def target_contact_decay_reward(
    env: "EnvBase",
    *,
    command_name: str = "grasp_target",
    hold_seconds: float = 1.0,
) -> torch.Tensor:
    """Per-step shaping for touching the currently commanded target, decaying with cumulative on-target time.

    Per fingertip, contribution is ``max(0, 1 - cumulative_target_time / hold_seconds)``
    while currently in contact with the target ball, else 0. ``cumulative_target_time``
    is the command term's ``_target_contact_time`` counter, which only resets when
    the deck advances or the env resets — so tap-and-release cannot refund the
    per-target reward budget.
    """
    cmd = env.command_manager.get_term(command_name)
    cumulative = cmd._target_contact_time.to(torch.float32) * cmd._dt
    decay = (1.0 - cumulative / hold_seconds).clamp(min=0.0)
    on_target_now = cmd._contact_ball == cmd._target_idx.unsqueeze(-1)
    return (decay * on_target_now.to(decay.dtype)).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def non_hot_staleness_penalty(
    env: "EnvBase",
    *,
    command_name: str = "grasp_target",
    hold_seconds: float = 1.0,
) -> torch.Tensor:
    """Count of fingertips currently in continuous contact with a non-hot ball for > ``hold_seconds``.

    Reads the command term's existing per-fingertip contact bookkeeping
    (``_contact_ball`` / ``_contact_steps``), so updates here track the same
    state machine used to drive grasp-target advancement.
    """
    cmd = env.command_manager.get_term(command_name)
    durations = cmd._contact_steps.to(torch.float32) * cmd._dt
    non_hot = (cmd._contact_ball >= 0) & (cmd._contact_ball != cmd.hot_ball_idx)
    stale = non_hot & (durations > hold_seconds)
    return stale.to(torch.float32).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def command_hot_success_reward(env: "EnvBase", *, command_name: str = "grasp_target") -> torch.Tensor:
    """1.0 on the step the hot target was held lifted long enough (also triggers termination)."""
    return env.command_manager.get_term(command_name)._hot_success.to(torch.float32)


@TERMINATION_TERM_REGISTRY.register()
def command_hot_success(env: "EnvBase", *, command_name: str = "grasp_target") -> torch.Tensor:
    """Terminate when the grasp-target command flags the hot ball as held lifted."""
    return env.command_manager.get_term(command_name)._hot_success
