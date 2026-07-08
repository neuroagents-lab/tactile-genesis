"""
Load task environment and visualize the sensor placements and sensor readings.

Usage:
    python sensor_viewer.py --task in_hand_repose
"""

from typing import TYPE_CHECKING

import eden as en
import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from eden.envs.base import RLEnvBase
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.vis.keybindings import Key, KeyAction, Keybind

from registry import get_argparser, get_task_config_from_args, make_runner
from tactile_sensors import TACTILE_SENSORS

if TYPE_CHECKING:
    from genesis.engine.sensors.base_sensor import Sensor


# Per-type vector-field plot tuning: (scale_factor, max_magnitude, vector description).
_VECTOR_FIELD_PLOT_OPTS: dict[str, tuple[float, float, str]] = {
    "elastomer": (1.0, 0.005, "marker displacement"),
    "force_torque": (0.01, 1.0, "force"),
    "force": (0.01, 1.0, "force"),
    "proximity": (0.2, 1.0, "force"),
}

# Rotation applied to every vector-field plot's 3D inputs (positions + per-frame vectors)
# so the projected 2D plot ends up rotated 90 degrees clockwise. ``gu.orthogonals`` builds
# a right-handed ``(u, v, normal)`` basis for the projection, so a -pi/2 rotation about
# ``plot_normal`` is equivalent to a 90-degree CW rotation in the plotted axes. Kept in
# sync between this script and scripts/hand_tactile_sandbox.py.
PLOT_ROTATION_ANGLE = -np.pi / 2


def _plot_rotation_quat(plot_normal) -> torch.Tensor:
    """Quaternion (wxyz) that rotates 3D vectors by ``PLOT_ROTATION_ANGLE`` about ``plot_normal``."""
    n = np.asarray(plot_normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    half = PLOT_ROTATION_ANGLE / 2.0
    s = float(np.sin(half))
    q = np.array([float(np.cos(half)), n[0] * s, n[1] * s, n[2] * s], dtype=np.float32)
    return torch.as_tensor(q, device=gs.device)


def _align_normal_quat(from_vec: torch.Tensor, to_vec: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) that rotates unit ``from_vec`` onto unit ``to_vec`` via the shortest arc.

    Used to flatten each sensor's probe patch against the viewing plane: rotates the
    sensor's mean probe normal onto the plot's view normal so probes that sit at an
    angle to the camera (e.g. the thumb) end up parallel to the viewing plane.
    """
    device, dtype = from_vec.device, from_vec.dtype
    f = from_vec / torch.linalg.norm(from_vec).clamp(min=1e-12)
    t = to_vec.to(device=device, dtype=dtype)
    t = t / torch.linalg.norm(t).clamp(min=1e-12)
    dot = float((f * t).sum())
    if dot > 1.0 - 1e-9:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    if dot < -1.0 + 1e-9:
        helper = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        if abs(float((f * helper).sum())) > 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        axis = torch.linalg.cross(f, helper)
        axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
        return torch.cat([torch.zeros(1, device=device, dtype=dtype), axis])
    half = f + t
    half = half / torch.linalg.norm(half).clamp(min=1e-12)
    w = (f * half).sum().unsqueeze(0)
    xyz = torch.linalg.cross(f, half)
    return torch.cat([w, xyz])


def _read_sensor_vectors(sensor_type: str, sensor: "Sensor") -> torch.Tensor:
    """Read one sensor's per-probe 3D vectors as a ``(n_probes, 3)`` tensor in the sensor link frame."""
    data = sensor.read()
    # ElastomerTaxel.read() returns the displacement tensor directly; KinematicTaxel /
    # ProximityTaxel return a NamedTuple whose `.force` field holds the per-probe force.
    vectors = data if sensor_type == "elastomer" else data.force
    if vectors.ndim == 3:  # ([n_envs,] n_probes, 3) -> take env 0
        vectors = vectors[0]
    return vectors.reshape(-1, 3)


def _plot_all_sensors(
    scene: gs.Scene,
    sensor_type: str,
    sensors: list["Sensor"],
    plot_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> None:
    """Plot every probe of every sensor in one combined vector field, projected onto a 2D plane.

    Probe positions and per-probe vectors are transformed from each sensor's link-local frame into
    a common world frame so all sensors share one coordinate system. Each sensor's patch is then
    rotated about its centroid so its mean probe normal aligns with ``plot_normal`` (this flattens
    off-axis patches such as the thumb against the viewing plane), and the whole plot is rotated by
    ``PLOT_ROTATION_ANGLE`` about ``plot_normal`` so the layout appears 90 degrees CW in the
    matplotlib axes. Positions are captured once at recording start (the plotter holds them fixed),
    so the projection reflects the hand pose at that moment -- vectors still update live.
    """
    if not IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping plot setup.")
        return
    if not sensors:
        print(f"No '{sensor_type}' sensors found; skipping plot setup.")
        return

    if sensor_type not in _VECTOR_FIELD_PLOT_OPTS:
        # Scalar sensors (depth / agg_force / bool) have no per-probe vector to project, so
        # neither per-sensor flattening nor plot rotation apply here.
        scene.start_recording(
            lambda: tuple(sensor.read().max() for sensor in sensors),
            gs.recorders.MPLLinePlot(
                title=f"{sensors[0].__class__.__name__} max reading",
                labels=[str(i) for i in range(len(sensors))],
                x_label="step",
                y_label="value",
                history_length=200,
            ),
        )
        return

    scale_factor, max_magnitude, desc = _VECTOR_FIELD_PLOT_OPTS[sensor_type]
    plot_rot_q = _plot_rotation_quat(plot_normal)
    plot_normal_t = torch.as_tensor(np.asarray(plot_normal, dtype=np.float32).reshape(3), device=gs.device)

    # World-frame probe positions, concatenated across all sensors (captured once, held fixed).
    # For each sensor: (1) drop radius-0 padding probes, (2) flatten the patch about its
    # centroid by rotating its mean probe normal onto plot_normal, then (3) apply the
    # global CW plot rotation.
    def _valid_probe_mask(sensor: "Sensor") -> torch.Tensor | None:
        """Bool mask of probes with radius > 0 (None when scalar / all-valid -- no filtering needed)."""
        pr = getattr(getattr(sensor, "_options", None), "probe_radius", None)
        if pr is None or isinstance(pr, (int, float)):
            return None
        radii = torch.as_tensor(np.asarray(pr, dtype=np.float32).reshape(-1), device=gs.device)
        if radii.numel() == 0:
            return None
        mask = radii > 0.0
        return mask if not bool(mask.all()) else None

    world_positions = []
    per_sensor: list[tuple["Sensor", torch.Tensor, torch.Tensor | None]] = []  # (sensor, align_quat, valid_mask)
    for sensor in sensors:
        local_pos = sensor.probe_local_pos.reshape(-1, 3)
        link_pos = sensor._link.get_pos().reshape(3)
        link_quat = sensor._link.get_quat().reshape(4)
        world_pos = gu.transform_by_trans_quat(local_pos, link_pos, link_quat)
        valid_mask = _valid_probe_mask(sensor)
        if valid_mask is not None:
            world_pos = world_pos[valid_mask]
        if world_pos.shape[0] == 0:
            continue  # nothing left to plot for this sensor
        local_normal = getattr(sensor, "probe_local_normal", None)
        if local_normal is not None and local_normal.numel() > 0:
            norms = local_normal.reshape(-1, 3)
            if valid_mask is not None:
                norms = norms[valid_mask]
            world_normals = gu.transform_by_quat(norms, link_quat)
            align_q = _align_normal_quat(world_normals.mean(dim=0), plot_normal_t)
        else:
            align_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device, dtype=torch.float32)
        centroid = world_pos.mean(dim=0)
        aligned_pos = centroid + gu.transform_by_quat(world_pos - centroid, align_q)
        world_positions.append(gu.transform_by_quat(aligned_pos, plot_rot_q))
        per_sensor.append((sensor, align_q, valid_mask))
    positions = torch.cat(world_positions, dim=0).detach().cpu().numpy()

    def read_all_sensor_vectors() -> torch.Tensor:
        """Concatenate every sensor's per-probe vectors, in plot frame (world -> patch-flatten -> CW rotate)."""
        chunks = []
        for sensor, align_q, valid_mask in per_sensor:
            vectors = _read_sensor_vectors(sensor_type, sensor)
            if valid_mask is not None:
                vectors = vectors[valid_mask]
            link_quat = sensor._link.get_quat().reshape(4)
            world_v = gu.transform_by_quat(vectors, link_quat)
            aligned_v = gu.transform_by_quat(world_v, align_q)
            chunks.append(gu.transform_by_quat(aligned_v, plot_rot_q))
        return torch.cat(chunks, dim=0)

    scene.start_recording(
        read_all_sensor_vectors,
        gs.recorders.MPLVectorFieldPlot(
            title=f"All {sensors[0].__class__.__name__} probes -- {desc} ({len(sensors)} sensors)",
            positions=positions,
            normal=plot_normal,
            scale_factor=scale_factor,
            max_magnitude=max_magnitude,
        ),
    )


def _debug_print_tactile_sensor(sensor_type: str, sensor: "Sensor", t: float) -> None:
    data = sensor.read()
    if sensor_type == "elastomer":
        print(f"t={t:.2f}s  max|displacement|={data.max():.4f}")
    elif sensor_type == "depth":
        print(f"t={t:.2f}s  max depth={data.max():.4f}")
    elif sensor_type in ("force", "force_torque"):
        print(f"t={t:.2f}s  max|F|={torch.linalg.norm(data.force, axis=-1).max():.4f}")
    elif sensor_type == "proximity":
        magnitude = torch.linalg.norm(data.force, dim=-1)
        print(f"t={t:.2f}s  mean|F|={magnitude.mean():.5f}  max|F|={magnitude.max():.5f}")


def _lookup_obs_value(obs: object, key: str) -> object | None:
    if isinstance(obs, torch.Tensor):
        return None
    try:
        if key in obs:
            return obs[key]
    except (KeyError, TypeError):
        pass
    return None


def _to_printable_obs(obs: object) -> object:
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu()
    return obs


def _debug_print_sensor_obs(obs: object, t: float) -> None:
    sensor_obs = _lookup_obs_value(obs, "tactile_sensors")
    if sensor_obs is None:
        return

    term_obs = _lookup_obs_value(sensor_obs, "tactile_sensors")
    if term_obs is not None:
        sensor_obs = term_obs

    print(f"t={t:.2f}s  sensor obs={_to_printable_obs(sensor_obs)}")


def main():
    """Main function for task viewer."""

    parser = get_argparser(description="View and debug task environments with manual control.")
    parser.add_argument(
        "--plot_normal",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="View axis for the combined sensor plot; probes are projected onto the plane perpendicular to it.",
    )
    args = parser.parse_args()

    # Initialize Eden
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/temp",
    )

    # Load task config
    config = get_task_config_from_args(args, upload_logs=False)
    config.env_options.num_eval_envs = 1
    config.env_options.background_color = (1.0, 1.0, 1.0)

    # Create environment
    env_base = RLEnvBase.from_config(config, show_viewer=True, eval_mode=True)
    env_base.pre_build_setup()

    scene = env_base.scene
    sensors = scene.sim._sensor_manager.sensors
    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=False,
        )
    )
    sensor_type = args.sensors.split("/")[-1]
    sensor_cls_name = TACTILE_SENSORS[sensor_type].sensor_cls.__name__
    for i, sensor in enumerate(sensors):
        print(f"Sensor {i}: {sensor.__class__.__name__}")
    sensors_to_plot = [sensor for sensor in sensors if sensor_cls_name in sensor.__class__.__name__]

    _plot_all_sensors(
        scene,
        sensor_type=sensor_type,
        sensors=sensors_to_plot,
        plot_normal=tuple(args.plot_normal),
    )

    env_base.scene.build(n_envs=1)
    env_base.post_build_setup()

    env = RslRlVecEnvWrapper(env_base, config.runner_options)

    # Load policy from checkpoint
    policy = None
    if args.checkpoint:
        runner = make_runner(env, checkpoint=args.checkpoint)
        policy = runner.get_inference_policy(device=gs.device)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
    )

    try:
        while is_running:
            obs, rewards, dones, *_ = env.step(actions)
            _debug_print_sensor_obs(obs, scene.t * scene.dt)
            for sensor in sensors_to_plot:
                _debug_print_tactile_sensor(sensor_type, sensor, scene.t * scene.dt)
            if policy is not None:
                with torch.no_grad():
                    actions[:] = policy(obs)
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
