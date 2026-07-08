"""
What this script does:
- Randomly perturbs robot DOFs (and object pose if not fixed-base), then keeps only grasps
  that pass 3 checks: surface distance, penetration, and stability checks.
- Runs in parallel over many envs until `num_grasps` valid samples are collected.

Fast start:
- `python scripts/grasps_generator.py --task=<task> --config=<yaml> --num_grasps=<N> [--num_envs=<M>]`
- Typical config path: `conf/sample_grasps/<task>_<robot>.yaml`

Config precedence (important):
- CLI args are parsed first.
- Then `grasp_generation` in YAML overrides matching CLI/default values.
- Keep grasp-generation settings in YAML for reproducible runs.

Copy-paste YAML template (`grasp_generation` section):
```yaml
grasp_generation:
  # Sampling scale and output.
  num_envs: 256
  num_grasps: 64
  output_file: src/assets/grasps/<task>_<robot>_grasps_64.pt

  # Exploration ranges around scene defaults.
  perturb_range: 0.25
  obj_pos_shift: "-0.02,0.02,-0.02,0.02,-0.02,0.02"   # x_min,x_max,y_min,y_max,z_min,z_max (m)
  obj_euler_range: "-10,10,-10,10,-180,180"           # roll/pitch/yaw offset bounds in degrees

  # Task 1: enough existing surface-distance sensors read below threshold.
  required_num_surface_distance_fingers: 3
  surface_distance_threshold: 0.08

  # Task 2: reject deep penetration (object contact + finger self-contact).
  max_penetration: 0.003

  # Task 3: post-check stability (skipped for fixed-base objects).
  stability_steps: 10
  max_displacement: 0.01
  save_object_pose: true
```

Output:
- Saves `.pt` with keys: `joint_pos`, `robot_pos`, `robot_quat`, `obj_pos`, `obj_quat`, `task`
  (plus `geom_idx` for heterogeneous objects).
"""

import argparse
import fnmatch
import warnings
from pathlib import Path

import eden as en
import genesis as gs
import genesis.utils.geom as gu
import torch
from eden.envs.base import RLEnvBase
from eden.options import EventManagerOptions

from registry import get_argparser, get_task_config_from_args, load_config_section


def parse_numeric_sequence(value, expected_len, *, name, cast_type=float):
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    if len(items) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} values, got {items}")
    return [cast_type(item) for item in items]


def expand_root_pose_override(value, expected_len, num_envs, device, *, name):
    values = parse_numeric_sequence(value, expected_len, name=name)
    return torch.tensor(values, dtype=gs.tc_float, device=device).reshape(1, expected_len).expand(num_envs, -1).clone()


def resolve_surface_distance_sensors(env, pattern="*surface_distance*"):
    """Return configured surface-distance sensors, raising if none are available."""
    sensor_names = [name for name in env.sensors if fnmatch.fnmatch(name, pattern)]
    if not sensor_names:
        available = sorted(env.sensors)
        raise RuntimeError(
            f"No sensors matched {pattern!r}. Configure surface-distance sensors before running grasp generation. "
            f"Available sensors: {available}"
        )
    return {name: env.sensors[name] for name in sorted(sensor_names)}


def _sensor_min_distance_per_env(sensor, num_envs):
    data = sensor.read()
    if not isinstance(data, tuple):
        data = (data,)

    mins = []
    for tensor in data:
        if not isinstance(tensor, torch.Tensor):
            continue
        values = tensor.float()
        if values.ndim == 0:
            values = values.reshape(1, 1).expand(num_envs, 1)
        elif values.ndim == 1:
            values = (
                values.reshape(num_envs, -1)
                if values.numel() == num_envs
                else values.reshape(1, -1).expand(num_envs, -1)
            )
        else:
            values = values.reshape(values.shape[0], -1)
            if values.shape[0] == 1 and num_envs > 1:
                values = values.expand(num_envs, -1)
        if values.shape[0] != num_envs:
            raise RuntimeError(
                f"Surface-distance sensor returned {tuple(tensor.shape)}, expected first dimension {num_envs}."
            )
        mins.append(values.amin(dim=1))

    if not mins:
        raise RuntimeError(f"Surface-distance sensor {sensor!r} did not return tensor readings.")
    return torch.stack(mins, dim=1).amin(dim=1)


def check_surface_distance(
    num_envs,
    surface_distance_sensors,
    surface_distance_threshold,
    required_num_surface_distance_fingers,
):
    """Condition 1: enough existing surface-distance sensors below threshold."""
    close_by_sensor_name = {}
    for sensor_name, sensor in surface_distance_sensors.items():
        min_distance = _sensor_min_distance_per_env(sensor, num_envs)
        close_by_sensor_name[sensor_name] = min_distance <= surface_distance_threshold

    close_count = torch.zeros(num_envs, dtype=torch.int, device=next(iter(close_by_sensor_name.values())).device)
    for is_close in close_by_sensor_name.values():
        close_count += is_close.int()
    return close_count >= required_num_surface_distance_fingers, close_count


def build_link_index_tensor(link_names, hand, device):
    if not link_names:
        return None
    link_indices = sorted({hand.get_link(link_name).idx for link_name in link_names})
    return torch.tensor(link_indices, dtype=gs.tc_int, device=device)


def membership_mask(values, allowed_values):
    if allowed_values is None or allowed_values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return (values.unsqueeze(-1) == allowed_values).any(dim=-1)


def get_max_penetration_per_env(num_envs, contacts, device, extra_mask=None):
    penetration = contacts.get("penetration")
    if not isinstance(penetration, torch.Tensor):
        return torch.zeros(num_envs, device=device), False

    penetration = torch.clamp(penetration, min=0.0)
    valid_mask = contacts.get("valid_mask")
    if isinstance(valid_mask, torch.Tensor) and valid_mask.shape == penetration.shape:
        penetration = torch.where(valid_mask, penetration, torch.zeros_like(penetration))

    if extra_mask is not None:
        if extra_mask.shape != penetration.shape:
            return torch.zeros(num_envs, device=device), False
        penetration = torch.where(extra_mask, penetration, torch.zeros_like(penetration))

    # Contacts API can return 1D or 2D penetration; unsupported ranks are treated as unavailable.
    if penetration.ndim not in (1, 2):
        return torch.zeros(num_envs, device=device), False
    if penetration.numel() == 0:
        return torch.zeros(num_envs, device=device, dtype=torch.float32), True
    if penetration.ndim == 2:
        return penetration.amax(dim=1), True
    return penetration.max().expand(num_envs), True


def check_object_stable(obj, initial_pos, max_displacement):
    """Check if object hasn't moved too much."""
    current_pos = obj.get_pos()
    displacement = torch.norm(current_pos - initial_pos, p=2, dim=-1)
    return displacement <= max_displacement


def check_penetration(num_envs, hand, obj, max_penetration, device):
    """Check that hand-object contact penetration stays under the configured threshold."""
    try:
        contacts = hand.get_contacts(with_entity=obj)
    except (AttributeError, NotImplementedError):
        return (
            torch.ones(num_envs, device=device, dtype=torch.bool),
            torch.zeros(num_envs, device=device),
            False,
        )

    max_penetration_values, penetration_available = get_max_penetration_per_env(num_envs, contacts, device)
    if not penetration_available:
        return (
            torch.ones(num_envs, device=device, dtype=torch.bool),
            torch.zeros(num_envs, device=device),
            False,
        )
    return max_penetration_values <= max_penetration, max_penetration_values, True


def check_finger_self_penetration(num_envs, hand, finger_link_indices, max_penetration, device):
    """Check that self-collision penetration between finger links stays under threshold."""
    try:
        contacts = hand.get_contacts(with_entity=hand)
    except (AttributeError, NotImplementedError):
        return (
            torch.ones(num_envs, device=device, dtype=torch.bool),
            torch.zeros(num_envs, device=device),
            False,
        )

    link_a = contacts.get("link_a")
    link_b = contacts.get("link_b")
    if not isinstance(link_a, torch.Tensor) or not isinstance(link_b, torch.Tensor):
        return (
            torch.ones(num_envs, device=device, dtype=torch.bool),
            torch.zeros(num_envs, device=device),
            False,
        )

    finger_pair_mask = membership_mask(link_a, finger_link_indices) & membership_mask(link_b, finger_link_indices)
    finger_pair_mask = finger_pair_mask & (link_a != link_b)
    max_penetration_values, penetration_available = get_max_penetration_per_env(
        num_envs,
        contacts,
        device,
        extra_mask=finger_pair_mask,
    )
    if not penetration_available:
        return (
            torch.ones(num_envs, device=device, dtype=torch.bool),
            torch.zeros(num_envs, device=device),
            False,
        )

    return max_penetration_values <= max_penetration, max_penetration_values, True


def main():
    """Main function for grasp sampling."""

    parser = get_argparser(description="Sample and validate grasps for manipulation tasks.")
    parser.add_argument("--viewer", "-v", action="store_true", help="Show viewer during grasp sampling")
    parser.add_argument(
        "--bottom_aligned", action="store_true", help="Keep the object bottom exactly on the floor plane"
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments to use")
    parser.add_argument("--num_grasps", type=int, default=None, help="Target number of grasps to collect")
    parser.add_argument("--perturb_range", type=float, default=0.4, help="Joint angle perturbation range")
    parser.add_argument("--stability_steps", type=int, default=10, help="Steps to test stability")
    parser.add_argument(
        "--required_num_surface_distance_fingers",
        type=int,
        default=2,
        help="Condition 1 minimum number of surface-distance sensors below threshold",
    )
    parser.add_argument(
        "--surface_distance_threshold",
        type=float,
        default=0.06,
        help="Condition 1 surface-distance threshold in meters",
    )
    parser.add_argument(
        "--max_penetration",
        type=float,
        default=0.0,
        help="Maximum allowed hand-object penetration depth in meters (0.0 enforces no penetration)",
    )
    parser.add_argument("--max_displacement", type=float, default=0.01, help="Max object displacement")
    parser.add_argument(
        "--obj_pos_shift",
        type=str,
        default="-0.02,0.02,-0.02,0.02,-0.02,0.02",
        help="Object position shift ranges (comma-separated): x_min,x_max,y_min,y_max,z_min,z_max",
    )
    parser.add_argument(
        "--obj_euler_range",
        type=str,
        default="-10,10,-10,10,-180,180",
        help="Object euler angle ranges in degrees (comma-separated): x_min,x_max,y_min,y_max,z_min,z_max",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output path for saved grasps (.pt). Defaults to grasps/{task}_{robot}_grasps_{num_grasps}.pt",
    )
    parser.add_argument(
        "--save_object_pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store/replay object root pose in the grasp file. Disable this when the task object "
            "must stay at its built-in reset pose and only the hand pose should be restored."
        ),
    )
    args = parser.parse_args()

    grasp_generation_overrides = load_config_section([args.config], "grasp_generation")
    # YAML values intentionally override CLI/defaults so grasp sampling stays self-contained per config file.
    for key, value in grasp_generation_overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown grasp_generation setting: {key}")
        setattr(args, key, value)

    if args.num_grasps is None:
        parser.error("Specify --num_grasps or set grasp_generation.num_grasps in the YAML config.")

    required_num_surface_distance_fingers = args.required_num_surface_distance_fingers
    if required_num_surface_distance_fingers < 1:
        raise ValueError(
            f"required_num_surface_distance_fingers must be >= 1, got {required_num_surface_distance_fingers}."
        )
    obj_pos_shift_values = parse_numeric_sequence(args.obj_pos_shift, 6, name="obj_pos_shift")
    obj_euler_range_values = parse_numeric_sequence(args.obj_euler_range, 6, name="obj_euler_range")

    # Initialize Eden
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/temp",
    )

    # Load task config
    config = get_task_config_from_args(args, upload_logs=False)
    num_grasps = args.num_grasps
    perturb_range = args.perturb_range
    stability_steps = args.stability_steps
    surface_distance_threshold = args.surface_distance_threshold
    max_penetration = args.max_penetration
    max_displacement = args.max_displacement
    output_file = args.output_file
    save_object_pose = args.save_object_pose
    obj_pos_shift_x = [obj_pos_shift_values[0], obj_pos_shift_values[1]]
    obj_pos_shift_y = [obj_pos_shift_values[2], obj_pos_shift_values[3]]
    obj_pos_shift_z = [obj_pos_shift_values[4], obj_pos_shift_values[5]]
    obj_euler_range_x = [obj_euler_range_values[0], obj_euler_range_values[1]]
    obj_euler_range_y = [obj_euler_range_values[2], obj_euler_range_values[3]]
    obj_euler_range_z = [obj_euler_range_values[4], obj_euler_range_values[5]]

    print("Sampling grasps:")
    print(
        f"num_grasps={num_grasps} perturb_range={perturb_range} "
        f"surface_distance_threshold={surface_distance_threshold} stability_steps={stability_steps} "
        f"required_num_surface_distance_fingers={required_num_surface_distance_fingers} "
        f"max_penetration={max_penetration} max_displacement={max_displacement} "
        f"save_object_pose={save_object_pose}"
    )
    print(
        f"obj_pos_shift: x={obj_pos_shift_x}, y={obj_pos_shift_y}, z={obj_pos_shift_z} "
        f"obj_euler: x={obj_euler_range_x}, y={obj_euler_range_y}, z={obj_euler_range_z}"
    )

    # Grasp sampling should not depend on task reset events (e.g., LoadGraspPose),
    # otherwise new tasks cannot bootstrap their first grasp file.
    config.event_options = EventManagerOptions()
    if args.num_envs is not None:
        config.env_options.num_envs = args.num_envs
    config.env_options.background_color = (1.0, 1.0, 1.0)

    # Create environment
    env = RLEnvBase.from_config(config, show_viewer=args.viewer)
    env.pre_build_setup()
    n_envs = config.env_options.num_envs

    # Get entities
    hand = env.entities["robot"]
    obj = env.entities["obj"]

    # Finger links are only needed for the self-penetration filter.
    fingertip_link_names = hand.options.metadata.fingertip_links

    env.scene.build(n_envs=n_envs)
    env.post_build_setup()

    surface_distance_sensors = resolve_surface_distance_sensors(env)
    print(f"Surface-distance sensors: {sorted(surface_distance_sensors)}")
    if required_num_surface_distance_fingers > len(surface_distance_sensors):
        raise ValueError(
            "required_num_surface_distance_fingers cannot exceed the number of matched "
            f"surface-distance sensors: {required_num_surface_distance_fingers} > {len(surface_distance_sensors)}."
        )

    # Check if object is fixed-base (e.g., screwdriver URDF welded to world)
    is_fixed_base = getattr(obj.options, "is_fixed_base", False)
    if is_fixed_base:
        print("Object is fixed-base: skipping object pose randomization and stability check")

    # Check if object is heterogeneous and build env-to-geom mapping
    is_heterogeneous = obj.is_heterogeneous
    env_to_geom_idx = None
    unique_geom_indices = None
    if is_heterogeneous:
        # Build mapping from environment index to geom index
        env_to_geom_idx = torch.zeros(n_envs, dtype=gs.tc_int, device=gs.device)
        for geom in obj.links[0].geoms:
            active_envs = geom.active_envs_idx
            env_to_geom_idx[active_envs] = geom.idx
        unique_geom_indices = torch.unique(env_to_geom_idx).cpu().tolist()
        print(f"Object is heterogeneous with {len(obj.links[0].geoms)} different morphs")
        print(f"Unique geom indices: {unique_geom_indices}")
    else:
        print("Object is homogeneous (same geometry across all environments)")

    # Get hand configuration
    num_dofs = len(hand.dofs_idx_local)
    dofs_min_limit, dofs_max_limit = hand.get_dofs_limit()
    default_dofs_pos = hand.default_dofs_pos.clone()

    print(f"Environment: {args.task}")
    print(f"Hand: {hand.options.file}")
    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of fingertips: {len(fingertip_link_names)}")

    device = env.device

    # Reset environment
    env.reset()

    # Storage for successful grasps
    if is_heterogeneous:
        # Store grasps per geom
        successful_grasps_per_geom = {geom_idx: [] for geom_idx in unique_geom_indices}
        total_tested_per_geom = {geom_idx: 0 for geom_idx in unique_geom_indices}
    else:
        successful_grasps = []
    total_tested = 0
    num_iters = 0

    # Prepare bounds for sampling
    finger_link_names = list(getattr(hand.options.metadata, "finger_links", [])) + list(fingertip_link_names)
    finger_link_indices = build_link_index_tensor(finger_link_names, hand, device)

    perturb_min = torch.full((num_dofs,), -perturb_range, device=device)
    perturb_max = torch.full((num_dofs,), perturb_range, device=device)

    obj_pos_shift_bounds = torch.tensor(
        [
            obj_pos_shift_x,
            obj_pos_shift_y,
            obj_pos_shift_z,
        ],
        device=device,
        dtype=torch.float32,
    )

    obj_euler_bounds = torch.tensor(
        [
            obj_euler_range_x,
            obj_euler_range_y,
            obj_euler_range_z,
        ],
        device=device,
        dtype=torch.float32,
    )

    # Function to check if we have enough grasps
    def has_enough_grasps():
        if is_heterogeneous:
            return all(len(grasps) >= num_grasps for grasps in successful_grasps_per_geom.values())
        else:
            return len(successful_grasps) >= num_grasps

    while not has_enough_grasps():
        num_iters += 1

        env.reset()
        base_obj_root_pos = obj.get_pos().clone()
        base_obj_root_quat = obj.get_quat().clone()

        # Generate random perturbations for each environment
        perturbations = torch.rand(n_envs, num_dofs, device=device) * (perturb_max - perturb_min) + perturb_min
        perturbed_dofs = default_dofs_pos + perturbations
        perturbed_dofs = torch.clamp(perturbed_dofs, dofs_min_limit, dofs_max_limit)
        hand.set_dofs_pos(perturbed_dofs, hand.dofs_idx_local)

        # Sample random perturbations to object pose (skip for fixed-base objects)
        if not is_fixed_base:
            obj_pos_shift = (
                torch.rand(n_envs, 3, device=device) * (obj_pos_shift_bounds[:, 1] - obj_pos_shift_bounds[:, 0])
                + obj_pos_shift_bounds[:, 0]
            )
            obj_euler_perturbations = (
                torch.rand(n_envs, 3, device=device) * (obj_euler_bounds[:, 1] - obj_euler_bounds[:, 0])
                + obj_euler_bounds[:, 0]
            )
            obj_quat_offsets = gu.xyz_to_quat(obj_euler_perturbations, rpy=True, degrees=True)
            obj_quat = gu.transform_quat_by_quat(base_obj_root_quat, obj_quat_offsets)
            obj_pos = base_obj_root_pos + obj_pos_shift
            if args.bottom_aligned:
                # Use the rotated object's actual AABB minimum to keep its bottom on the floor.
                obj.set_quat(obj_quat, relative=False)
                aabb = obj.get_AABB()
                obj_pos[:, 2] = base_obj_root_pos[:, 2] - aabb[:, 0, 2]
            obj.set_pos(obj_pos)
            obj.set_quat(obj_quat, relative=False)

        env.scene.step()

        # Check condition 1: enough configured surface-distance sensors read below threshold.
        surface_distance_ok, surface_distance_count = check_surface_distance(
            n_envs,
            surface_distance_sensors,
            surface_distance_threshold,
            required_num_surface_distance_fingers,
        )

        # Check condition 2: No penetration between hand-object or finger-finger pairs.
        obj_penetration_ok, max_obj_penetration_per_env, penetration_available = check_penetration(
            n_envs,
            hand,
            obj,
            max_penetration,
            device,
        )
        self_penetration_ok, max_self_penetration_per_env, self_penetration_available = check_finger_self_penetration(
            n_envs,
            hand,
            finger_link_indices,
            max_penetration,
            device,
        )
        if not penetration_available:
            warnings.warn(
                "Hand-object penetration data unavailable from contact API; penetration filtering disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self_penetration_available:
            warnings.warn(
                "Finger self-penetration data unavailable from contact API; self-collision filtering disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
        penetration_ok = obj_penetration_ok & self_penetration_ok
        max_penetration_per_env = torch.maximum(max_obj_penetration_per_env, max_self_penetration_per_env)

        # Check condition 3: Object stability (skip for fixed-base objects — inherently stable)
        if is_fixed_base:
            stability_ok = torch.ones(n_envs, device=device, dtype=torch.bool)
        else:
            initial_obj_pos = obj.get_pos().clone()
            for _ in range(stability_steps):
                hand.control_dofs_pos(perturbed_dofs, hand.dofs_idx_local)
                env.scene.step()
            stability_ok = check_object_stable(obj, initial_obj_pos, max_displacement)

        # Extract successful grasps
        valid_grasps = surface_distance_ok & penetration_ok & stability_ok
        valid_indices = valid_grasps.nonzero(as_tuple=False).squeeze(-1)

        if len(valid_indices) > 0:
            current_robot_pos = hand.get_pos()
            current_robot_quat = hand.get_quat()
            current_obj_pos = obj.get_pos()
            current_obj_quat = obj.get_quat()
            for idx in valid_indices:
                grasp_data = {
                    "joint_pos": perturbed_dofs[idx].cpu(),
                    "robot_pos": current_robot_pos[idx].cpu(),
                    "robot_quat": current_robot_quat[idx].cpu(),
                    "obj_pos": current_obj_pos[idx].cpu(),
                    "obj_quat": current_obj_quat[idx].cpu(),
                }
                if is_heterogeneous:
                    geom_idx = env_to_geom_idx[idx].cpu().item()
                    grasp_data["geom_idx"] = torch.tensor(geom_idx)
                    # Only add if this geom still needs more grasps
                    if len(successful_grasps_per_geom[geom_idx]) < num_grasps:
                        successful_grasps_per_geom[geom_idx].append(grasp_data)
                else:
                    successful_grasps.append(grasp_data)

        # Update per-geom testing counts
        if is_heterogeneous:
            for geom_idx in unique_geom_indices:
                # Count how many envs have this geom
                num_envs_with_geom = (env_to_geom_idx == geom_idx).sum().item()
                total_tested_per_geom[geom_idx] += num_envs_with_geom

        total_tested += n_envs
        num_successful = len(valid_indices)
        num_surface_distance = int(surface_distance_ok.sum().item())
        num_penetration = int(penetration_ok.sum().item())
        num_stability = int(stability_ok.sum().item())
        mean_surface_distance = float(surface_distance_count.float().mean().item())
        mean_penetration = float(max_penetration_per_env.mean().item())
        mean_obj_penetration = float(max_obj_penetration_per_env.mean().item())
        mean_self_penetration = float(max_self_penetration_per_env.mean().item())
        max_penetration_value = float(max_penetration_per_env.max().item())
        max_obj_penetration_value = float(max_obj_penetration_per_env.max().item())
        max_self_penetration_value = float(max_self_penetration_per_env.max().item())

        # Print progress
        if is_heterogeneous:
            progress_str = " | ".join(
                [f"Geom {g}: {len(successful_grasps_per_geom[g])}/{num_grasps}" for g in unique_geom_indices]
            )
            print(
                f"Iter {num_iters}: {num_successful}/{n_envs} valid grasps "
                f"| surface={num_surface_distance}/{n_envs} "
                f"no_pen={num_penetration}/{n_envs} stable={num_stability}/{n_envs} "
                f"| mean_surface={mean_surface_distance:.2f} "
                f"mean_pen={mean_penetration:.5f} max_pen={max_penetration_value:.5f} "
                f"(obj={mean_obj_penetration:.5f}/{max_obj_penetration_value:.5f}, "
                f"self={mean_self_penetration:.5f}/{max_self_penetration_value:.5f}) | {progress_str}"
            )
        else:
            print(
                f"Iter {num_iters}: {num_successful}/{n_envs} valid grasps "
                f"| surface={num_surface_distance}/{n_envs} "
                f"no_pen={num_penetration}/{n_envs} stable={num_stability}/{n_envs} "
                f"| mean_surface={mean_surface_distance:.2f} "
                f"mean_pen={mean_penetration:.5f} max_pen={max_penetration_value:.5f} "
                f"(obj={mean_obj_penetration:.5f}/{max_obj_penetration_value:.5f}, "
                f"self={mean_self_penetration:.5f}/{max_self_penetration_value:.5f}) "
                f"| Total: {len(successful_grasps)}/{num_grasps}"
            )

    # Save results
    # Flatten per-geom grasps into single list for heterogeneous objects
    if is_heterogeneous:
        successful_grasps = []
        for geom_idx in unique_geom_indices:
            successful_grasps.extend(successful_grasps_per_geom[geom_idx][:num_grasps])

    if successful_grasps:
        output_file = (
            Path(output_file) if output_file else Path("grasps") / f"{args.task}_{args.robot}_grasps_{num_grasps}.pt"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Stack tensors for each component
        joint_pos_tensor = torch.stack([g["joint_pos"] for g in successful_grasps])
        robot_pos_tensor = torch.stack([g["robot_pos"] for g in successful_grasps])
        robot_quat_tensor = torch.stack([g["robot_quat"] for g in successful_grasps])
        save_data = {
            "joint_pos": joint_pos_tensor,
            "robot_pos": robot_pos_tensor,
            "robot_quat": robot_quat_tensor,
            "task": args.task,
            "save_object_pose": save_object_pose,
        }
        if save_object_pose:
            obj_pos_tensor = torch.stack([g["obj_pos"] for g in successful_grasps])
            obj_quat_tensor = torch.stack([g["obj_quat"] for g in successful_grasps])
            save_data["obj_pos"] = obj_pos_tensor
            save_data["obj_quat"] = obj_quat_tensor
            save_data["obj_pose_frame"] = "root"

        # Add geom_idx if object is heterogeneous
        if is_heterogeneous:
            geom_idx_tensor = torch.stack([g["geom_idx"] for g in successful_grasps])
            save_data["geom_idx"] = geom_idx_tensor
            print(geom_idx_tensor)
            print("geom idxs:", geom_idx_tensor.unique())

        torch.save(save_data, output_file)

        print(f"\n✓ Saved {len(successful_grasps)} successful grasps to {output_file}")
        if is_heterogeneous:
            for geom_idx in unique_geom_indices:
                geom_count = (geom_idx_tensor == geom_idx).sum().item()
                geom_tested = total_tested_per_geom[geom_idx]
                success_rate = (geom_count / geom_tested * 100) if geom_tested > 0 else 0
                print(f"  Geom {geom_idx}: {geom_count} grasps, success rate: {success_rate:.1f}%")
        else:
            print(f"  Success rate: {len(successful_grasps) / total_tested * 100:.1f}%")
        print(f"  Joint pos shape: {joint_pos_tensor.shape}")
        print(f"  Robot pos shape: {robot_pos_tensor.shape}")
        print(f"  Robot quat shape: {robot_quat_tensor.shape}")
        if save_object_pose:
            print(f"  Object pos shape: {obj_pos_tensor.shape}")
            print(f"  Object quat shape: {obj_quat_tensor.shape}")
        else:
            print("  Object pose: not saved (runtime task default pose will be used)")
        if is_heterogeneous:
            print(f"  Geom idx shape: {geom_idx_tensor.shape}")
            print(f"  Unique geom indices in saved data: {geom_idx_tensor.unique().tolist()}")
        print(f"  Example grasp joint_pos: {[round(x, 4) for x in successful_grasps[0]['joint_pos'].tolist()]}")
        if save_object_pose:
            print(f"  Example grasp obj_pos: {[round(x, 4) for x in successful_grasps[0]['obj_pos'].tolist()]}")
        if is_heterogeneous:
            print(f"  Example grasp geom_idx: {successful_grasps[0]['geom_idx'].item()}")
    else:
        print("\n✗ No successful grasps found!")


if __name__ == "__main__":
    main()
