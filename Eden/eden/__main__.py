"""Eden command-line interface (training, inference, viewers, and utilities)."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys

import numpy as np


def show_entity(entity_name: str, show_collision: bool = False):
    import torch

    import eden as en
    from eden.constants import ReferenceSource
    from eden.envs.base import EnvBase
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
    from eden.options.materials import RigidMaterialOptions
    from eden.utils.devices.gui_slider import SliderGUI

    # NOTE: look for the corresponding options module in (* is a file name under each directory):
    # 1) eden.options.robots.*.<entity_name>

    # search for the corresponding options module
    entity_options = None
    base_path = os.path.dirname(__file__)
    for obj_module in ["eden.options.robots"]:
        for file in os.listdir(os.path.join(base_path, obj_module.split("eden.")[1].replace(".", "/"))):
            if file.endswith(".py"):
                module_name = file[:-3]
                module = importlib.import_module(f"{obj_module}.{module_name}")
                if hasattr(module, entity_name):
                    entity_options = getattr(module, entity_name)
                    break
        if entity_options is not None:
            break

    if entity_options is None:
        raise ValueError(f"No entity options found for {entity_name}")

    en.init(debug=True)

    # If the entity declares actuated DOFs (RobotOptions subclasses, or
    # articulated EntityOptions like ArcticObjectOptions), wire up an
    # ImplicitPDController and an ImGui SliderGUI so the user can drive each
    # joint from the viewer. Gravity is compensated so joint targets don't sag
    # while the user is dragging sliders. Plain mesh / primitive entities still
    # fall through to the passive visualizer loop.
    dofs_field = getattr(entity_options, "model_fields", {}).get("dofs_name")
    dofs_name = list(dofs_field.default) if dofs_field is not None and isinstance(dofs_field.default, list) else []
    obj_kwargs: dict = dict(is_fixed_base=True, vis_mode="collision" if show_collision else "visual")
    if dofs_name:
        obj_kwargs["material"] = RigidMaterialOptions(gravity_compensation=1.0)

    action_options = (
        ActionManagerOptions(
            joint_controller=ImplicitPDController.configure(
                entity_name="obj",
                dofs_name=dofs_name,
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
            ),
        )
        if dofs_name
        else None
    )

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, show_world_frame=True),
        scene_options=SceneOptions(obj=entity_options(**obj_kwargs)),
        action_options=action_options,
        show_viewer=True,
    )
    env.build()
    env.reset()

    if not dofs_name:
        while True:
            env.scene.visualizer.update()
        return

    entity = env.entities["obj"]
    _, dof_idx = entity.find_named_dofs_idx_local(dofs_name, preserve_order=True)
    lo, hi = entity.get_dofs_limit(dof_idx)
    limits = torch.stack([lo, hi], dim=-1).squeeze(0).cpu()
    init_pos = entity.get_dofs_pos(dof_idx).squeeze(0).cpu()

    slider = SliderGUI(
        viewer=env.scene.viewer,
        dofs_name=dofs_name,
        dofs_pos_limits=limits,
        initial_position=init_pos,
        reset_callback=env.reset,
    )

    while env.scene.viewer.is_alive():
        action = slider.get_command().to(env.device)
        env.step({"joint_controller": action})

    slider.close()


def snap_entity(entity_name: str, show_collision: bool = False, output_path: str | None = None):
    import eden as en
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    # NOTE: look for the corresponding options module in (* is a file name under each directory):
    # 1) eden.options.robots.*.<entity_name>

    # search for the corresponding options module
    entity_options = None
    base_path = os.path.dirname(__file__)
    for obj_module in ["eden.options.robots"]:
        for file in os.listdir(os.path.join(base_path, obj_module.split("eden.")[1].replace(".", "/"))):
            if file.endswith(".py"):
                module_name = file[:-3]
                module = importlib.import_module(f"{obj_module}.{module_name}")
                if hasattr(module, entity_name):
                    entity_options = getattr(module, entity_name)
                    break
        if entity_options is not None:
            break

    if entity_options is None:
        raise ValueError(f"No entity options found for {entity_name}")

    en.init(debug=True)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, show_world_frame=True),
        scene_options=SceneOptions(
            obj=entity_options(is_fixed_base=True, vis_mode="collision" if show_collision else "visual"),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(),
        ),
        show_viewer=False,
    )
    env.build()

    cam = env.cameras["camera"]
    AABB = env.entities["obj"].get_AABB().cpu()
    cam.set_pose(
        pos=(AABB[0, 1] + AABB[0, 1, 2]).numpy() * np.array([2, 2, 1.0]),
        lookat=AABB.mean(-2).numpy().squeeze(0),
    )
    cam.snapshot(output_path)


def clean_up_logs(log_dir: str | None = None):
    # remove all empty log directories
    import glob
    import shutil

    base_path = os.path.dirname(os.path.dirname(__file__))
    log_dir = log_dir or os.path.join(base_path, "logs")

    logs = glob.glob(os.path.join(log_dir, "**"), recursive=True)
    empty_dirs = 0
    removed_dirs = 0
    for path in logs:
        # NOTE: remove empty directories
        if os.path.isdir(path) and not os.listdir(path):
            empty_dirs += 1
            os.rmdir(path)
        # NOTE: remove directories with only config.json in it
        if os.path.isdir(path) and len(os.listdir(path)) == 1 and "config.json" in os.listdir(path):
            shutil.rmtree(path)
            removed_dirs += 1
    print(f"Removed {removed_dirs} directories with only config.json in it")
    print(f"Removed {empty_dirs} empty logs")


def print_task_list():
    """Print available tasks in Eden, sourced from TASK_REGISTRY (auto-discovery + eager registrations)."""
    from eden.tasks import TASK_REGISTRY

    names = sorted(TASK_REGISTRY.list_tasks())
    if not names:
        print("No tasks registered.")
        return

    # Resolve modifier names per task. get_modifiers triggers a module import
    # for auto-discovered tasks; some tasks may fail to import on this machine
    # (missing assets, optional deps, etc.) — surface that as a row tag instead
    # of crashing the whole listing.
    rows: list[tuple[int, str, str]] = []
    for i, name in enumerate(names, 1):
        try:
            mods = TASK_REGISTRY.get_modifiers(name)
            mod_names = ", ".join(type(m).__name__ for m in mods) or "—"
        except Exception as exc:
            mod_names = f"(import error: {type(exc).__name__})"
        rows.append((i, name, mod_names))

    try:
        from prettytable import PrettyTable

        table = PrettyTable(["No.", "Task Name", "Modifiers"])
        table.title = "Available Tasks in Eden"
        table.align["Task Name"] = "l"
        table.align["Modifiers"] = "l"
        # Wrap only the Modifiers column when the terminal is narrower than
        # the natural table width; leave No. and Task Name at natural width
        # so they don't get squeezed into vertical-split single chars.
        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        no_w = max(len("No."), len(str(len(rows))))
        name_w = max(len("Task Name"), max(len(name) for _, name, _ in rows))
        # Budget = terminal − (no_w + name_w) − padding/borders (≈10 chars).
        modifiers_budget = max(20, term_cols - no_w - name_w - 10)
        table.max_width["Modifiers"] = modifiers_budget
        for row in rows:
            table.add_row(row)
        print(table)
    except ImportError:
        # Fall back to plain text so `eden list` works without prettytable.
        width = max(len(name) for _, name, _ in rows)
        for i, name, mods in rows:
            print(f"  {i:>3}. {name:<{width}}  {mods}")


def _validate_task_arg(args, parser):
    """Ensure ``args.task`` is set and registered; print a helpful error otherwise."""
    from eden.tasks import TASK_REGISTRY

    if not args.task:
        parser.error("--task is required (run 'eden list' to see available tasks).")
    if args.task not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.list_tasks())) or "none"
        parser.error(f"unknown task '{args.task}'. Available: {available}")


def train_task(extra_args: list[str]) -> None:
    """``eden train --task <name> [task-mod flags]`` — train a registered task.

    Forwards ``extra_args`` (everything after ``train``) to
    :func:`eden.tasks.parser.get_task_argparser`. The argparser also uses
    ``extra_args`` (not ``sys.argv``) for its ``--task`` peek, so the same
    arg slice drives both mod-flag registration and the final parse.
    Runs the rsl_rl training loop.
    """
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Train a registered task with TaskMod-driven CLI.",
        argv=extra_args,
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="rsl_rl logger backend.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override the task's training num_envs.",
    )
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num_envs must be a positive integer.")

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{args.task}",
        performance_mode=True,
    )

    config = get_task_config_from_args(args, run_name=args.run_name)
    if args.num_envs is not None:
        config.env_options.num_envs = args.num_envs
    if hasattr(config.runner_options, "logger"):
        config.runner_options.logger = args.logger
    config.save_as_json()

    _run_train_rsl_rl(config, checkpoint=args.checkpoint)


def inference_task(extra_args: list[str]) -> None:
    """``eden inference --task <name> --checkpoint <path> [task-mod flags]``.

    Loads a trained checkpoint and runs the eval loop for ``episode_length_s``.
    """
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Run inference for a registered task.",
        argv=extra_args,
    )
    parser.add_argument(
        "--show_viewer",
        action="store_true",
        help="Open the Genesis viewer window during inference.",
    )
    parser.add_argument(
        "--rec",
        action="store_true",
        help="Record an mp4 via a 'rec' camera (saved under the run's log dir).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Cap inference at N env steps (default: one episode, i.e. max_episode_length).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Stop after N completed episodes (counted via env auto-resets).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel eval envs (default: 1).",
    )
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if not args.checkpoint:
        parser.error("--checkpoint is required for 'eden inference'.")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be a positive integer.")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")
    if args.num_envs <= 0:
        parser.error("--num_envs must be a positive integer.")

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{args.task}",
        performance_mode=True,
    )

    config = get_task_config_from_args(args, run_name=args.run_name)
    config.env_options.num_eval_envs = args.num_envs
    if hasattr(config.runner_options, "logger"):
        config.runner_options.logger = "tensorboard"  # never wandb-log a checkpoint replay
    if args.rec:
        if config.cameras_options is None:
            config.cameras_options = CamerasOptions()
        config.cameras_options.rec = CameraOptions(
            cam_pos=(1.5, 1.0, 2.0), cam_lookat=(0.0, 0.0, 1.0), follow_entity_name="robot"
        )
    config.save_as_json()

    detected = _detect_checkpoint_framework(args.checkpoint)
    if detected is not None and detected != "rsl_rl":
        sys.exit(
            f"checkpoint at {args.checkpoint!r} looks like a {detected} checkpoint, "
            f"but this build only supports rsl_rl. Supply an rsl_rl checkpoint."
        )

    _run_eval_rsl_rl(
        config,
        args.checkpoint,
        show_viewer=args.show_viewer,
        record=args.rec,
        max_steps=args.steps,
        max_episodes=args.episodes,
    )


def deploy_task(extra_args: list[str]) -> None:
    """``eden deploy --task <name> --checkpoint <path> [task-mod flags]``.

    Run a trained policy on real hardware. The deployment backend is picked
    automatically from the robot bound to the task's scene entity via
    :func:`eden.extensions.deployment.resolve_deployment_for` — to support a
    new robot, add a single ``bind_deployment(...)`` line in its deployment
    module.

    The function disables training-only randomization (push, friction /
    COM / encoder-bias randomization) and forces ``num_envs=1`` so the
    policy drives a single physical robot. The sim env is still built (with
    the viewer optionally open) so it can mirror robot state for
    visualization while the policy runs on hardware.
    """
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Deploy a trained task policy to a real robot.",
        argv=extra_args,
    )
    parser.add_argument(
        "--show_viewer",
        action="store_true",
        help="Open the Genesis viewer to mirror the hardware state during deployment.",
    )
    parser.add_argument(
        "--deploy_kp_kd_scale",
        type=float,
        default=1.0,
        help="Scale on default Kp/Kd at deployment. Ramp up from 0.2 → 1.0 when first commissioning.",
    )
    parser.add_argument(
        "--entity_name",
        type=str,
        default="robot",
        help="Scene entity to deploy (defaults to the canonical 'robot').",
    )
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if not args.checkpoint:
        parser.error("--checkpoint is required for 'eden deploy'.")

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{args.task}",
        performance_mode=True,
    )

    config = get_task_config_from_args(args, run_name=args.run_name)
    _strip_training_randomization(config)
    config.env_options.num_envs = 1
    config.env_options.num_eval_envs = 1
    config.env_options.episode_length_s = 1e9
    if hasattr(config.runner_options, "logger"):
        config.runner_options.logger = "tensorboard"
    # Snapshot the *deploy* config under a distinct filename so it can't be
    # mistaken for a training-time ``config.json``. The deploy variant has
    # randomization stripped and ``num_envs=1`` — replaying it as a training
    # run would silently produce a randomization-free policy.
    config.save_as_json(os.path.join(en.log_dir, "deploy_config.json"))

    _run_deploy_rsl_rl(
        config,
        checkpoint=args.checkpoint,
        show_viewer=args.show_viewer,
        entity_name=args.entity_name,
        deploy_kp_kd_scale=args.deploy_kp_kd_scale,
    )


# Event terms ``eden deploy`` disables before building the env. These are
# either training-only randomizations (would inject perturbations the policy
# wasn't asked to handle at deploy time) or reset-time state overrides
# (would clobber the real robot's pose / joint config on env.reset()).
# ``disable_term`` raises KeyError for terms the config doesn't declare,
# so the loop stays safe across tasks with different randomization shapes.
_DEPLOY_DISABLED_EVENT_TERMS: tuple[str, ...] = (
    # Training-only perturbations:
    "push_robot",  # interval velocity pushes
    "erfi_50",  # ERFI-50 hybrid torque randomization
    "randomize_torque_noise",  # plain RFI torque noise (if used by other tasks)
    "randomize_motor_strength",  # per-episode motor-strength scaling
    "randomize_kp_kd_gains",
    "randomize_friction",  # foot/feet friction scaling
    "randomize_com",  # base COM shift
    "randomize_mass_shift",
    "randomize_link_mass_scale",
    "randomize_backlash_epsilon",
    "encoder_bias",  # per-episode dofs_pos sensor offset
    "randomize_startup_dofs_pos_bias",
    # Reset-time state overrides — at deploy time the env should mirror the
    # robot's *actual* state, not a randomized fresh start:
    "reset_base",  # random yaw + xy offset on env.reset()
    "reset_dofs",  # random joint perturbation on env.reset()
)


def _strip_training_randomization(config) -> None:
    """Disable training-only event terms for deploy and log what was stripped."""
    import eden as en

    stripped: list[str] = []
    for term in _DEPLOY_DISABLED_EVENT_TERMS:
        try:
            config.event_options.disable_term(term)
            stripped.append(term)
        except KeyError:
            pass

    remaining = list(config.event_options.term_keys())
    if stripped:
        en.logger.info(f"[deploy] Stripped randomization event terms: {stripped}")
    if remaining:
        en.logger.info(f"[deploy] Remaining active event terms: {remaining}")
    else:
        en.logger.info("[deploy] No event terms remain active.")


def _run_deploy_rsl_rl(
    config,
    *,
    checkpoint: str,
    show_viewer: bool,
    entity_name: str,
    deploy_kp_kd_scale: float,
) -> None:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import eden as en
    from eden.envs.wrappers.rsl_rl_env import (
        RslRlVecEnvWrapper,
        _audit_rsl_rl_checkpoint_keys,
        _ensure_modern_rsl_rl_checkpoint,
    )
    from eden.extensions.deployment import resolve_deployment_for

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=show_viewer, eval_mode=True)
    sim_robot = env.unwrapped.entities[entity_name]
    robot_options = getattr(env.unwrapped.config.scene_options, entity_name)

    deployment_cls = resolve_deployment_for(robot_options)
    en.logger.info(f"[deploy] Using {deployment_cls.__name__} for {type(robot_options).__name__}")
    # Instantiating the deployment swaps the sim entity for a ``RobotStateEntity``
    # wrapper but does *not* touch the wire. Hardware contact is deferred to
    # ``dep.connect()`` below, after we've verified the policy loads.
    dep = deployment_cls(
        env.unwrapped,
        deployment_cls.configure(
            entity_name=entity_name,
            sync=True,
            connect_timeout_s=10.0,
            deploy_kp_kd_scale=deploy_kp_kd_scale,
        ),
    )

    # Load the policy *before* opening the DDS channel. If the checkpoint is
    # malformed (key mismatch, shape mismatch, unknown layout the migration
    # helper couldn't fix), we want to abort with the original file untouched
    # rather than leaving the hardware in a half-connected state. ``strict=False``
    # mirrors the inference path's load (``make_rsl_rl_runner``) — migrated
    # legacy checkpoints can have buffer-shape divergences (e.g. the
    # ``EmpiricalNormalization.count`` scalar dtype) that don't affect the
    # policy weights themselves.
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.device)
    checkpoint = _ensure_modern_rsl_rl_checkpoint(checkpoint, device=env.device)
    _audit_rsl_rl_checkpoint_keys(runner, checkpoint, env.device)
    runner.load(checkpoint, map_location=env.device, strict=False)
    policy = runner.get_inference_policy(device=env.device)

    dep.connect()
    try:
        obs = dep.reset()
        while True:
            with torch.no_grad():
                raw_action = policy(obs)
            obs = dep.step(raw_action)
            sim_robot.set_quat(dep.state.base_quat)
            sim_robot.set_dofs_pos(dep.state.dofs_pos)
            sim_robot.set_dofs_vel(dep.state.dofs_vel)
            if show_viewer:
                env.unwrapped.scene.visualizer.update()
    except KeyboardInterrupt:
        en.logger.info("[deploy] Interrupted — closing deployment.")
    finally:
        dep.close()


def _run_train_rsl_rl(config, checkpoint: str | None = None) -> None:
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner

    env = RslRlVecEnvWrapper.from_config(config)
    runner = make_rsl_rl_runner(env, checkpoint=checkpoint)
    runner.learn(num_learning_iterations=config.runner_options.max_iterations, init_at_random_ep_len=True)


def _detect_checkpoint_framework(path: str) -> str | None:
    """Guess which RL framework saved a checkpoint, or ``None`` if unclear.

    rsl_rl saves ``actor_state_dict`` / ``critic_state_dict`` (current
    format) or a single combined ``model_state_dict`` with ``actor.*`` /
    ``critic.*`` keys (legacy format, auto-converted inside
    ``make_rsl_rl_runner``). rl_games saves a ``model`` state_dict
    alongside ``epoch`` / ``optimizer``. Used to fail fast on framework
    mismatches before the runner blows up deep inside ``load`` with an
    opaque KeyError.
    """
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    if "actor_state_dict" in state:
        return "rsl_rl"
    if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        keys = state["model_state_dict"].keys()
        if any(k.startswith("actor.") or k.startswith("critic.") for k in keys):
            return "rsl_rl"
    if "model" in state and ("epoch" in state or "optimizer" in state):
        return "rl_games"
    return None


def _run_eval_rsl_rl(
    config,
    checkpoint: str,
    show_viewer: bool = False,
    record: bool = False,
    max_steps: int | None = None,
    max_episodes: int | None = None,
) -> None:
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner

    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True, show_viewer=show_viewer)
    runner = make_rsl_rl_runner(env, checkpoint=checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    _run_eval_loop(env, policy, record=record, max_steps=max_steps, max_episodes=max_episodes)


def _run_eval_loop(
    env,
    policy_fn,
    *,
    record: bool = False,
    max_steps: int | None = None,
    max_episodes: int | None = None,
) -> None:
    """Inference loop for the rsl_rl path."""
    import math

    import torch
    from tqdm import tqdm

    rec = env.unwrapped.cameras["rec"] if record else None
    if rec is not None:
        rec.start_recording()

    # Per-metric running mean over auto-reset events. metric_manager.reset()
    # publishes "Metrics/<name>" scalars *averaged across the envs that reset
    # on this step* into extras["log"]. To get a correct running mean across
    # episodes (and a correct --episodes cap when num_envs > 1), we weight
    # each step's value by the number of envs that reset.
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    n_resets = 0

    # Step budget: explicit --steps wins; else --episodes * max_episode_length
    # / num_envs as an upper bound (we still break early on episode count);
    # else one episode worth of steps.
    episode_len = max(1, int(env.unwrapped.max_episode_length))
    num_envs = max(1, int(env.num_envs))
    if max_steps is not None:
        step_budget = max_steps
    elif max_episodes is not None:
        step_budget = max(1, math.ceil(max_episodes * episode_len / num_envs))
    else:
        step_budget = episode_len

    # rsl_rl env.reset() returns (obs, extras); rl_games returns just obs.
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    with torch.no_grad():
        for _ in tqdm(range(step_budget), desc="inference"):
            step_out = env.step(policy_fn(obs))
            obs, dones, extras = step_out[0], step_out[2], step_out[-1]
            n_done = int(dones.sum().item())
            log = extras.get("log") if isinstance(extras, dict) else None
            if n_done > 0 and log:
                metric_keys = [k for k in log if isinstance(k, str) and k.startswith("Metrics/")]
                if metric_keys:
                    n_resets += n_done
                    for key in metric_keys:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(log[key]) * n_done
                        metric_counts[key] = metric_counts.get(key, 0) + n_done
                    if max_episodes is not None and n_resets >= max_episodes:
                        break
            if rec is not None:
                rec.render_rgb()

    if rec is not None:
        rec.stop_recording()

    if metric_sums:
        print(f"\n=== Inference metrics (averaged over {n_resets} episode reset(s)) ===")
        for key in sorted(metric_sums):
            avg = metric_sums[key] / max(metric_counts[key], 1)
            print(f"  {key:<32s} {avg: .4f}")


def mp4_to_gif(video_path: str):
    import subprocess

    gif_path = video_path.replace(".mp4", ".gif")
    # Run: `ffmpeg -i $1 -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" "${1%.mp4}.gif"`
    subprocess.run(
        f'ffmpeg -i {video_path} -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {gif_path}',
        shell=True,
    )


def main():
    """Eden CLI entry point."""
    parser = argparse.ArgumentParser(description="Eden - Robot Learning Environments")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List tasks command
    subparsers.add_parser("list", help="List all available tasks")

    subparsers.add_parser(
        "train",
        help="Train a registered task. Use 'eden train --task <name> --help' for task options.",
        add_help=False,
    )
    subparsers.add_parser(
        "inference",
        help="Run inference for a registered task with a trained checkpoint.",
        add_help=False,
    )
    subparsers.add_parser(
        "deploy",
        help="Deploy a trained task policy on real hardware (robot picked from the task's scene).",
        add_help=False,
    )

    parser_gif = subparsers.add_parser("gif", help="Convert mp4 video to gif")
    parser_gif.add_argument("video_path", type=str, help="Path to video file")

    view_parser = subparsers.add_parser(
        "view",
        help="View an entity by name",
    )
    view_parser.add_argument("target", type=str, help="Entity name")
    view_parser.add_argument("-c", "--collision", action="store_true", help="Show collision geometry")

    snap_parser = subparsers.add_parser("snap", help="Take a snapshot of the scene/entity")
    snap_parser.add_argument("entity_name", type=str, help="Name of the entity to snapshot")
    snap_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the snapshot",
        default="snapshot.png",
    )
    snap_parser.add_argument("-c", "--collision", action="store_true", help="Show collision geometry")

    subparsers.add_parser("clean", help="Clean up logs")
    args, extra_args = parser.parse_known_args()

    # 'train', 'inference', and 'deploy' subcommands forward unknown args to
    # their own parsers. For every other command, treat unknown args as a user
    # error (mirrors parse_args behavior) so typos like
    # `eden view robot --collison` aren't silently ignored.
    if extra_args and args.command not in ("train", "inference", "deploy"):
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")

    if args.command == "list":
        print_task_list()
    elif args.command == "train":
        train_task(extra_args)
    elif args.command == "inference":
        inference_task(extra_args)
    elif args.command == "deploy":
        deploy_task(extra_args)
    elif args.command == "gif":
        mp4_to_gif(args.video_path)
    elif args.command == "view":
        show_entity(args.target, args.collision)
    elif args.command == "snap":
        snap_entity(args.entity_name, args.collision, args.output)
    elif args.command == "clean":
        clean_up_logs()
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    """
    Usage:
    eden list
    eden view <entity_name>
    """
    main()
