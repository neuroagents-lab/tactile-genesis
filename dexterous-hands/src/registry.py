import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path as FsPath
from typing import Any

import eden as en
import genesis as gs
import torch
import yaml
from eden.envs.wrappers.rsl_rl_env import DDPDistillationRunner, DDPOnPolicyRunner, RslRlVecEnvWrapper
from eden.options import CameraOptions
from eden.options.camera import CamerasOptions
from eden.options.options import ConfigurableOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.parser import get_task_argparser as get_eden_task_argparser
from eden.tasks.registry import cli_fields
from eden.utils import distributed as eden_dist
from eden.utils.configs import EdenRLConfig
from eden.utils.registry import Registry
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

DEFAULT_TASK = "in_palm_rotate"


TASK_REGISTRY.add_search_path("tasks")
ROBOT_REGISTRY = Registry("ROBOT")
HYPERPARAMS_REGISTRY = Registry("HYPERPARAMS")


_loaded_tasks = False


def _load_tasks():
    global _loaded_tasks
    if not _loaded_tasks:
        import entities.robots  # noqa: F401
        import tasks  # noqa: F401

        _loaded_tasks = True


def _checkpoint_iter(path: FsPath) -> int:
    """Iteration number N parsed from a ``model_<N>.pt`` filename (-1 if absent)."""
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(log_dir: str | FsPath) -> str | None:
    """Return the ``model_<N>.pt`` under ``log_dir`` with the largest N.

    Uses a numeric sort on N so ``model_100.pt`` beats ``model_20.pt`` (a plain
    lexical sort would not).
    """
    candidates = [p for p in FsPath(log_dir).glob("**/model_*.pt") if _checkpoint_iter(p) >= 0]
    if not candidates:
        return None
    return str(max(candidates, key=_checkpoint_iter))


def get_checkpoint_path(checkpoint: str | None, try_load_latest: bool = True) -> str | None:
    if checkpoint:
        checkpoint_path = FsPath(checkpoint).expanduser()
        if checkpoint_path.exists():
            return str(checkpoint_path)
        if checkpoint != "latest":
            raise FileNotFoundError(f"Checkpoint file '{checkpoint}' not found.")
        en.logger.warning(f"Checkpoint file '{checkpoint}' not found, looking for latest checkpoint instead")

    if try_load_latest:
        return find_latest_checkpoint(en.log_dir)

    return None


def build_run_name(
    run_name: str = "",
    task: str = "",
    robot: str = "",
    sensors: str = "none",
    config: str = "",
) -> str:
    """Construct the deterministic run name used for the log directory.

    Mirrors the run-name assembly in ``main.main`` so the training script and
    the distillation status tooling agree on log-dir paths.
    """
    parts: list[str] = []
    if run_name:
        parts.append(run_name)
    if task:
        parts.append(task)
    if robot:
        parts.append(robot)
    parts.append("no_tactile" if sensors in ("", "none") else sensors)
    if config:
        parts.append(FsPath(config).stem)
    return "-".join(parts)


def load_config_section(config_paths: list[str | None], section_name: str) -> dict[str, Any]:
    """Load and merge one top-level section from config files."""
    merged_section = {}
    for config_path in config_paths:
        if not config_path:
            continue

        path_lower = config_path.lower()
        if path_lower.endswith(".json"):
            with open(config_path, "r") as f:
                config_data = json.load(f)
        elif path_lower.endswith((".yaml", ".yml")):
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
        else:
            raise ValueError(f"Unsupported config file extension: {config_path}. Use .json, .yaml, or .yml")

        section = config_data.get(section_name)
        if isinstance(section, dict):
            merged_section.update(section)

    return merged_section


def build_modifiers_from_args(args: Any, task_name: str) -> dict[str, Any]:
    if not task_name or task_name not in TASK_REGISTRY:
        return {}
    out: dict[str, Any] = {}
    for mod in TASK_REGISTRY.get_modifiers(task_name):
        for field_name, _annotation, _default, _help in cli_fields(type(mod)):
            flag = mod.flag_for(field_name)
            if hasattr(args, flag):
                out[flag] = getattr(args, flag)
    return out


def get_task_config(
    run_name: str = "",
    task_name: str = "",
    modifiers: dict[str, Any] | None = None,
    config_override_path: str = "",
) -> ConfigurableOptions:
    config = TASK_REGISTRY.build(task_name, **(modifiers or {}))

    if config_override_path:
        config = config.with_overrides_from_file(config_override_path)

    if config.runner_options is not None:
        class_name = config.__class__.__name__
        if config.runner_options.experiment_name == class_name:
            config.runner_options.experiment_name = task_name
        if config.runner_options.wandb_project == "eden-" + class_name:
            config.runner_options.wandb_project = "eden-" + task_name
        config.runner_options.run_name = re.sub(r"[/\\#?%:]", "-", run_name)
        config.runner_options.enable_print = False

    return config


def get_task_config_from_args(args: argparse.Namespace, run_name: str = "", upload_logs: bool = True):
    _load_tasks()

    if args.full_config:
        config = EdenRLConfig.load_from_file(args.full_config)
        if args.config:
            config = config.with_overrides_from_file(args.config)
    else:
        mod_dict = build_modifiers_from_args(args, args.task)
        config = get_task_config(
            run_name=run_name,
            task_name=args.task,
            modifiers=mod_dict,
            config_override_path=args.config,
        )
    runner_options = getattr(config, "runner_options", None)
    if runner_options is not None and eden_dist.is_distributed():
        config.runner_options.device = "cuda:0"
    if runner_options is not None and args.cpu:
        config.runner_options.device = "cpu"

    if runner_options is not None and not upload_logs:
        # Don't log to wandb. Setting to None will cause error with rsl_rl runner.
        config.runner_options.logger = "tensorboard"

    return config


def get_argparser(description: str):
    _load_tasks()
    parser = get_eden_task_argparser(description=description, default_task_name=DEFAULT_TASK)
    parser.add_argument("--actor_checkpoint", type=str, default=None, help="Path to the actor checkpoint file.")
    parser.add_argument("--critic_checkpoint", type=str, default=None, help="Path to the critic checkpoint file.")
    return parser


def position_camera_config(config: EdenRLConfig) -> EdenRLConfig:
    # Respect an explicitly configured recording camera from the task/defaults YAML.
    if config.cameras_options is None:
        config.cameras_options = CamerasOptions()
    if getattr(config.cameras_options, "rec", None) is not None:
        return config

    n_per_row = config.env_options.num_eval_envs ** (0.5)
    corner_pos = tuple(s * n_per_row for s in config.env_options.env_spacing)
    config.cameras_options._locked = False
    config.cameras_options.rec = CameraOptions(
        cam_pos=(corner_pos[0] * 0.9, corner_pos[1] * 0.8, 0.8),
        cam_lookat=(corner_pos[0] * 0.5, corner_pos[1] * 0.5, 0.1),
        fov=40.0,
    )
    return config


def save_video_from_checkpoint(
    config: EdenRLConfig,
    checkpoint: str | None,
    seconds: float = 10.0,
    video_path: str | None = None,
    tactile_npz_path: str | None = None,
) -> None:
    print(f"Saving video from checkpoint: {checkpoint} ...")
    # Set camera position for recording
    config = position_camera_config(config)

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)
    runner = make_runner(env, checkpoint)

    policy = runner.get_inference_policy(device=gs.device)
    obs, *_ = env.reset()
    rec = env.unwrapped.cameras["rec"]
    rec.start_recording()

    # Optionally record per-step tactile sensor readings for offline animation.
    recorder = None
    if tactile_npz_path:
        from tactile_record import TactileEpisodeRecorder

        task_name = getattr(getattr(config, "runner_options", None), "experiment_name", "") or ""
        recorder = TactileEpisodeRecorder(env.unwrapped, task=task_name)
        if recorder.sensor_names:
            print(f"Recording tactile sensors: {', '.join(recorder.sensor_names)}")
        else:
            print("[tactile] no tactile_* sensors on env; skipping --save_tactile npz.")
            recorder = None

    steps = int(seconds / env.dt)
    print(f"Running policy for {seconds} seconds ({steps} steps)...")
    t = 0.0
    with torch.no_grad():
        for _ in range(steps):
            actions = policy(obs)
            obs, *_ = env.step(actions)
            rec.render_rgb()
            if recorder is not None:
                t += env.dt
                recorder.record(t)

    video_path = video_path or (checkpoint.replace(".pt", ".mp4") if checkpoint else "inference.mp4")
    rec.stop_recording(video_path, fps=1.0 / env.dt)

    if recorder is not None:
        saved = recorder.save(tactile_npz_path)
        print(f"Tactile readings saved to {saved}")

    if hasattr(env, "close"):
        env.close()


def launch_video_from_checkpoint(config_path: str | FsPath, checkpoint: str, seconds: float = 10.0) -> None:
    """Launch checkpoint video recording in a subprocess so training can continue."""
    config_path = FsPath(config_path)
    checkpoint_path = FsPath(checkpoint)
    video_path = str(checkpoint_path.with_suffix(".mp4"))
    log_path = str(checkpoint_path.with_suffix(".video.log"))
    if not config_path.exists():
        en.logger.warning(
            f"Skipping async checkpoint video: expected config file next to checkpoint at '{config_path}'"
        )
        return

    script_path = FsPath(sys.argv[0])
    if not script_path.exists():
        script_path = FsPath.cwd() / "main.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--mode=play",
        f"--full_config={config_path}",
        f"--checkpoint={checkpoint}",
        f"--video_output={video_path}",
        f"--record_seconds={seconds}",
        "--cpu",
    ]

    log_file = open(log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=FsPath.cwd(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    print(f"Launched video worker pid={process.pid}: {video_path} (log: {log_path})")


def save_video_from_runner(
    config: EdenRLConfig,
    runner: OnPolicyRunner | DistillationRunner,
    video_path: str,
    seconds: float = 10.0,
) -> None:
    """Save a video from an already-loaded runner without reloading checkpoint files."""
    print(f"Saving video from loaded runner: {video_path} ...")
    config._locked = False
    config = position_camera_config(copy.deepcopy(config))

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)
    if hasattr(runner.alg, "actor"):
        runner.alg.actor.reset()
    if hasattr(runner.alg, "student"):
        runner.alg.student.reset()
    policy = runner.get_inference_policy(device=gs.device)
    obs, *_ = env.reset()
    rec = env.unwrapped.cameras["rec"]
    rec.start_recording()

    steps = int(seconds / env.dt)
    print(f"Running loaded policy for {seconds} seconds ({steps} steps)...")
    with torch.no_grad():
        for _ in range(steps):
            actions = policy(obs)
            obs, *_ = env.step(actions)
            rec.render_rgb()

    rec.stop_recording(video_path, fps=1.0 / env.dt)
    if hasattr(runner.alg, "actor"):
        runner.alg.actor.reset()
    if hasattr(runner.alg, "student"):
        runner.alg.student.reset()

    if hasattr(env, "close"):
        env.close()


def patch_on_policy_runner_to_save_video():
    # monkey patch OnPolicyRunner to save a video of the policy when saving a checkpoint.
    old_save = OnPolicyRunner.save

    def new_save(self, path: str, infos: dict | None = None) -> None:
        old_save(self, path, infos)
        if "model_0" in path:
            return  # Skip saving video for initial checkpoint.
        try:
            parent = FsPath(path).resolve().parent
            eden_cfg = parent / "eden_config.yaml"
            if eden_cfg.is_file():
                config_path = eden_cfg
            else:
                config_path = parent / "config.yaml"
            launch_video_from_checkpoint(config_path, path, seconds=10)
        except Exception as e:
            print(f"Error saving video: {e}")
            import traceback

            traceback.print_exc()

    OnPolicyRunner.save = new_save


def make_runner(
    env: RslRlVecEnvWrapper,
    checkpoint: str | None = None,
    actor_checkpoint: str | None = None,
    critic_checkpoint: str | None = None,
    log_dir: str | None = None,
    load_actor_distribution: bool = True,
):
    """
    Create an OnPolicyRunner and optionally load a checkpoint.
    """
    log_dir = log_dir if log_dir is not None else en.log_dir

    if "distill" in env.runner_dict["class_name"].lower():
        runner_cls = DDPDistillationRunner if eden_dist.is_distributed() else DistillationRunner
    else:
        runner_cls = DDPOnPolicyRunner if eden_dist.is_distributed() else OnPolicyRunner
    runner = runner_cls(env, env.runner_dict, log_dir=log_dir, device=env.runner_dict["device"])

    if checkpoint is None:
        checkpoint_path = None
    else:
        checkpoint_path = get_checkpoint_path(checkpoint, try_load_latest=checkpoint == "latest")

    load_kwargs = {"map_location": gs.device, "strict": False}
    loaded_checkpoint = False
    if checkpoint_path:
        runner.load(checkpoint_path, **load_kwargs)
        print("Loaded checkpoint from:", checkpoint_path, load_kwargs)
        loaded_checkpoint = True
    if actor_checkpoint:
        actor_load_kwargs = {**load_kwargs, "load_cfg": {"actor": True}}
        if load_actor_distribution:
            runner.load(actor_checkpoint, **actor_load_kwargs)
        else:
            loaded_dict = torch.load(actor_checkpoint, weights_only=False, map_location=load_kwargs["map_location"])
            actor_key = "actor_state_dict" if "actor_state_dict" in loaded_dict else "student_state_dict"
            actor_state_dict = loaded_dict[actor_key]
            removed_keys = []
            for key in ("distribution.log_std_param", "distribution.std_param"):
                if key in actor_state_dict:
                    actor_state_dict.pop(key)
                    removed_keys.append(key)
            runner.alg.load(loaded_dict, actor_load_kwargs["load_cfg"], actor_load_kwargs["strict"])
            if removed_keys:
                print("Skipped actor distribution keys from checkpoint:", removed_keys)
            print(
                "Loaded actor checkpoint from:",
                actor_checkpoint,
                {**actor_load_kwargs, "load_actor_distribution": load_actor_distribution},
            )
        loaded_checkpoint = True
    if critic_checkpoint:
        critic_load_kwargs = {**load_kwargs, "load_cfg": {"critic": True}}
        runner.load(critic_checkpoint, **critic_load_kwargs)
        print("Loaded critic checkpoint from:", critic_checkpoint, critic_load_kwargs)
        loaded_checkpoint = True

    if loaded_checkpoint:
        if hasattr(runner.alg, "actor"):
            runner.alg.actor.reset()
        if hasattr(runner.alg, "student"):
            runner.alg.student.reset()

    return runner


def get_hyperparam_ranges(task_name: str) -> dict[str, Any]:
    if not _loaded_tasks:
        _load_tasks()
    if task_name not in HYPERPARAMS_REGISTRY:
        raise ValueError(f"No hyperparameter ranges found for task: {task_name}")

    return HYPERPARAMS_REGISTRY.get(task_name)
