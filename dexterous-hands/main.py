"""Main training script for dexterous manipulation tasks using Eden + RSL-RL."""

import argparse
import os
import shutil
import sys

import eden as en
import genesis as gs
import torch
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from tensordict import TensorDict

from optimization import apply_params_to_config, run_evaluation, sample_hyperparams
from registry import (
    build_run_name,
    find_latest_checkpoint,
    get_argparser,
    get_hyperparam_ranges,
    get_task_config_from_args,
    make_runner,
    patch_on_policy_runner_to_save_video,
    save_video_from_checkpoint,
    save_video_from_runner,
)

# Exit code emitted when --resume hit a state_dict size mismatch and main.py has
# wiped the log dir. babel/run_distill.sh watches for this and re-execs the same
# command once; the retry finds no checkpoint and starts fresh from the teacher.
STALE_CHECKPOINT_EXIT_CODE = 77


def _wipe_stale_log_dir(log_dir: str) -> None:
    """Remove a log dir whose latest checkpoint no longer matches the live model."""
    if not log_dir or not os.path.isdir(log_dir):
        return
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.finish(exit_code=1, quiet=True)
    except Exception:
        pass
    shutil.rmtree(log_dir)


def train(config, args):
    """Train the policy."""
    if getattr(args, "alg", "") == "distill":
        assert args.checkpoint is not None, "Teacher checkpoint is required for distillation."

    if not args.no_checkpoint_video:
        patch_on_policy_runner_to_save_video()

    # Resume support: with --resume, load the latest checkpoint already present
    # in the (fixed) log dir instead of the teacher. A distillation model_*.pt
    # embeds the teacher and restores the optimizer + iteration counter, so
    # `make_runner` picks up exactly where the preempted job left off.
    checkpoint_arg = args.checkpoint
    resume_checkpoint = None
    if getattr(args, "resume", False):
        resume_checkpoint = find_latest_checkpoint(en.log_dir)
        if resume_checkpoint:
            checkpoint_arg = resume_checkpoint
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("--resume set but no prior checkpoint found in log dir; starting fresh.")

    config.env_options.background_color = (1.0, 1.0, 1.0)
    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.show_viewer)
    # Eden replay config (``--full_config``); keep separate from W&B ``config.yaml`` in the log dir.
    config.save_as_yaml(path=f"{en.log_dir}/eden_config.yaml", lock_config=False)
    try:
        runner = make_runner(
            env,
            checkpoint_arg,
            args.actor_checkpoint,
            args.critic_checkpoint,
            load_actor_distribution=getattr(args, "stage", None) != 3,
        )
    except RuntimeError as exc:
        # A state_dict size mismatch under --resume means the saved checkpoint's
        # architecture has drifted from what the live config builds. We can't
        # patch the weights across that drift, so wipe the log dir and let the
        # wrapper re-exec for a fresh start. Only auto-discovered resume
        # checkpoints are eligible; explicit --actor_checkpoint / --critic
        # / --checkpoint mismatches re-raise as before.
        if resume_checkpoint is not None and "size mismatch" in str(exc):
            print(f"[STALE_CHECKPOINT] Shape mismatch loading resume checkpoint:\n  {resume_checkpoint}")
            print(f"[STALE_CHECKPOINT] {exc}")
            print(f"[STALE_CHECKPOINT] Wiping log dir: {en.log_dir}")
            _wipe_stale_log_dir(en.log_dir)
            print(f"[STALE_CHECKPOINT] Exiting {STALE_CHECKPOINT_EXIT_CODE} for wrapper to re-exec fresh.")
            sys.exit(STALE_CHECKPOINT_EXIT_CODE)
        raise
    if (checkpoint_arg or args.actor_checkpoint or args.critic_checkpoint) and not args.no_loaded_video:
        save_video_from_runner(config, runner, f"{en.log_dir}/loaded_policy.mp4", seconds=args.record_seconds)
    runner.add_git_repo_to_log(__file__)

    # `learn` runs `current_learning_iteration + num_learning_iterations` total,
    # so on resume we ask only for the remaining iterations.
    max_iterations = config.runner_options.max_iterations
    remaining = max(0, max_iterations - runner.current_learning_iteration)
    if remaining == 0:
        print(f"\nAlready at {runner.current_learning_iteration}/{max_iterations} iterations; skipping training.")
    else:
        print(
            f"\nStarting training: {remaining} iterations ({runner.current_learning_iteration} -> {max_iterations})..."
        )
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)

    print("\nTraining complete!")
    print(f"Logs saved to: {en.log_dir}")

    if args.no_checkpoint_video:
        return
    args.checkpoint = f"{en.log_dir}/model_{max_iterations - 1}.pt"
    play_config = get_task_config_from_args(args, upload_logs=False)
    play(play_config, args)


def play(config, args):
    """Run inference with trained policy."""
    tactile_npz_path = None
    if getattr(args, "save_tactile", False):
        tactile_npz_path = args.tactile_output or os.path.join(en.log_dir, "tactile_readings.npz")
    save_video_from_checkpoint(
        config,
        args.checkpoint,
        seconds=args.record_seconds,
        video_path=args.video_output,
        tactile_npz_path=tactile_npz_path,
    )


def rollout_benchmark(config, args):
    """Run one episode of the policy at several num_envs values and report per-step timing."""
    import copy
    import time

    import numpy as np

    num_envs_list = args.benchmark_num_envs
    report_fastest_k = 100

    print(f"\nRollout benchmark: num_envs sweep = {num_envs_list}")
    print(f"Final stats computed on the fastest {report_fastest_k} steps of each rollout.")
    results: list[dict] = []
    all_times_path = os.path.join(en.log_dir, "rollout_benchmark_times.npz")
    raw_times: dict[str, np.ndarray] = {}

    for n_envs in num_envs_list:
        print(f"\n=== num_envs = {n_envs} ===")
        cfg = copy.deepcopy(config)
        cfg.env_options._locked = False
        cfg.env_options.num_eval_envs = n_envs

        env = RslRlVecEnvWrapper.from_config(cfg, show_viewer=False, eval_mode=True)
        runner = make_runner(env, args.checkpoint, args.actor_checkpoint, args.critic_checkpoint)
        policy = runner.get_inference_policy(device=gs.device)

        obs, *_ = env.reset()
        steps = max(1, int(env.unwrapped.max_episode_length))
        print(f"Running {steps} steps with {n_envs} envs...")

        step_times_ms: list[float] = []
        with torch.no_grad():
            wall_t0 = time.perf_counter()
            for _ in range(steps):
                t0 = time.perf_counter()
                actions = policy(obs)
                obs, *_ = env.step(actions)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)
            wall_ms = (time.perf_counter() - wall_t0) * 1000.0

        arr = np.asarray(step_times_ms, dtype=np.float64)
        raw_times[f"n{n_envs}"] = arr
        k = min(report_fastest_k, arr.size)
        fastest = np.sort(arr)[:k]
        mean_ms = float(fastest.mean())
        std_ms = float(fastest.std())
        median_ms = float(np.median(fastest))
        p95_ms = float(np.percentile(fastest, 95))
        min_ms = float(fastest.min())
        max_ms = float(fastest.max())
        full_max_ms = float(arr.max())
        full_argmax_step = int(arr.argmax())
        wall_per_step_ms = wall_ms / steps
        print(
            f"  fastest-{k}: mean = {mean_ms:.3f}  std = {std_ms:.3f}  median = {median_ms:.3f}  "
            f"p95 = {p95_ms:.3f}  min = {min_ms:.3f}  max = {max_ms:.3f}"
        )
        print(
            f"  (full {steps} steps: max = {full_max_ms:.3f} at step {full_argmax_step}, "
            f"wall {wall_ms:.1f} ms)"
        )
        results.append(
            dict(
                n_envs=n_envs,
                mean=mean_ms,
                std=std_ms,
                median=median_ms,
                p95=p95_ms,
                min=min_ms,
                max=max_ms,
                full_max=full_max_ms,
                wall_per_step=wall_per_step_ms,
                steps=steps,
                k=k,
            )
        )

        if hasattr(env, "close"):
            env.close()

    np.savez(all_times_path, **raw_times)
    print(f"\nRaw per-step times saved to {all_times_path}")

    print(f"\nBenchmark summary (stats over fastest {report_fastest_k} steps per rollout):")
    header = f"{'num_envs':>10}  {'mean':>10}  {'std':>10}  {'median':>10}  {'p95':>10}  {'max':>10}  {'full_max':>10}"
    print(header)
    for r in results:
        print(
            f"{r['n_envs']:>10}  {r['mean']:>10.3f}  {r['std']:>10.3f}  {r['median']:>10.3f}  "
            f"{r['p95']:>10.3f}  {r['max']:>10.3f}  {r['full_max']:>10.3f}"
        )


def optimize(config, args):
    """Run Optuna hyperparameter optimization over reward weights and PPO policy/algorithm."""
    import optuna

    hyperparam_ranges = get_hyperparam_ranges(config.runner_options.experiment_name)

    MIN_STARTUP_TRIALS = 10
    n_trials = args.opt_trials
    n_iterations = args.opt_iterations
    n_eval_episodes = args.opt_eval_episodes

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=min(MIN_STARTUP_TRIALS, n_trials),
            seed=config.runner_options.seed,
        ),
        pruner=optuna.pruners.NopPruner(),
        storage=args.opt_storage,
        load_if_exists=bool(args.opt_storage),
        direction="maximize",
    )

    config.runner_options.max_iterations = n_iterations

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparams(trial, hyperparam_ranges=hyperparam_ranges)
        trial_config = apply_params_to_config(params, config)

        # Train
        env = RslRlVecEnvWrapper.from_config(trial_config, show_viewer=False)

        trial_log_dir = f"{en.log_dir}/trial_{trial.number}"
        runner = make_runner(
            env,
            args.checkpoint,
            args.actor_checkpoint,
            args.critic_checkpoint,
            log_dir=trial_log_dir,
            load_actor_distribution=getattr(args, "stage", None) != 3,
        )
        runner.learn(num_learning_iterations=n_iterations, init_at_random_ep_len=True)

        # Evaluate
        trial_config.env_options._locked = False
        trial_config.env_options.num_eval_envs = n_eval_episodes
        return run_evaluation(trial_config, runner, n_eval_episodes, gs.device, args.opt_save_video)

    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials:", len(study.trials))
    if study.best_trial is None:
        print("No completed trials.")
        return None

    print("Best trial:")
    print("  Value (mean return):", study.best_trial.value)
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    return study


def deploy(config, args):
    if args.robot != "xhand1":
        raise ValueError("Deploy mode is only supported for XHand1 robot.")

    sensor_type = args.sensors.split("/")[-1]
    if sensor_type not in {"", "none", "bool", "agg_bool", "agg_force"}:
        raise ValueError("Deploy mode requires --sensors to be 'none', 'bool', 'agg_bool', or 'agg_force'.")

    from eden.envs.base import RLEnvBase

    from deploy import RoboTeraXHandTactileDeployment

    config.env_options.num_eval_envs = 1
    config.env_options.background_color = (1.0, 1.0, 1.0)
    config.reward_options = None
    config.metric_options = None
    config.termination_options = None

    # Build the env in two stages (mirrors RslRlVecEnvWrapper.from_config) so the
    # tactile recorder can call scene.start_recording() before the scene is built
    # (start_recording is @assert_unbuilt).
    base_env = RLEnvBase.from_config(config, show_viewer=args.show_viewer, eval_mode=True)

    recorder = None
    sim_sensor_names: dict[int, str] = {}
    if args.record_tactile:
        from tactile_compare import (
            FINGER_NAMES,
            TACTILE_PLOT_CHANNELS,
            TactileComparisonRecorder,
            finger_indices,
            fingertip_link_order,
            read_real_tactile,
            read_sim_tactile,
            resolve_sim_sensor_names,
            start_tactile_plot,
        )

        plot_channels = TACTILE_PLOT_CHANNELS.get(sensor_type)
        if plot_channels is None:
            raise ValueError(
                f"--record_tactile not supported for sensor type {sensor_type!r}; "
                f"known types: {sorted(TACTILE_PLOT_CHANNELS)}."
            )
        sim_sensor_names = resolve_sim_sensor_names(
            base_env,
            fingertip_links=fingertip_link_order(config.scene_options.robot),
            sensor_type=sensor_type,
        )
        if not sim_sensor_names:
            tactile_keys = sorted(k for k in getattr(base_env, "sensors", {}) if k.startswith("tactile_"))
            print(
                f"[tactile] no {sensor_type!r} fingertip sim sensors found; recording real hand only "
                f"(sim columns -> nan). tactile_* sensors in env: {tactile_keys or 'none'}"
            )
        if args.tactile_fingers:
            display_fingers = finger_indices(args.tactile_fingers)
        else:
            # The real hand reports all five fingers; sim adds whichever links are sensored.
            display_fingers = sorted(set(range(len(FINGER_NAMES))) | set(sim_sensor_names))
        recorder = TactileComparisonRecorder(
            en.log_dir, display_fingers=display_fingers, channels=plot_channels
        )
        start_tactile_plot(
            base_env.scene,
            recorder,
            title=f"xhand1 deploy: real vs sim tactile ({sensor_type})",
            history_length=args.tactile_history,
        )

    base_env.build()
    env = RslRlVecEnvWrapper(base_env, options=config.runner_options)
    runner = make_runner(env, args.checkpoint, args.actor_checkpoint, args.critic_checkpoint)
    policy = runner.get_inference_policy(device=gs.device)

    if hasattr(runner.alg, "actor"):
        runner.alg.actor.reset()
    if hasattr(runner.alg, "student"):
        runner.alg.student.reset()

    deployment = RoboTeraXHandTactileDeployment(
        env.unwrapped,
        RoboTeraXHandTactileDeployment.configure(
            entity_name="robot",
            tactile_sensor_type=sensor_type,
            tactile_bool_threshold=args.deploy_bool_threshold,
        ),
    )

    obs = None
    try:
        deployment.connect()
        sim_obs, _ = env.reset()
        deploy_obs_dict = deployment.reset()
        deploy_obs = TensorDict(deploy_obs_dict, batch_size=[env.num_envs])
        print("SIM OBS", sim_obs)
        print("DEPLOYMENT OBS", deploy_obs)
        obs = sim_obs if args.deploy_sim_obs else deploy_obs
        num_steps = int(args.record_seconds / env.dt)
        print(f"Deploying for {args.record_seconds} seconds ({num_steps} steps)")

        with torch.no_grad():
            for i in range(num_steps):
                print(f"--- STEP {i} ---")
                if args.stochastic:
                    action = policy(obs, stochastic_output=True)
                else:
                    action = policy(obs)
                # print("ACTION", action)
                sim_obs, *_ = env.step(action)
                deploy_obs_dict = deployment.step(action)
                deploy_obs = TensorDict(deploy_obs_dict, batch_size=[env.num_envs])
                print(f"SIM PROPRIO\n{sim_obs['proprio']}\nDEPLOYMENT PROPRIO\n{deploy_obs['proprio']}")
                obs = sim_obs if args.deploy_sim_obs else deploy_obs

                if recorder is not None:
                    try:
                        sim_tac = read_sim_tactile(
                            base_env, sim_sensor_names, device=gs.device, sensor_type=sensor_type
                        )
                        real_tac = read_real_tactile(
                            deployment.state,
                            sensor_type=sensor_type,
                            threshold=args.deploy_bool_threshold,
                        )
                        recorder.record(t=i * env.dt, sim=sim_tac, real=real_tac)
                    except Exception as exc:  # noqa: BLE001 - keep the deploy loop alive.
                        print(f"[tactile] recording disabled after error: {type(exc).__name__}: {exc}")
                        recorder.close()
                        recorder = None
    finally:
        if recorder is not None:
            recorder.close()
            print(f"Tactile CSV saved to {recorder.csv_path}")
        deployment.close()
        env.close()


def main():
    parser = get_argparser(description="Train or evaluate RL agents for dexterous manipulation tasks.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode to run: train, play/inference, optimize, deploy, or rollout-benchmark",
    )
    parser.add_argument("--show_viewer", action="store_true", help="Show the viewer during run")
    # Training (mode=train)
    parser.add_argument("--load_best_hyperparams", action="store_true", help="Load best hyperparameters from study")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in the (fixed) log dir, if one exists.",
    )
    parser.add_argument(
        "--no_loaded_video",
        action="store_true",
        help="Skip rendering loaded_policy.mp4 of the loaded checkpoint at the start of training. "
        "Useful for distillation sweeps where the teacher checkpoint is always loaded and the "
        "render adds latency to every job.",
    )
    parser.add_argument(
        "--no_checkpoint_video",
        action="store_true",
        help="Skip per-checkpoint async video saves (model_*.mp4 / model_*.video.log) and the "
        "final play() video at the end of training. Used by distillation sweeps to avoid the "
        "genesis probe-debug IndexError that fires when tactile sensors have history_length > 0.",
    )
    # Inference (mode=play)
    parser.add_argument("--record_seconds", "-t", type=float, default=10.0, help="Seconds to run inference for")
    parser.add_argument(
        "--save_tactile",
        action="store_true",
        help="Save per-step tactile sensor readings of the play episode to an .npz.",
    )
    parser.add_argument("--tactile_output", type=str, default=None, help=argparse.SUPPRESS)
    # Deploy (mode=deploy)
    parser.add_argument(
        "--deploy_bool_threshold",
        type=float,
        default=100.0,
        help="Magnitude threshold for deploy bool tactile mode (||calc_pressure|| > threshold).",
    )
    parser.add_argument(
        "--deploy_sim_obs",
        action="store_true",
        help="Use sim observations during deploy while still stepping and printing deployment observations.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions during deployment.")
    parser.add_argument(
        "--record_tactile",
        action="store_true",
        help="Record + live-plot real-vs-sim fingertip tactile during deploy; saves a CSV to the log dir.",
    )
    parser.add_argument("--tactile_history", type=int, default=500, help="Live tactile plot history length.")
    parser.add_argument(
        "--tactile_fingers",
        nargs="+",
        choices=["thumb", "index", "mid", "ring", "pinky"],
        default=None,
        help="Which fingertips to plot (default: all fingers with data). CSV still records all.",
    )
    # Hyperparameter optimization (mode=optimize)
    parser.add_argument("--study_name", type=str, default="", help="The study name")
    parser.add_argument("--opt_save_video", action="store_true", help="Save videos of evaluations")
    parser.add_argument("--opt_trials", type=int, default=100, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--opt_iterations", type=int, default=1000, help="Training iterations per trial")
    parser.add_argument("--opt_eval_episodes", type=int, default=10, help="Evaluation episodes per trial")
    parser.add_argument("--opt_storage", type=str, default=None, help="Database URL for Optuna study")
    parser.add_argument("--video_output", type=str, default=None, help=argparse.SUPPRESS)
    # Rollout benchmark (mode=rollout-benchmark)
    parser.add_argument(
        "--benchmark_num_envs",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="num_envs values to sweep over for rollout-benchmark mode.",
    )
    args = parser.parse_args()

    run_name = build_run_name(
        run_name=args.run_name,
        task=args.task,
        robot=getattr(args, "robot", ""),
        sensors=getattr(args, "sensors", "none"),
        config=args.config,
    )

    # Initialize Eden
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{run_name}/{args.mode}",
        performance_mode=args.mode not in {"play", "deploy"},  # Disable performance mode for viewer/video modes
    )
    # Training uses a fixed (timestamp-free) log dir so a preempted job that is
    # resubmitted with --resume writes to the same place and finds its
    # checkpoints. Other modes keep Eden's timestamped dirs.
    if args.mode == "train":
        eden_timestamped_dir = en.log_dir
        en.log_dir = os.path.abspath(f"logs/{run_name}/train")
        os.makedirs(en.log_dir, exist_ok=True)
        # en.init() already created an empty timestamped dir; drop it.
        if os.path.isdir(eden_timestamped_dir) and not os.listdir(eden_timestamped_dir):
            os.rmdir(eden_timestamped_dir)
    print("Logging to", en.log_dir)

    config = get_task_config_from_args(args, run_name=run_name, upload_logs=args.mode == "train")

    if args.load_best_hyperparams:
        from optimization import apply_params_to_config, optuna

        study = optuna.load_study(study_name=args.study_name, storage=args.opt_storage)
        config = apply_params_to_config(study.best_params, config)

    if args.mode == "train":
        train(config, args)
    elif args.mode in ["play", "inference"]:
        play(config, args)
    elif args.mode in ["optimize", "opt"]:
        optimize(config, args)
    elif args.mode == "deploy":
        deploy(config, args)
    elif args.mode == "rollout-benchmark":
        rollout_benchmark(config, args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
