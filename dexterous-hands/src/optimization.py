import json
from typing import TYPE_CHECKING, Any

import genesis as gs
import optuna
import torch
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

from registry import position_camera_config

if TYPE_CHECKING:
    from eden.utils.configs import EdenRLConfig

# list indicates categorical, tuple indicates range
DEFAULT_HYPERPARAM_RANGES = {
    "learning_rate": (1e-7, 1e-3),
    "clip_param": (0.1, 0.2),
    "entropy_coef": (1e-5, 0.01),
    "gamma": [0.99, 0.995, 0.998, 0.999],
    "lam": (0.9, 1.0),
    "num_learning_epochs": (3, 10),
    "num_mini_batches": [2, 4, 8, 16],
    "desired_kl": (0.005, 0.02),
    "max_grad_norm": (0.5, 2.0),
    "value_loss_coef": (0.5, 2.0),
    "init_std": (0.3, 1.5),
    # These parameters are not searched, use the default values from the config.
    # "activation": ["elu", "relu"],
    # "actor_hidden_dims": [
    #     "[512, 32]",
    #     "[256, 128, 32]",
    #     "[512, 256, 128]",
    #     "[512, 256, 128, 16]",
    # ],
    # "critic_hidden_dims": [
    #     "[512, 32]",
    #     "[256, 128, 32]",
    #     "[512, 256, 128]",
    #     "[512, 256, 128, 16]",
    # ],
}

RECURRENT_HYPERPARAM_RANGES = {
    "rnn_type": ["gru", "lstm"],
    "rnn_hidden_dim": (128, 512),
    "rnn_num_layers": (1, 3),
}

_PARAM_TO_CONFIG_KEY_MAP = {
    "learning_rate": "algorithm",
    "clip_param": "algorithm",
    "entropy_coef": "algorithm",
    "gamma": "algorithm",
    "lam": "algorithm",
    "num_learning_epochs": "algorithm",
    "num_mini_batches": "algorithm",
    "desired_kl": "algorithm",
    "max_grad_norm": "algorithm",
    "value_loss_coef": "algorithm",
    "gradient_length": "algorithm",
    "schedule": "algorithm",
    "init_std": "actor_distribution",
    "init_noise_std": "actor_distribution",  # Backward compatibility with old study names.
    "activation": "actor_critic",
    "actor_hidden_dims": "actor",
    "critic_hidden_dims": "critic",
    "rnn_type": "actor",
    "rnn_hidden_dim": "actor",
    "rnn_num_layers": "actor",
}


def apply_params_to_config(params: dict[str, Any], config: "EdenRLConfig") -> "EdenRLConfig":
    """
    Apply Optuna params to config by copying and updating only rewards and runner.

    Hyperparameters starting with "rew_" are considered reward weights.
    Hyperparameters starting with "aux_" set the ``weight`` of the auxiliary
    distillation loss whose ``name`` matches the suffix (e.g. ``aux_obj_tilt``
    sets the weight of the auxiliary loss named ``obj_tilt``).
    """
    # Update reward weights by copying each term, only changing weight where we have a param
    reward_dump = config.reward_options.model_dump()
    reward_keys = [k for k in reward_dump if k not in ("_option_module_", "_option_class_")]
    reward_updates = {}
    for name in reward_keys:
        term = getattr(config.reward_options, name)
        rew_key = f"rew_{name}"
        if rew_key in params:
            reward_updates[name] = term.model_copy(update={"weight": params[rew_key]})
        else:
            reward_updates[name] = term
    new_rewards = type(config.reward_options)(**reward_updates)

    # Update runner algorithm and actor/critic from params
    algorithm_updates = {k: params[k] for k in params if _PARAM_TO_CONFIG_KEY_MAP.get(k) == "algorithm"}

    # Auxiliary distillation loss weights: params "aux_<name>" set the weight of
    # the auxiliary loss whose "name" is <name>. Only applies when the runner's
    # algorithm carries auxiliary losses (i.e. the distillation stage).
    aux_overrides = {k[len("aux_") :]: v for k, v in params.items() if k.startswith("aux_")}
    existing_aux = getattr(config.runner_options.algorithm, "auxiliary_losses", None)
    if aux_overrides and existing_aux:
        updated_aux = []
        for spec in existing_aux:
            spec = dict(spec)
            aux_name = spec.get("name") or spec.get("target_obs")
            if aux_name in aux_overrides:
                spec["weight"] = aux_overrides[aux_name]
            updated_aux.append(spec)
        algorithm_updates["auxiliary_losses"] = updated_aux
    actor_updates = {}
    critic_updates = {}
    actor_distribution_updates = {}
    for k in params:
        target = _PARAM_TO_CONFIG_KEY_MAP.get(k)
        if target not in {"actor", "critic", "actor_critic", "actor_distribution"}:
            continue
        v = params[k]
        if "hidden_dims" in k:
            v = json.loads(v)

        if target == "actor":
            actor_updates[k] = v
        elif target == "critic":
            critic_updates[k] = v
        elif target == "actor_critic":
            actor_updates[k] = v
            critic_updates[k] = v
        elif target == "actor_distribution":
            actor_distribution_updates["init_std"] = v

    # The on-policy runner exposes "actor"/"critic"; the distillation runner
    # exposes "student"/"teacher". Resolve the field names so actor/critic params
    # (and the auxiliary loss params, which only exist on the distill runner)
    # apply to whichever runner the config was built with.
    runner = config.runner_options
    actor_field = "actor" if hasattr(runner, "actor") else "student"
    critic_field = "critic" if hasattr(runner, "critic") else "teacher"

    if actor_distribution_updates:
        actor_model = getattr(runner, actor_field)
        if hasattr(actor_model, "distribution_cfg"):
            actor_updates["distribution_cfg"] = actor_model.distribution_cfg.model_copy(
                update=actor_distribution_updates
            )

    runner_update = {"algorithm": runner.algorithm.model_copy(update=algorithm_updates)}
    if actor_updates:
        runner_update[actor_field] = getattr(runner, actor_field).model_copy(update=actor_updates)
    if critic_updates:
        runner_update[critic_field] = getattr(runner, critic_field).model_copy(update=critic_updates)
    new_runner = runner.model_copy(update=runner_update)

    return config.model_copy(update={"reward_options": new_rewards, "runner_options": new_runner})


def sample_hyperparams(
    trial: optuna.Trial,
    hyperparam_ranges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Sample reward weights and PPO actor/critic/algorithm hyperparameters for a single trial.
    """
    if hyperparam_ranges is None:
        hyperparam_ranges = DEFAULT_HYPERPARAM_RANGES

    params = {}

    for name, range in hyperparam_ranges.items():
        range_type = type(range)
        if range_type is list:
            params[name] = trial.suggest_categorical(name, range)
        elif range_type is tuple:
            assert len(range) == 2
            log = range[1] / range[0] >= 100
            if isinstance(range[0], int) and isinstance(range[1], int):
                params[name] = trial.suggest_int(name, range[0], range[1], log=log)
            else:
                params[name] = trial.suggest_float(name, range[0], range[1], log=log)
        else:
            raise ValueError(f"Invalid range {range} for {name}.")

    return params


def run_evaluation(config, runner, n_episodes: int, device, save_video: bool = False, metric_name: str = "objective"):
    """Run the trained policy on env for n_episodes and return mean episode return."""
    if save_video:
        config = position_camera_config(config)

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)

    policy = runner.get_inference_policy(device=device)
    # Recurrent policies cache hidden state with training batch size; eval env may use fewer envs (e.g. num_eval_envs).
    # Reset so the first forward allocates hidden state for env.num_envs.
    if hasattr(runner.alg, "actor"):
        runner.alg.actor.reset()
    if hasattr(runner.alg, "student"):
        runner.alg.student.reset()
    num_envs = env.num_envs
    episode_returns = torch.zeros(n_episodes, device="cpu", dtype=gs.tc_float)
    n_completed_episodes = 0
    obs, _ = env.reset()
    current_returns = torch.zeros(num_envs, device=gs.device, dtype=gs.tc_float)
    metric_manager = env.unwrapped.metric_manager

    if save_video:
        rec = env.unwrapped.cameras["rec"]
        rec.start_recording()

    with torch.no_grad():
        while n_completed_episodes < n_episodes:
            actions = policy(obs)
            obs, _, dones, *_ = env.step(actions)
            current_returns += metric_manager._step_metric[metric_name]
            done_flat = dones.flatten().to(torch.bool)
            if done_flat.any():
                done_returns = current_returns[done_flat].cpu()
                n_done_this_step = done_returns.shape[0]
                n_to_record = min(n_done_this_step, n_episodes - n_completed_episodes)
                episode_returns[n_completed_episodes : n_completed_episodes + n_to_record] = done_returns[:n_to_record]
                n_completed_episodes += n_to_record
                current_returns[done_flat] = 0.0
    if save_video:
        rec.stop_recording(runner.logger.log_dir)
    env.close()
    return episode_returns.mean().item() if n_completed_episodes else 0.0
