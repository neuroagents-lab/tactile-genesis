"""RSL RL environment wrapper for Eden.

Auto-detects the installed ``rsl_rl`` version and translates the runner
config dict accordingly.

Usage::

    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner

    env = RslRlVecEnvWrapper.from_config(config)
    runner = make_rsl_rl_runner(env)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

import eden as en
from eden.options.learning.rsl_rl.translate import translate_runner_dict
from eden.utils import distributed as eden_dist

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions


class RslRlVecEnvWrapper(VecEnv, gym.Env):
    """Wrapper for RSL RL environments.

    Works with rsl_rl v3 / v4 / v5 — the ``VecEnv`` interface is identical
    across all versions.  The ``runner_dict`` property automatically
    translates the Eden config to whatever format the installed version
    expects.

    Parameters
    ----------
    env: RLEnvBase
        The environment to wrap.
    clip_actions: float | None
        The action clip value.
    """

    def __init__(
        self,
        env: RLEnvBase,
        options: RslRlBaseRunnerOptions,
    ):
        self._runner_dict_cache: dict | None = None

        self.env = env
        self.runner_options = options
        self.clip_actions = options.clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self._configure_gym_env_spaces()

        # Reset at the start since rsl_rl does not call reset.
        self.unwrapped.reset()

    @classmethod
    def from_config(cls, cfg, **kwargs) -> "RslRlVecEnvWrapper":
        """Create a wrapper from a configuration object.

        Parameters
        ----------
        cfg: EdenRLConfig
            The configuration object.
        **kwargs: dict
            Additional keyword arguments forwarded to
            ``RLEnvBase.from_config`` (e.g. ``show_viewer``, ``eval_mode``).

        Warning
        -------
        Unlike ``RLEnvBase.from_config``, this method also **builds** the
        environment.
        """
        from eden.envs.base import RLEnvBase

        env = RLEnvBase.from_config(cfg, **kwargs)
        env.build()
        en.logger.info(env.summary())
        return cls(env, options=cfg.runner_options)

    @property
    def runner_dict(self) -> dict:
        """Version-translated runner config dict, ready for ``OnPolicyRunner``.

        Auto-detects the installed rsl_rl version and translates the runner
        options accordingly.  The result is cached.
        """
        if self._runner_dict_cache is None:
            from eden.utils.torch import resolve_device

            options = self.runner_options.dict()
            options["device"] = resolve_device(options["device"])
            self._runner_dict_cache = translate_runner_dict(options)
        return self._runner_dict_cache

    @property
    def config(self):
        base = self.unwrapped.config
        cfg = type(base).model_construct(**{name: getattr(base, name) for name in base.model_fields})
        object.__setattr__(cfg, "runner_options", self.runner_options)
        return cfg

    # rsl_rl's OnPolicyRunner accesses env.cfg internally
    cfg = config

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> RLEnvBase:
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    @property
    def dt(self) -> float:
        return self.unwrapped.dt

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.unwrapped.reset()
        return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.unwrapped.step(actions)
        term_or_trunc = terminated | truncated
        assert isinstance(rew, torch.Tensor)
        assert isinstance(term_or_trunc, torch.Tensor)
        dones = term_or_trunc.to(dtype=torch.long)
        if self.unwrapped.max_episode_length_s == float("inf"):
            extras["time_outs"] = truncated
        return (
            TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]),
            rew,
            dones,
            extras,
        )

    def close(self) -> None:
        pass

    def _configure_gym_env_spaces(self) -> None:
        self.single_observation_space = gym.spaces.Dict()
        for (
            group_name,
            group_term_names,
        ) in self.unwrapped.observation_manager.active_terms.items():
            has_concatenated_obs = self.unwrapped.observation_manager._group_obs_options[group_name].concatenate_terms
            group_dim = self.unwrapped.observation_manager.group_obs_dim[group_name]
            if has_concatenated_obs:
                assert isinstance(group_dim, tuple)
                self.single_observation_space[group_name] = gym.spaces.Box(
                    low=-math.inf, high=math.inf, shape=group_dim
                )
            else:
                assert not isinstance(group_dim, tuple)
                for term_name, term_dim in zip(group_term_names, group_dim, strict=False):
                    self.single_observation_space[group_name] = gym.spaces.Dict(
                        {term_name: gym.spaces.Box(low=-math.inf, high=math.inf, shape=term_dim)}
                    )

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

        if self.clip_actions is not None:
            self.single_action_space = gym.spaces.Box(
                shape=(self.num_actions,),
                low=-self.clip_actions,
                high=self.clip_actions,
            )
        else:
            self.single_action_space = gym.spaces.Box(shape=(self.num_actions,), low=-math.inf, high=math.inf)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


# runners


class DDPRunnerMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # rsl_rl's ``Logger.process_env_step`` gates on ``log_dir is not None``
        # (not on ``disable_logs``), so on non-zero ranks the rollout loop
        # appends ``extras["log"]`` to ``Logger.ep_extras`` every step.
        # ``Logger.log`` then short-circuits on ``disable_logs`` and never
        # reaches its ``ep_extras.clear()`` — the dicts hold GPU tensor refs
        # from reward / metric / command managers, so the list pins VRAM
        # that grows linearly with iters. (Issue #469.) Neutralize the
        # append on non-zero ranks without touching ``log_dir`` (which
        # ``OnPolicyRunner.save`` still uses for the periodic-checkpoint
        # path that runs on every rank).
        if self.is_distributed and self.gpu_global_rank != 0:
            self.logger.process_env_step = lambda *_a, **_k: None

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = eden_dist.get_world_size()
        self.is_distributed = eden_dist.is_distributed()

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = eden_dist.get_local_rank()
        self.gpu_global_rank = eden_dist.get_global_rank()
        # CUDA_VISIBLE_DEVICES isolation in eden.__init__ makes gs.device == cuda:0
        # for every rank; routing through resolve_device keeps DDP and non-DDP
        # device selection on the same code path.
        from eden.utils.torch import resolve_device

        self.device = resolve_device("auto")

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        self.cfg["multi_gpu"] = self.multi_gpu_cfg


class DDPOnPolicyRunner(DDPRunnerMixin, OnPolicyRunner):
    """Drop-in replacement for :class:`OnPolicyRunner` for Eden's single-GPU-per-process setup.

    Expects the NCCL process group to be **already initialized** by
    ``eden.init()``.  The device is always ``cuda:0`` because
    ``CUDA_VISIBLE_DEVICES`` restricts each process to one physical GPU.

    Usage
    -----
    ``torchrun --standalone --nproc_per_node=NUM_GPUS my_script.py``
    """


class DDPDistillationRunner(DDPRunnerMixin, DistillationRunner):
    """Drop-in replacement for :class:`DistillationRunner` for Eden's single-GPU-per-process setup.

    Expects the NCCL process group to be **already initialized** by
    ``eden.init()``.  The device is always ``cuda:0`` because
    ``CUDA_VISIBLE_DEVICES`` restricts each process to one physical GPU.

    Usage
    -----
    ``torchrun --standalone --nproc_per_node=NUM_GPUS my_script.py``
    """


def _detect_std_type(runner_dict: dict) -> str | None:
    """Find the destination Gaussian std parameterisation in a runner dict.

    Returns 'scalar', 'log', or None if unknown. Handles rsl_rl v3 / v4 / v5
    layouts produced by :func:`translate_runner_dict`:

    - v5 (current): ``runner_dict["actor"]["distribution_cfg"]["std_type"]``
    - v4:           same key path as v5
    - v3:           ``runner_dict["policy"]["noise_std_type"]``
    """
    actor = runner_dict.get("actor")
    if isinstance(actor, dict):
        dist = actor.get("distribution_cfg")
        if isinstance(dist, dict) and isinstance(dist.get("std_type"), str):
            return dist["std_type"]
    policy = runner_dict.get("policy")
    if isinstance(policy, dict) and isinstance(policy.get("noise_std_type"), str):
        return policy["noise_std_type"]
    return None


def _maybe_convert_legacy_checkpoint(path: str, *, std_type: str) -> str:
    """Return ``path`` unchanged for new-format checkpoints, else convert.

    rsl_rl's current ``PPO.load`` expects top-level ``actor_state_dict`` /
    ``critic_state_dict`` keys, each shaped like the new ``MLPModel``
    state dict (``mlp.*`` / ``obs_normalizer.*`` / ``distribution.*``).
    Older Eden runs wrote a single combined ``model_state_dict`` with
    ``actor.*`` / ``actor_obs_normalizer.*`` / ``critic.*`` /
    ``critic_obs_normalizer.*`` / ``std`` keys. Loading those into the
    current rsl_rl raises ``KeyError: 'actor_state_dict'``.

    When we detect the legacy layout we write a sibling
    ``<name>.converted.pt`` next to the original (or to a temp file if
    that directory isn't writable) and return that path. The original
    file is never modified. The conversion is **idempotent**: if the
    sibling exists *and* post-dates the source, it's reused without
    re-reading the original checkpoint — so repeat ``eden inference``
    calls on the same legacy file pay one ``torch.load`` only on the
    first invocation. Touching the source ``ckpt.pt`` (e.g. re-training
    overwrites it) automatically invalidates the cached sibling and
    forces re-conversion.

    Parameters
    ----------
    path:
        Filesystem path to the checkpoint to (possibly) convert.
    std_type:
        Destination Gaussian distribution parameterisation, one of
        ``'scalar'`` (writes ``distribution.std_param``) or ``'log'``
        (writes ``distribution.log_std_param``). MUST match the
        ``RslRlGaussianDistributionOptions.std_type`` of the task whose
        policy will be loaded — passing the wrong value silently
        miscompiles the checkpoint (a ``std_param`` of size N gets
        populated with raw std values when the forward path expects log
        std, producing NaN actions on first sample). The caller in
        :func:`make_rsl_rl_runner` pulls this from the runner dict via
        :func:`_detect_std_type`, so user code does not normally need to
        think about it.
    """
    import os
    import tempfile

    if std_type not in ("scalar", "log"):
        raise ValueError(
            f"_maybe_convert_legacy_checkpoint: std_type must be 'scalar' or "
            f"'log', got {std_type!r}. Cannot safely convert the legacy "
            f"``std`` parameter without knowing the destination."
        )

    # Fast path: if the converted sibling already exists and post-dates the
    # source, skip the entire load/convert/save cycle. Saves ~50-200 ms per
    # ``eden inference`` invocation on the second+ run against the same
    # legacy checkpoint (typical iterate-on-policy-eval workflow).
    #
    # The cache filename includes ``std_type`` so the same legacy checkpoint
    # converted for a ``'scalar'`` task vs a ``'log'`` task doesn't collide:
    # the two conversions route ``std`` to *different* parameters (with a
    # log() transform in the latter), so a single cache entry would silently
    # serve the wrong one when the destination switches.
    base, ext = os.path.splitext(path)
    cached_out_path = f"{base}.{std_type}.converted{ext or '.pt'}"
    if (
        os.path.exists(cached_out_path)
        and os.path.exists(path)
        and os.path.getmtime(cached_out_path) >= os.path.getmtime(path)
    ):
        en.logger.info(f"Reusing cached converted checkpoint at {cached_out_path}.")
        return cached_out_path

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "actor_state_dict" in state or "model_state_dict" not in state:
        return path

    msd = state["model_state_dict"]
    actor_sd: dict[str, Any] = {}
    critic_sd: dict[str, Any] = {}
    for k, v in msd.items():
        if k == "std":
            # Legacy rsl_rl saved RAW std (not log-std). Map to whichever
            # parameter the current rsl_rl GaussianDistribution exposes:
            # 'scalar' learns std directly, 'log' learns log(std).
            if std_type == "scalar":
                actor_sd["distribution.std_param"] = v
            else:  # 'log'
                # Clamp keeps log(0) out of the picture for any ill-formed
                # legacy checkpoints.
                actor_sd["distribution.log_std_param"] = torch.log(v.clamp(min=1e-6))
        elif k == "log_std":
            # Defensive: rsl_rl variants that stored log(std) directly.
            if std_type == "log":
                actor_sd["distribution.log_std_param"] = v
            else:  # 'scalar'
                actor_sd["distribution.std_param"] = torch.exp(v)
        elif k.startswith("actor_obs_normalizer."):
            actor_sd["obs_normalizer." + k[len("actor_obs_normalizer.") :]] = v
        elif k.startswith("actor."):
            actor_sd["mlp." + k[len("actor.") :]] = v
        elif k.startswith("critic_obs_normalizer."):
            critic_sd["obs_normalizer." + k[len("critic_obs_normalizer.") :]] = v
        elif k.startswith("critic."):
            critic_sd["mlp." + k[len("critic.") :]] = v
        else:
            en.logger.warning(f"Legacy checkpoint at {path}: dropping unrecognised key {k!r}")

    converted = {
        "actor_state_dict": actor_sd,
        "critic_state_dict": critic_sd,
        "optimizer_state_dict": state.get("optimizer_state_dict"),
        "iter": state.get("iter", 0),
        "infos": state.get("infos"),
    }

    out_path = cached_out_path
    try:
        torch.save(converted, out_path)
    except OSError:
        out_path = tempfile.NamedTemporaryFile(suffix=".converted.pt", delete=False).name
        torch.save(converted, out_path)

    en.logger.info(
        f"Converted legacy rsl_rl checkpoint {path} -> {out_path} "
        f"(model_state_dict -> actor_state_dict/critic_state_dict, std_type={std_type!r})"
    )
    return out_path


def _audit_rsl_rl_checkpoint_keys(runner, checkpoint_path: str, device) -> None:
    """Log a summary of key mismatches between checkpoint and runner models.

    rsl_rl's ``runner.load`` uses ``load_state_dict(..., strict=False)``,
    which silently skips keys present in the model but absent in the
    checkpoint (and ignores keys present only in the checkpoint). When
    loading converted legacy checkpoints — or any checkpoint produced by a
    slightly-different model spec — this can leave parts of the policy
    randomly initialized and the user has no visible signal until the
    robot starts behaving oddly. We audit before the load and surface
    ``missing`` / ``unexpected`` counts via ``en.logger`` so the silent
    no-load failure mode becomes observable.

    No effect on load behaviour — purely diagnostic. The runner-side load
    happens immediately after; if the audit fails to find a head it logs
    a warning and skips that side.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        en.logger.warning(f"[rsl_rl] Checkpoint audit could not load {checkpoint_path}: {exc}")
        return

    if not isinstance(ckpt, dict):
        return

    for side, alg_attr in (("actor", "actor"), ("critic", "critic")):
        ckpt_key = f"{side}_state_dict"
        if ckpt_key not in ckpt:
            en.logger.warning(f"[rsl_rl] Checkpoint has no '{ckpt_key}' — {side} will use random init.")
            continue
        model = getattr(runner.alg, alg_attr, None)
        if model is None:
            continue
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt[ckpt_key].keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        if missing or unexpected:
            en.logger.warning(
                f"[rsl_rl] {side} checkpoint key audit — "
                f"missing in ckpt: {len(missing)}, unexpected in ckpt: {len(unexpected)}. "
                f"With strict=False these are silently ignored — "
                f"missing buffers/weights will stay at their random init. "
                f"Sample missing: {missing[:3]}; sample unexpected: {unexpected[:3]}."
            )
        else:
            en.logger.info(f"[rsl_rl] {side} checkpoint keys match model state_dict exactly.")


def make_rsl_rl_runner(
    env: RslRlVecEnvWrapper,
    checkpoint: str | None = None,
    log_dir: str | None = None,
):
    """Build an rsl_rl runner for the given environment, optionally loading a checkpoint.

    Parameters
    ----------
    env: RslRlVecEnvWrapper
        The environment to create a runner for.
    checkpoint: str | None
        The checkpoint to load.
    log_dir: str | None
        The directory to save the logs to.

    Returns
    -------
    runner: OnPolicyRunner | DistillationRunner
        The runner for the environment.
    """
    log_dir = log_dir if log_dir is not None else en.log_dir

    if "distill" in env.runner_dict["class_name"].lower():
        runner_cls = DDPDistillationRunner if eden_dist.is_distributed() else DistillationRunner
    else:
        runner_cls = DDPOnPolicyRunner if eden_dist.is_distributed() else OnPolicyRunner
    runner = runner_cls(env, env.runner_dict, log_dir=log_dir, device=env.runner_dict["device"])

    if checkpoint:
        std_type = _detect_std_type(env.runner_dict)
        if std_type is None:
            # Can't safely guess — refuse to convert and let rsl_rl produce
            # its own error rather than silently mis-mapping ``std``.
            en.logger.warning(
                f"Could not detect std_type from runner_dict; skipping legacy "
                f"checkpoint conversion for {checkpoint}. If this is a legacy "
                f"checkpoint, you'll see the usual KeyError: 'actor_state_dict' "
                f"from rsl_rl — add std_type to the task's distribution_cfg."
            )
        else:
            checkpoint = _maybe_convert_legacy_checkpoint(checkpoint, std_type=std_type)
        # Pre-load diagnostic: surface silent key drift that ``strict=False``
        # would otherwise swallow. Purely informational; load proceeds either
        # way so a mismatched key still results in a partially-loaded policy
        # (just one the user knows about).
        _audit_rsl_rl_checkpoint_keys(runner, checkpoint, runner.device)
        runner.load(checkpoint, map_location=runner.device, strict=False)

    return runner
