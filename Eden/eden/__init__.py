"""Eden package entry point: per-rank GPU isolation for DDP and runtime initialization.

Importing :mod:`eden` performs per-rank GPU isolation **before** CUDA is initialized,
so Eden must be imported before ``genesis`` and ``torch`` (importing those first would
pin every rank to the same device). :func:`init` then configures the torch backends,
determinism, and Genesis logging.

Common entry points:

- :func:`init` — set up the runtime (TF32, determinism, per-rank logging).
- ``eden.envs.base.RLEnvBase.from_config`` — build a vectorized environment from a config.
- :data:`eden.tasks.TASK_REGISTRY` — look up built-in task configs by name.
"""

import logging as _stdlib_logging
import os


def _isolate_gpu_for_ddp() -> None:
    """Restrict ``CUDA_VISIBLE_DEVICES`` to one GPU per DDP rank.

    Must execute before any import that may initialize CUDA.
    No-op when ``WORLD_SIZE <= 1`` (single-GPU).
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
        if local_rank >= len(devices):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES has {len(devices)} entries but LOCAL_RANK={local_rank} "
                f"(WORLD_SIZE={world_size}). Ensure CUDA_VISIBLE_DEVICES lists at least "
                f"WORLD_SIZE GPUs or adjust --nproc_per_node."
            )
        gpu = devices[local_rank]
    else:
        gpu = str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["QD_VISIBLE_DEVICE"] = gpu
    os.environ["TI_VISIBLE_DEVICE"] = gpu

    # Give each rank its own Quadrants kernel cache to avoid lock contention.
    # Without this, all ranks fight over the same ticache.lock file and crash.
    default_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "quadrants", "qdcache", "kernel_compilation_manager"
    )
    base_cache = os.environ.get("QD_OFFLINE_CACHE_FILE_PATH", default_cache)
    os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = os.path.join(base_cache, f"rank_{local_rank}")


_isolate_gpu_for_ddp()

import genesis as gs

# Genesis' colorized __repr__ pulls ANSI codes from properties keyed on
# gs._theme; when gs.init() has not run yet these return None and the
# repr ends up with literal "None" strings. Seed a dark-mode default so
# Options pretty-print works at the REPL before gs.init() is called.
# gs.init() still overrides this when invoked.
if gs._theme is None:
    gs._theme = "dark"

import torch
from eden.constants import EventMode
from eden.utils import distributed
from eden.managers.terms import actions
from eden.managers.terms import commands
from eden.managers.terms import curricula
from eden.managers.terms import events
from eden.managers.terms import metrics
from eden.managers.terms import observations
from eden.managers.terms import recorders  # noqa: F401
from eden.managers.terms import rewards
from eden.managers.terms import terminations

# `eden._logging` is the submodule (underscore avoids shadowing stdlib `logging`);
# `eden.logger` (set in `init()`) is the Logger instance.
from eden._logging import Logger
from eden.options import robots
from eden.options import scenes
from eden.options import materials
from eden.options import surfaces

from eden.utils.misc import get_now
from eden.utils.torch import configure_torch_backends


def init(
    backend=gs.cpu,
    initialize_genesis: bool = True,
    debug: bool = False,
    logging_level: int | None = None,
    logger_verbose_time: bool = False,
    log_root_path: str | None = None,
    allow_tf32: bool = True,
    deterministic: bool = False,
    performance_mode: bool = False,
):
    """
    Initialize the Eden environment.

    Parameters
    ----------
    backend: str
        The backend device to use for the simulation.
    initialize_genesis: bool
        Whether to initialize the Genesis here if not already initialized.
    debug: bool
        Whether to run in debug mode.
    logging_level: int | str | None
        The logging level to use.
    logger_verbose_time: bool
        Whether to log the time in the logger.
    log_root_path: str | None
        The root path to use for the logs.
    allow_tf32: bool
        Whether to use TF32 precision for faster computation on Ampere+ GPUs.
    deterministic: bool
        Whether to use deterministic algorithms for reproducibility.
    performance_mode: bool
        Whether to use performance mode for the simulation.
    """
    configure_torch_backends(allow_tf32=allow_tf32, deterministic=deterministic)

    # Suppress Genesis logging on non-main ranks to avoid noisy output.
    gs_log_level = logging_level
    if distributed.is_distributed() and not distributed.is_main_process():
        gs_log_level = "warning"

    if initialize_genesis:
        if not gs._initialized:
            gs.init(
                backend=backend,
                precision="32",
                logging_level=gs_log_level,
                # NOTE: this is for the numerical stability of the simulation
                eps=1e-8,
                seed=int(os.environ.get("LOCAL_RANK", 0)),
                performance_mode=performance_mode,
            )
        else:
            gs.logger.info("Genesis is already initialized")

    # ── Initialize NCCL right after Genesis, before scene building ──
    # This matches the pattern in references/ddp_multi_gpu.py:
    #   gs.init() → torch.cuda.set_device(0) → dist.init_process_group("nccl")
    # Doing it here (before any scene.build) ensures the CUDA context is
    # in a clean state for NCCL communicator creation.
    if distributed.is_distributed():
        torch.cuda.set_device(0)
        distributed.setup("nccl")

    global logger
    if logging_level is None:
        logging_level = _stdlib_logging.DEBUG if debug else _stdlib_logging.INFO
    logger = Logger(logging_level, logger_verbose_time)

    # Only show greeting and version info on the main process.
    if distributed.is_main_process():
        from eden.utils.misc import _display_greeting, get_editable_package_commit

        _display_greeting(logger.INFO_length)

        eden_version = get_editable_package_commit(package_name="eden")
        genesis_version = get_editable_package_commit(package_name="genesis-world")
        logger.info(f"🍎 Eden ver: ~~<{eden_version[:7]}>~~, 🌏 Genesis ver: ~~<{genesis_version[:7]}>~~.")

    if distributed.is_distributed():
        logger.info(
            f"🚀 DDP enabled: rank {distributed.get_global_rank()}/{distributed.get_world_size()}"
            f" (local_rank={distributed.get_local_rank()})"
        )

    # NOTE: create a log directory — broadcast timestamp from rank 0 so all
    # ranks agree on the same path even if clocks are slightly offset.
    global log_dir
    base_log_dir = log_root_path or "./logs"
    timestamp = get_now() if distributed.is_main_process() else None
    if distributed.is_distributed():
        obj_list = [timestamp]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        timestamp = obj_list[0]
    else:
        timestamp = get_now()
    log_dir = os.path.join(base_log_dir, timestamp)
    log_dir = os.path.abspath(log_dir)
    if distributed.is_main_process():
        os.makedirs(log_dir, exist_ok=True)
    logger.info(f"💽 Log directory: ~~<{log_dir}>~~")


__all__ = [
    "init",
    "distributed",
    "EventMode",
    "robots",
    "scenes",
    "actions",
    "commands",
    "curricula",
    "events",
    "metrics",
    "observations",
    "recorders",
    "rewards",
    "terminations",
    "materials",
    "surfaces",
]
