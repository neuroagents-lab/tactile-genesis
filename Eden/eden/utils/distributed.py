"""
Distributed Data Parallel (DDP) utilities for Eden.

This module provides a simple, universal interface for multi-GPU training
with PyTorch DDP.  Eden uses the **NCCL** backend for collective
communication, matching the pattern in ``references/ddp_multi_gpu.py``.

Each process restricts ``CUDA_VISIBLE_DEVICES`` to one physical GPU so
that Genesis and PyTorch both see ``cuda:0``.  The NCCL process group is
initialized early in ``eden.init()`` — right after ``gs.init()`` and
before any scene building — so that NCCL gets a clean CUDA context.

Usage with ``torchrun``::

    torchrun --standalone --nproc_per_node=2 my_script.py

The module reads ``LOCAL_RANK``, ``RANK``, and ``WORLD_SIZE`` environment
variables set by ``torchrun`` (or any compatible launcher).
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch.distributed as dist


# ---------------------------------------------------------------------------
# Rank / world-size queries (work before and after setup)
# ---------------------------------------------------------------------------


def get_local_rank() -> int:
    """Return the local rank of the current process (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank() -> int:
    """Return the global rank of the current process."""
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Return the total number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    """Return ``True`` if running in a multi-process (DDP) context."""
    return get_world_size() > 1


def is_main_process() -> bool:
    """Return ``True`` if this is rank 0 (the main process)."""
    return get_global_rank() == 0


# ---------------------------------------------------------------------------
# Process-group lifecycle
# ---------------------------------------------------------------------------


def setup(backend: str = "nccl") -> None:
    """Initialize the ``torch.distributed`` process group.

    This uses ``init_method="env://"``, which expects ``MASTER_ADDR``,
    ``MASTER_PORT``, ``RANK``, and ``WORLD_SIZE`` to be set by the launcher
    (e.g. ``torchrun``).

    Parameters
    ----------
    backend : str
        Communication backend (default ``"nccl"``).
    """
    if dist.is_initialized():
        return
    if not is_distributed():
        return
    dist.init_process_group(backend=backend, init_method="env://")


def cleanup() -> None:
    """Destroy the process group and synchronize all ranks."""
    if not dist.is_initialized():
        return
    dist.barrier()
    dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes. No-op if not distributed."""
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Rank-aware helpers
# ---------------------------------------------------------------------------


def rank_print(*args: Any, **kwargs: Any) -> None:
    """Print with a ``[RANK x/N]`` prefix."""
    prefix = f"[RANK {get_global_rank()}/{get_world_size()}]"
    print(prefix, *args, **kwargs)


def main_only(fn):
    """Make a decorated function execute only on the main process."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_main_process():
            return fn(*args, **kwargs)
        return None

    return wrapper
