"""Random-seed utilities."""

import os
import random

import numpy as np
import torch


def set_random_seed(seed: int, torch_deterministic: bool = False) -> None:
    """Seed all random number generators for reproducibility.

    Note: MuJoCo Warp is not fully deterministic yet.
    See: https://github.com/google-deepmind/mujoco_warp/issues/562
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)  # Seed RNG for all devices.
    # torch.cuda.manual_seed* raises on CPU-only / MPS builds (Mac, Windows-without-CUDA).
    # torch.manual_seed above already covers CUDA when it's present, so the explicit calls
    # are belt-and-braces — gate them on availability to keep init portable.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Use deterministic algorithms when possible.
    if torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
