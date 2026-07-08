"""Sampling helpers (e.g. per-env probability gating)."""

import torch

from eden.utils.isaac_math import sample_uniform  # noqa: F401  (re-export)


def apply_probability(mask: torch.Tensor, probability: float) -> torch.Tensor:
    """Optionally gate a per-env boolean mask by a global Bernoulli draw.

    A single sample is drawn per call (not per env). ``probability=1.0`` is a no-op,
    ``probability=0.0`` zeroes the mask. Used by probabilistic-soft-limit termination
    terms — gating is coarse on purpose, so they behave as a stochastic safety net
    rather than a hard cap. Mirrors the convention in
    ``references/holosoma/.../termination/terms/locomotion.py:9-16``.

    The intermediate path uses a 0-dim bool tensor and a tensor-level ``&``
    instead of a Python ``if`` on ``sample.item()``, so the function stays
    async on GPU (no device→host sync per call).

    Raises ``ValueError`` if ``probability`` is NaN or outside ``[0, 1]`` — silently
    zeroing every termination on a misconfiguration would mask the bug.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability!r}.")
    # Constant cases short-circuit so we don't allocate a random sample.
    if probability >= 1.0:
        return mask
    if probability <= 0.0:
        return torch.zeros_like(mask, dtype=torch.bool)
    # Tensor-level gate: 0-dim bool broadcasts to the mask shape.
    gate = torch.rand((), device=mask.device) < probability
    return mask & gate
