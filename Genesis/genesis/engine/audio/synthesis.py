"""
Small shared DSP primitives for audio synthesis (used by the audio sources; the ContactAudio sensor can adopt these
later when its block loop is ported). All work elementwise so the coefficient tensors broadcast over (B, n, ...).
"""

import torch


def resonator_coeffs(freq: torch.Tensor, decay: torch.Tensor, dt: float):
    """
    Two-pole resonator feedback coefficients ``(a1, a2)`` for a mode at center frequency ``freq`` (Hz) with amplitude
    decay rate ``decay`` (1/s), used as ``y = a1*y1 - a2*y2 + drive``.
    """
    r = torch.exp(-decay * dt)
    return 2.0 * r * torch.cos(2.0 * torch.pi * freq * dt), r * r


def bandpass_coeffs(freq: torch.Tensor, bandwidth: torch.Tensor, dt: float):
    """
    Two-pole band-pass feedback coefficients ``(a1, a2)`` for a noise band centered at ``freq`` (Hz) with spectral
    width ``bandwidth`` (Hz). Same recurrence as :func:`resonator_coeffs`; the pole radius is set by the bandwidth.
    """
    r = torch.exp(-torch.pi * bandwidth * dt)
    return 2.0 * r * torch.cos(2.0 * torch.pi * freq * dt), r * r
