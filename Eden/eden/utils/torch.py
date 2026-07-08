"""Torch backend configuration and model-compile helpers."""

# Adapted from: https://github.com/mujocolab/mjlab/blob/main/src/mjlab/utils/torch.py
import torch
import warnings
from typing import Annotated, Callable, TypeVar
from packaging.version import parse

from pydantic import StringConstraints

T = TypeVar("T", bound=torch.nn.Module | Callable)


# Pydantic-validated device string. Accepts the ``"auto"`` sentinel (resolved
# lazily by ``resolve_device``), the unindexed forms ``"cpu" / "mps" / "cuda"``,
# and any indexed accelerator form ``"mps:N" / "cuda:N"``. Rejects typos like
# ``"cdua:0"`` or aliases like ``"gpu"`` at config-load time so the failure is
# a clear ValidationError instead of a confusing torch error several frames in.
DeviceStr = Annotated[
    str,
    StringConstraints(pattern=r"^(auto|cpu|(mps|cuda)(:\d+)?)$"),
]


def resolve_device(device: DeviceStr) -> str:
    """Resolve the sentinel ``"auto"`` to the active Genesis device.

    Allows runner config defaults (e.g. ``RslRlBaseRunnerOptions.device``) to
    avoid hardcoding ``"cuda:0"`` — which crashes on CPU-only Macs and
    Windows boxes without CUDA. Any non-``"auto"`` string is returned
    unchanged so explicit user choices still win.
    """
    if device != "auto":
        return device
    import genesis as gs

    if gs.device is None:
        raise RuntimeError(
            "runner device set to 'auto' but Genesis is not initialised yet — "
            "call `eden.init(...)` (which runs `gs.init`) before constructing the runner."
        )
    return str(gs.device)


def configure_torch_backends(allow_tf32: bool = True, deterministic: bool = False):
    """Configure PyTorch CUDA and cuDNN backends for performance/reproducibility.

    Parameters
    ----------
    allow_tf32: bool
        If True, use TF32 precision for faster computation on Ampere+ GPUs. If
        False, use standard IEEE FP32 precision.
    deterministic: bool
        If True, use deterministic algorithms (slower but reproducible).
        If False, allow cuDNN to benchmark and select fastest algorithms.

    Notes
    -----
    TF32 uses reduced precision (10-bit mantissa vs 23-bit for FP32) for internal
    matrix multiplications providing a speedup with minimal impact on accuracy.

    See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere for details.
    """
    torch_version = parse(torch.__version__.split("+")[0])  # Handle e.g., "2.9.0+cu118".
    if torch_version >= parse("2.9.0"):
        _configure_29(allow_tf32)
    else:
        _configure_pre29(allow_tf32)


def _configure_29(allow_tf32: bool):
    """Configure PyTorch CUDA and cuDNN backends for PyTorch 2.9+."""
    # tf32 for performance, ieee for full FP32 accuracy.
    precision = "tf32" if allow_tf32 else "ieee"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.fp32_precision = precision  # type: ignore


def _configure_pre29(allow_tf32: bool):
    """Configure PyTorch CUDA and cuDNN backends for PyTorch <2.9."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def compile_model(model: T, mode: str = "reduce-overhead") -> T:
    """Safely compile a model or function if ``torch.compile`` is available.

    Parameters
    ----------
    model : T
        The ``torch.nn.Module`` or function to compile.
    mode : str, optional
        The compilation mode. ``"reduce-overhead"`` is recommended for RL inference.

    Returns
    -------
    T
        The compiled model/function, or the original if compilation fails/is unavailable.
    """
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode=mode)
        except Exception as e:
            warnings.warn(f"torch.compile failed, using eager mode. Error: {e}")
            return model
    return model
