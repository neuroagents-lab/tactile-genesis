"""Reporting helpers for system identification (tables, plots, summary)."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from eden.extensions.sysid.parameter import ParameterSet
from eden.extensions.sysid.trajectory import Trajectory

if TYPE_CHECKING:
    from eden.extensions.sysid.base import FitResult


def parameter_table(
    initial: ParameterSet,
    identified: ParameterSet,
    digits: int = 4,
) -> str:
    """Human-readable comparison of initial vs identified parameters."""
    lines = [f"{'name':<32}{'initial':>14}{'identified':>14}{'rel Δ':>10}"]
    for p in identified:
        init = initial[p.name].value
        for i in range(p.size):
            label = p.name if p.size == 1 else f"{p.name}[{i}]"
            v0, v1 = float(init.ravel()[i]), float(p.value.ravel()[i])
            rel = (v1 - v0) / v0 if abs(v0) > 1e-12 else float("nan")
            lines.append(f"{label:<32}{v0:>14.{digits}g}{v1:>14.{digits}g}{rel * 100:>9.1f}%")
    return "\n".join(lines)


def signal_plots(
    predicted: Mapping[str, np.ndarray] | Mapping[str, Mapping[str, np.ndarray]],
    measured: Trajectory,
    signals: Sequence[str],
    save_dir: str | pathlib.Path,
    dof_names: Sequence[str] | None = None,
) -> list[pathlib.Path]:
    """Write per-signal matplotlib plots comparing one or more predictions against measurement.

    ``predicted`` accepts two forms:
    - **Single trace**: ``{signal_name: array(n_steps, dim)}`` — one
      dashed "predicted" line is plotted alongside the measurement.
    - **Multiple named traces**: ``{label: {signal_name: array(...)}}``
      — each label is drawn with its own line style, e.g.
      ``{"initial": pred0, "identified": pred1}`` to visualise the
      pre- vs post-optimisation fit on the same axes.

    Each signal gets one figure with one subplot per DOF column. Requires
    matplotlib; returns the list of written file paths.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("signal_plots requires matplotlib.") from exc

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    times = measured.times
    dof_names = tuple(dof_names) if dof_names is not None else measured.dof_names

    # Normalise predicted -> dict[label, dict[signal, array]].
    first_value = next(iter(predicted.values())) if predicted else None
    is_multi = isinstance(first_value, Mapping)
    if is_multi:
        traces: dict[str, Mapping[str, np.ndarray]] = dict(predicted)  # type: ignore[assignment]
    else:
        traces = {"predicted": predicted}  # type: ignore[dict-item]

    # Measurement = dotted reference. Predictions = continuous; semantically
    # meaningful labels get fixed colors so pre/post-fit are instantly
    # readable across runs. Unknown labels fall back to the default cycle.
    label_colors: dict[str, str] = {
        "initial": "lightblue",
        "baseline": "lightblue",
        "identified": "red",
    }
    written: list[pathlib.Path] = []

    for signal in signals:
        meas = measured.signal(signal)
        signal_traces = [(label, pred.get(signal)) for label, pred in traces.items()]
        signal_traces = [(label, arr) for label, arr in signal_traces if arr is not None]
        if meas is None or not signal_traces:
            continue
        n = min(min(arr.shape[0] for _, arr in signal_traces), meas.shape[0])
        dim = signal_traces[0][1].shape[-1]

        fig, axes = plt.subplots(dim, 1, figsize=(10, 1.2 * max(dim, 2)), sharex=True)
        if dim == 1:
            axes = [axes]
        for j in range(dim):
            axes[j].plot(
                times[:n],
                meas[:n, j],
                label="measured",
                linewidth=1.0,
                linestyle=":",
                color="black",
            )
            for label, pred in signal_traces:
                axes[j].plot(
                    times[:n],
                    pred[:n, j],
                    label=label,
                    linewidth=1.0,
                    linestyle="-",
                    color=label_colors.get(label),
                )
            title = dof_names[j] if dof_names and j < len(dof_names) else f"{signal}[{j}]"
            axes[j].set_ylabel(title, fontsize=8)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(signal)
        fig.tight_layout()
        out = save_dir / f"{signal}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)
    return written


def write_summary(
    save_dir: str | pathlib.Path,
    initial: ParameterSet,
    identified: ParameterSet,
    result: "FitResult",
) -> None:
    """Write identified parameters as YAML plus a human-readable summary.txt."""
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    identified.save_yaml(save_dir / "params_identified.yaml")
    initial.save_yaml(save_dir / "params_initial.yaml")

    table = parameter_table(initial, identified)
    header = f"cost={result.cost:.6g} nfev={result.nfev} {result.message}".strip()
    (save_dir / "summary.txt").write_text(f"{header}\n\n{table}\n")
