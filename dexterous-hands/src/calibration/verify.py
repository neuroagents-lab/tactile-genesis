"""Verify identified dexterous-hand parameters against a held-out trajectory.

Applies the YAML produced by ``identify.py`` to the sim twin, replays the
held-out action trace, and reports per-signal RMSE against the measured
response — both *before* and *after* applying the identified parameters
— so it is immediately visible whether identification improved the fit.

Robot-agnostic: ``--robot`` selects the hand the sim twin is built for
(default ``xhand1``); it must match the hand the trajectory came from.

Usage
-----

    python src/calibration/verify.py \
        --robot xhand1 \
        --trajectory data/xhand_prbs_verify.npz \
        --params results/xhand_sysid/params_identified.yaml \
        --output results/xhand_sysid/verify/

    Multiple paths and shell-style globs are accepted; every trajectory is
    verified and figures are written into ``--output`` (no ``plots/``
    subdirectory). Filenames are ``<trajectory_stem>_<signal>.png``.
"""

from __future__ import annotations

import glob
import pathlib
import sys
from typing import Mapping, Sequence

import eden as en
import numpy as np
from eden.extensions.sysid import Trajectory

from calibration.action_mod_sysid import PROPERTIES, install_action_mod_sysid_patch
from calibration.sysid_config import ParameterSet, make_argparser, make_sim_twin_config
from calibration.sysid_rollout import single_candidate_rollout


def _sanitize_plot_stem(stem: str) -> str:
    """Avoid path separators in plot filenames when the trajectory stem is odd."""
    return stem.replace("\\", "_").replace("/", "_")


def _expand_trajectory_inputs(items: Sequence[str]) -> list[pathlib.Path]:
    """Resolve ``--trajectory`` entries: literal files and glob patterns."""
    expanded: list[pathlib.Path] = []
    for item in items:
        raw = str(item)
        if glob.has_magic(raw):
            expanded.extend(pathlib.Path(p) for p in sorted(glob.glob(raw)) if pathlib.Path(p).is_file())
        else:
            p = pathlib.Path(raw)
            if not p.is_file():
                raise SystemExit(f"Trajectory file not found: {p}")
            expanded.append(p)
    seen: set[pathlib.Path] = set()
    out: list[pathlib.Path] = []
    for p in expanded:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    if not out:
        raise SystemExit(f"No trajectory files matched: {list(items)!r}")
    return out


def signal_plots(
    predicted: Mapping[str, np.ndarray] | Mapping[str, Mapping[str, np.ndarray]],
    measured: Trajectory,
    signals: Sequence[str],
    save_dir: str | pathlib.Path,
    dof_names: Sequence[str] | None = None,
    *,
    run_label: str | None = None,
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

    For ``dofs_pos``, if ``measured.action`` is present with the same trailing
    dimension as ``dofs_pos`` (true for :class:`DeploymentRecorder` data: the
    column is commanded joint position), it is drawn as **commanded** in
    addition to measured and predicted traces.

    ``run_label`` (e.g. trajectory filename stem) is used in the figure suptitle
    and, if set, in each output filename as ``{run_label}_{signal}.png``.
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
        "measured": "black",
        "commanded": "green",
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

        cmd: np.ndarray | None = None
        if signal == "dofs_pos" and measured.action is not None:
            cmd = np.asarray(measured.action, dtype=np.float64)
            if cmd.ndim == 1:
                cmd = cmd[:, None]
            if cmd.shape[-1] == meas.shape[-1]:
                n = min(n, cmd.shape[0], times.shape[0])
            else:
                cmd = None

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
                color=label_colors["measured"],
            )
            if cmd is not None:
                axes[j].plot(
                    times[:n],
                    cmd[:n, j],
                    label="commanded" if j == 0 else "_commanded",
                    linewidth=1.0,
                    linestyle="-.",
                    color=label_colors["commanded"],
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
        fig.suptitle(f"{run_label} — {signal}" if run_label else signal)
        fig.tight_layout()
        if run_label:
            safe = _sanitize_plot_stem(run_label)
            out = save_dir / f"{safe}_{signal}.png"
        else:
            out = save_dir / f"{signal}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)
    return written


def _rmse(pred: np.ndarray, meas: np.ndarray) -> float:
    n = min(pred.shape[0], meas.shape[0])
    return float(np.sqrt(np.mean((pred[:n] - meas[:n]) ** 2)))


def _per_signal_rmse(
    env,
    params: ParameterSet,
    trajectory: Trajectory,
    signals: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pred = single_candidate_rollout(env, params, trajectory, entity_name="robot", signals=signals)
    out: dict[str, float] = {}
    for s in signals:
        meas = trajectory.signal(s)
        if meas is None or s not in pred:
            continue
        out[s] = _rmse(pred[s], meas)
    return out, pred


def main() -> int:
    parser = make_argparser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more .npz trajectories; entries may be glob patterns (quote wildcards).",
    )
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["dofs_pos"],
        choices=["dofs_pos", "dofs_vel", "dofs_torque"],
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    traj_paths = _expand_trajectory_inputs(args.trajectory)
    signals = tuple(args.signals)

    en.init(performance_mode=False)
    config = make_sim_twin_config(args.robot, num_envs=1, sim_dt=args.sim_dt, decimation=args.decimation)
    from eden.envs.base import RLEnvBase

    env = RLEnvBase.from_config(config)
    env.build()
    install_action_mod_sysid_patch()
    entity = env.entities["robot"]

    identified = ParameterSet.load_yaml(args.params)

    # Baseline = URDF-default values, read straight from the freshly-built sim
    # twin. Same shape / bounds as the identified set so the rollout applies
    # them through the same code path.
    baseline = identified.copy()
    for p in baseline:
        row = PROPERTIES[str(p.property)].read_row(env, list(p.dof_names))
        p.value = row.copy() if p.per_dof else np.array([float(row.mean())])

    args.output.mkdir(parents=True, exist_ok=True)
    any_improved = False
    all_plot_paths: list[pathlib.Path] = []

    for traj_path in traj_paths:
        trajectory = Trajectory.load(traj_path)
        baseline_rmse, baseline_pred = _per_signal_rmse(env, baseline, trajectory, signals)
        identified_rmse, identified_pred = _per_signal_rmse(env, identified, trajectory, signals)

        stem = traj_path.stem
        lines = [
            f"Trajectory {stem}:",
            "signal             baseline RMSE   identified RMSE   improvement",
        ]
        summary: dict[str, dict[str, float]] = {}
        for s in signals:
            b = baseline_rmse.get(s, float("nan"))
            i = identified_rmse.get(s, float("nan"))
            imp = 100.0 * (b - i) / b if (b and np.isfinite(b)) else float("nan")
            lines.append(f"{s:<18} {b:>14.6g} {i:>17.6g} {imp:>12.1f}%")
            summary[s] = {"baseline": b, "identified": i, "improvement_pct": imp}
        report = "\n".join(lines)
        en.logger.info("\n" + report)

        plot_paths = signal_plots(
            predicted={"baseline": baseline_pred, "identified": identified_pred},
            measured=trajectory,
            signals=signals,
            save_dir=args.output,
            dof_names=list(entity.dofs_name),
            run_label=stem,
        )
        all_plot_paths.extend(plot_paths)

        any_improved = any_improved or any(
            np.isfinite(summary[s]["improvement_pct"]) and summary[s]["improvement_pct"] > 1.0 for s in signals
        )

    if all_plot_paths:
        en.logger.info("Saved plot images:\n" + "\n".join(str(p.resolve()) for p in all_plot_paths))

    # Non-zero exit if no trajectory showed >1% RMSE improvement on any requested signal.
    return 0 if any_improved else 2


if __name__ == "__main__":
    sys.exit(main())
