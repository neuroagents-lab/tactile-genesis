"""Record + live-visualize xhand1 fingertip tactile readings: real hand vs sim.

Used by both ``main.py`` deploy mode and ``scripts/manual_calibration_gui.py``.
The real hand exposes per-finger aggregate force (``calc_pressure``) via
``RobotState.extra["fingertip_sensors"]``; the sim exposes the same quantity
through ``agg_force`` tactile sensors (``postprocess_agg_force`` converts a
sensor's raw read into the identical ``[fx, fy, fz]`` convention). This module
aligns the two by canonical finger index, writes them to a CSV in the log dir,
and feeds a 5-subplot live line plot (one finger per subplot, fx/fy/fz x sim/real).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from genesis.options.recorders import MPLLinePlot as _MPLLinePlotOptions
from genesis.recorders import register_recording as _register_recording
from genesis.recorders.plotters import MPLLinePlotter as _MPLLinePlotter

from tactile_sensors import TACTILE_SENSORS

# Canonical finger order. Position i in all three tuples is the same physical
# finger; ``FINGER_IDS`` are the real-hand finger ids used by the xhand SDK
# (eden ``_SENSOR_FINGERS``), ``FINGERTIP_LINKS`` the matching sim link names.
FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "mid", "ring", "pinky")
FINGER_IDS: tuple[int, ...] = (2, 5, 7, 9, 11)
FINGERTIP_LINKS: tuple[str, ...] = (
    "right_hand_thumb_rota_link2",
    "right_hand_index_rota_link2",
    "right_hand_mid_link2",
    "right_hand_ring_link2",
    "right_hand_pinky_link2",
)
_REAL_ID_TO_INDEX: dict[int, int] = {fid: i for i, fid in enumerate(FINGER_IDS)}
_AXES: tuple[str, ...] = ("fx", "fy", "fz")

# Per-sensor-type channel layout for the recorder + plot. Each entry lists the
# channel names that ``read_sim_tactile`` / ``read_real_tactile`` produce for that
# sensor type (one float per channel per finger); the recorder uses this to size
# the CSV columns, the plot legend, and the per-line styling. ``bool`` collapses
# to a single "any contact on this finger" bit; ``agg_force`` keeps the three
# Cartesian force axes.
TACTILE_PLOT_CHANNELS: dict[str, tuple[str, ...]] = {
    "agg_force": _AXES,
    "bool": ("bit",),
}

# Real-hand tactile calibration params (per-finger calc_pressure scale).
SENSOR_PARAM_PATH: Path = Path(__file__).resolve().parents[1] / "conf" / "sensor" / "xhand1_deploy_sensor_params.yaml"


# =============================== mapping helpers ===============================


def fingertip_link_order(robot_cfg: Any = None) -> list[str]:
    """Ordered fingertip link names, from robot metadata when available."""
    metadata = getattr(robot_cfg, "metadata", None)
    links = getattr(metadata, "fingertip_links", None)
    if links:
        return list(links)
    return list(FINGERTIP_LINKS)


def resolve_sim_sensor_names(
    env: Any, *, fingertip_links: Sequence[str], sensor_type: str = "agg_force"
) -> dict[int, str]:
    """Map canonical finger index -> sim ``tactile_<sensor_type>_<link>`` sensor name.

    Scans the env's *actual* sensors (``env.sensors``), so it works whether the
    sensors were enabled via ``--sensors`` or baked into a loaded config. Only
    sensors with a ``xhand1_func`` (``agg_force``, ``bool``) yield a per-finger
    reading comparable to the real hand; an empty result means the caller
    records a real-only (sim = NaN) trace.
    """
    available = set(getattr(env, "sensors", {}).keys())
    resolved: dict[int, str] = {}
    prefix = f"tactile_{sensor_type}_"
    for idx, link in enumerate(fingertip_links):
        name = f"{prefix}{link}"
        if name in available:
            resolved[idx] = name
    return resolved


def resolve_real_finger_ids(deploy_state: Any = None) -> list[int]:
    """Real-hand finger ids in canonical order (from state extra when present)."""
    extra = getattr(deploy_state, "extra", None)
    if isinstance(extra, dict):
        order = extra.get("fingertip_sensor_order")
        if isinstance(order, (list, tuple)) and order:
            return list(order)
    return list(FINGER_IDS)


def finger_indices(names: Sequence[str]) -> list[int]:
    """Map finger names (e.g. ``["thumb", "index"]``) to sorted canonical indices."""
    out: list[int] = []
    for name in names:
        key = str(name).strip().lower()
        if key not in FINGER_NAMES:
            raise ValueError(f"Unknown finger {name!r}; expected one of {list(FINGER_NAMES)}.")
        out.append(FINGER_NAMES.index(key))
    return sorted(set(out))


def load_tactile_scales(path: str | Path | None = None) -> dict[int, np.ndarray]:
    """Per-finger ``[fx, fy, fz]`` calibration multipliers for the real xhand1 tactile reading.

    Reads :data:`SENSOR_PARAM_PATH` (``conf/sensor/xhand1_deploy_sensor_params.yaml``):
    a global ``scale``, optional ``per_finger`` name overrides, and a per-axis
    ``axis_scale`` (the real sensor reports fx/fy with the opposite sign to sim).
    Returns ``{finger_id: array([sx, sy, sz])}`` (= ``per_finger_scale *
    axis_scale``) for the five sensor fingers ``(2, 5, 7, 9, 11)``. A missing
    file yields all ones (identity -- no scaling, no sign flip).
    """
    yaml_path = Path(path) if path is not None else SENSOR_PARAM_PATH
    scale = 1.0
    per_finger: dict[str, float] = {}
    axis_scale = np.ones(3, dtype=np.float64)
    if yaml_path.is_file():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        scale = float(data.get("scale", 1.0))
        per_finger = {str(k).strip().lower(): float(v) for k, v in (data.get("per_finger") or {}).items()}
        raw_axis = data.get("axis_scale")
        if raw_axis is not None:
            axis_scale = np.asarray(raw_axis, dtype=np.float64).reshape(-1)
            if axis_scale.shape != (3,):
                raise ValueError(f"axis_scale must have 3 values [fx, fy, fz], got {list(raw_axis)!r}.")
    return {fid: per_finger.get(name, scale) * axis_scale for fid, name in zip(FINGER_IDS, FINGER_NAMES)}


# ================================== readers ====================================


def read_sim_tactile(
    env: Any, sim_sensor_names: dict[int, str], *, device: Any, sensor_type: str = "agg_force"
) -> dict[int, np.ndarray]:
    """Per-finger sim tactile reading for env 0, keyed by canonical finger index.

    Shape per finger depends on ``sensor_type`` (see :data:`TACTILE_PLOT_CHANNELS`):
    ``agg_force`` returns ``[fx, fy, fz]``; ``bool`` returns a single ``[bit]``
    that is 1.0 when *any* taxel on the finger is in contact, else 0.0.

    Fingers whose sensor is missing are simply absent from the result. We always
    collapse the gs sensor's within-step history via ``temporal_reduction='median'``
    so the recorder/plot sees one float per channel per finger regardless of how
    the policy's ``TactileSensorRead`` term is configured.
    """
    spec = TACTILE_SENSORS.get(sensor_type)
    if spec is None:
        raise ValueError(f"Unknown sensor_type {sensor_type!r}; expected one of {list(TACTILE_SENSORS)}.")
    num_envs = getattr(env, "num_envs", 1)
    out: dict[int, np.ndarray] = {}
    for idx, name in sim_sensor_names.items():
        sensor = env.sensors[name]
        feat = spec.postprocess(
            sensor.read(), sensor=sensor, num_envs=num_envs, device=device, temporal_reduction="median"
        )
        per_finger = feat[0].detach().cpu()
        if sensor_type == "bool":
            # ``ContactProbe.postprocess`` returns one 0/1 float per taxel; the real hand
            # cannot resolve per-taxel data, so we collapse the per-finger taxel vector
            # to a single "any contact" bit. ``any()`` keeps the semantic crisp (matches
            # the real-side thresholded magnitude bit).
            out[idx] = np.asarray([float(per_finger.any())], dtype=np.float64)
        else:
            out[idx] = np.asarray(per_finger, dtype=np.float64).reshape(-1)
    return out


def read_real_tactile(
    deploy_state: Any, *, sensor_type: str = "agg_force", threshold: float = 0.0
) -> dict[int, np.ndarray]:
    """Per-finger real-hand tactile reading, keyed by canonical finger index.

    Shape per finger depends on ``sensor_type``:
    ``agg_force`` returns ``calc_pressure = [fx, fy, fz]``;
    ``bool`` returns ``[(||calc_pressure|| > threshold)]`` -- a single bit per
    finger, matching the per-finger collapse on the sim side.
    """
    extra = getattr(deploy_state, "extra", None)
    if not isinstance(extra, dict):
        return {}
    fingertip_sensors = extra.get("fingertip_sensors")
    if not isinstance(fingertip_sensors, dict):
        return {}
    out: dict[int, np.ndarray] = {}
    for finger_id, reading in fingertip_sensors.items():
        idx = _REAL_ID_TO_INDEX.get(int(finger_id))
        if idx is None or not isinstance(reading, dict) or "calc_pressure" not in reading:
            continue
        calc_pressure = np.asarray(reading["calc_pressure"], dtype=np.float64).reshape(3)
        if sensor_type == "bool":
            magnitude = float(np.linalg.norm(calc_pressure))
            out[idx] = np.array([float(magnitude > threshold)], dtype=np.float64)
        else:
            out[idx] = calc_pressure
    return out


# ================================== recorder ===================================


class TactileComparisonRecorder:
    """Owns the CSV file and the in-memory sample buffer feeding the live plot."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        finger_names: Sequence[str] = FINGER_NAMES,
        filename: str = "tactile_comparison.csv",
        record_action: bool = False,
        num_actions: int | None = None,
        display_fingers: Sequence[int] | None = None,
        channels: Sequence[str] = _AXES,
    ) -> None:
        self.finger_names = tuple(finger_names)
        self.csv_path = Path(log_dir) / filename
        self._record_action = record_action and num_actions is not None
        self._num_actions = int(num_actions) if self._record_action else 0
        self._file: Any = None
        self._writer: Any = None
        self._rows_since_flush = 0
        self.n_rows = 0
        # Per-finger channel layout (e.g. ("fx","fy","fz") for agg_force, ("bit",) for bool).
        # Drives CSV column count, plot line count, and the per-channel line styling.
        self.channels: tuple[str, ...] = tuple(channels)
        # Fingers shown in the live plot. The CSV always records all fingers;
        # ``display_fingers`` only narrows the plot (e.g. drop frozen/unsensored
        # finger links). ``None`` -> show every finger.
        if display_fingers is None:
            display_idx = list(range(len(self.finger_names)))
        else:
            display_idx = sorted({int(i) for i in display_fingers if 0 <= int(i) < len(self.finger_names)})
        self.display_indices: list[int] = display_idx or list(range(len(self.finger_names)))
        self.display_names: list[str] = [self.finger_names[i] for i in self.display_indices]
        # Canonical finger index to show alone, or None for every display finger.
        self.isolated_finger: int | None = None
        # Plot buffer: one zeroed (sim + real) tuple per finger so the plotter has a
        # valid shape before the first record() call.
        self._n_lines: int = 2 * len(self.channels)
        self._latest: dict[str, tuple[float, ...]] = {name: (0.0,) * self._n_lines for name in self.finger_names}

    # -- csv ---------------------------------------------------------------

    def _header(self) -> list[str]:
        cols = ["t"]
        if self._record_action:
            cols += [f"action_{i}" for i in range(self._num_actions)]
        for name in self.finger_names:
            cols += [f"sim_{name}_{ch}" for ch in self.channels]
            cols += [f"real_{name}_{ch}" for ch in self.channels]
        return cols

    def _open(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._header())

    def record(
        self,
        t: float,
        *,
        sim: dict[int, np.ndarray] | None,
        real: dict[int, np.ndarray] | None,
        action: np.ndarray | None = None,
    ) -> None:
        """Append one CSV row and update the live-plot sample.

        ``sim`` / ``real`` map canonical finger index -> per-channel array (length
        ``len(self.channels)``); missing fingers are written as ``nan``.
        """
        if self._file is None:
            self._open()

        sim = sim or {}
        real = real or {}
        nan_ch = (float("nan"),) * len(self.channels)

        row: list[float] = [float(t)]
        if self._record_action:
            act = np.full(self._num_actions, float("nan")) if action is None else np.asarray(action).reshape(-1)
            row += [float(v) for v in act[: self._num_actions]]

        plot_sample: dict[str, tuple[float, ...]] = {}
        for idx, name in enumerate(self.finger_names):
            sim_v = tuple(float(v) for v in sim[idx]) if idx in sim else nan_ch
            real_v = tuple(float(v) for v in real[idx]) if idx in real else nan_ch
            row += [*sim_v, *real_v]
            # The live plotter derives axis limits via min/max and rejects NaN/Inf,
            # so the plot buffer substitutes 0.0 for absent values; the CSV keeps nan.
            plot_sample[name] = tuple(
                0.0 if (v != v or v in (float("inf"), float("-inf"))) else v for v in (*sim_v, *real_v)
            )

        self._writer.writerow(row)
        self._latest = plot_sample
        self.n_rows += 1
        self._rows_since_flush += 1
        if self._rows_since_flush >= 50:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    # -- plotting ----------------------------------------------------------

    @property
    def plot_labels(self) -> dict[str, list[str]]:
        """``MPLLinePlot`` ``labels=`` dict: one subplot per display finger, ``2 * n_channels`` lines each."""
        return {
            name: [f"sim_{ch}" for ch in self.channels] + [f"real_{ch}" for ch in self.channels]
            for name in self.display_names
        }

    def plot_data(self) -> dict[str, tuple[float, ...]]:
        """Data func for ``MPLLinePlot`` (matches :attr:`plot_labels`)."""
        return {name: self._latest[name] for name in self.display_names}


# =============================== plot wiring ===================================

# Per-channel line style: fx/fy/fz share a hue; sim is light + dashed, real is
# dark + solid ("the real trace is the solid one"). Applied to every subplot, so
# colors stay consistent across fingers and runs. The ``bit`` entries cover the
# single-channel ``bool`` plot: real-hand trace is a solid step, sim is a dashed
# step at half opacity so an overlap is still readable.
_TACTILE_LINE_STYLE: dict[str, tuple[str, str]] = {
    "sim_fx": ("lightcoral", "--"),
    "sim_fy": ("yellowgreen", "--"),
    "sim_fz": ("deepskyblue", "--"),
    "real_fx": ("firebrick", "-"),
    "real_fy": ("darkgreen", "-"),
    "real_fz": ("navy", "-"),
    "sim_bit": ("lightsteelblue", "--"),
    "real_bit": ("navy", "-"),
}


class TactileLinePlot(_MPLLinePlotOptions):
    """Options marker selecting :class:`TactileLinePlotter`.

    Same fields as ``MPLLinePlot``; the distinct type is what routes
    ``scene.start_recording`` to the custom plotter via the recorder registry.
    """


@_register_recording(TactileLinePlot)
class TactileLinePlotter(_MPLLinePlotter):
    """``MPLLinePlotter`` with fixed per-channel colors and a matching legend.

    The stock plotter colors lines from a shared global cycle (so colors drift
    when other plots are open) and emits one legend entry per line across every
    subplot. This subclass recolors each line from :data:`_TACTILE_LINE_STYLE`
    after the base ``build()``, then rebuilds the legend from one subplot's
    (recolored) lines -- the swatches are taken straight from those lines, so the
    legend always matches the curves. Genesis itself is left untouched; this is a
    plain subclass registered through the public ``register_recording`` hook.
    """

    def build(self) -> None:
        super().build()
        for subplot_lines in self.lines.values():
            for line in subplot_lines:
                style = _TACTILE_LINE_STYLE.get(line.get_label())
                if style is not None:
                    color, linestyle = style
                    line.set_color(color)
                    line.set_linestyle(linestyle)
        # Labels repeat across subplots; rebuild a single legend from one
        # subplot's recolored lines so swatches match and it is not 30 entries.
        for legend in list(self.fig.legends):
            legend.remove()
        handles = next(iter(self.lines.values()), [])
        if handles:
            self.fig.legend(
                handles=handles,
                labels=[h.get_label() for h in handles],
                ncol=len(handles),
                loc="outside lower center",
            )
        self.fig.canvas.draw()
        # The legend resize can shift axis bboxes; refresh the blit backgrounds.
        self.caches_bbox = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]


def start_tactile_plot(
    scene: Any,
    recorder: TactileComparisonRecorder,
    *,
    title: str,
    history_length: int = 500,
) -> Any:
    """Start a live tactile line plot (one subplot per display finger).

    Each subplot draws six lines (``sim_fx/fy/fz``, ``real_fx/fy/fz``) with fixed
    colors via :class:`TactileLinePlotter`. Returns the plotter, or ``None`` if
    plotting is unavailable -- CSV recording still works in that case.
    """
    try:
        from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
    except Exception:  # noqa: BLE001 - plotting is optional; CSV still works.
        return None
    if not IS_MATPLOTLIB_AVAILABLE:
        return None

    try:
        return scene.start_recording(
            recorder.plot_data,
            TactileLinePlot(
                title=title,
                labels=recorder.plot_labels,
                x_label="step",
                y_label="force [fx/fy/fz]",
                history_length=history_length,
            ),
        )
    except Exception:  # noqa: BLE001 - keep the caller alive if plotting fails.
        return None


def reset_plotter(plotter: Any) -> None:
    """Fully reset a live MPL line plotter so it restarts cleanly after an env reset.

    Clears the data buffers AND rescales the axes. ``MPLLinePlotter._update_plot``
    only ever *extends* its x/y limits, never shrinks them, so just clearing the
    buffer leaves the window stuck at the pre-reset range with the fresh data
    drawn off-screen. Must be called from the loop thread; holds the plotter lock
    so it does not race the redraw. Works for any ``MPLLinePlotter`` (the joint
    plot) or :class:`TactileLinePlotter`.
    """
    if plotter is None:
        return
    line_plot = getattr(plotter, "line_plot", None)
    axes = getattr(plotter, "axes", None)
    lock = getattr(plotter, "_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        if line_plot is not None and hasattr(line_plot, "clear_data"):
            line_plot.clear_data()
        # Collapse the limits; the next _update_plot re-extends them from fresh data.
        for ax in axes:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-1e-6, 1e-6)
        if hasattr(plotter, "cache_xmax"):
            plotter.cache_xmax = -1
        fig = getattr(plotter, "fig", None)
        if fig is not None:
            fig.canvas.draw()
            if axes:
                plotter.caches_bbox = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
    except Exception:  # noqa: BLE001 - resetting the view is best-effort.
        pass
    finally:
        if lock is not None:
            lock.release()


def apply_isolation(plotter: Any, recorder: TactileComparisonRecorder) -> None:
    """Show only ``recorder.isolated_finger``'s subplot (or all when it is None).

    Toggles per-subplot axis visibility; with matplotlib's constrained layout the
    remaining visible subplot expands to fill the figure. Best-effort: must be
    called from the loop thread (not the plotter thread) and holds the plotter
    lock so it does not race the redraw.
    """
    if plotter is None or not getattr(plotter, "axes", None):
        return
    keys = list(recorder.plot_labels.keys())  # subplot order == display order
    iso = recorder.isolated_finger
    lock = getattr(plotter, "_lock", None)
    try:
        if lock is not None:
            lock.acquire()
        for ax, key in zip(plotter.axes, keys):
            ax.set_visible(iso is None or (key in FINGER_NAMES and FINGER_NAMES.index(key) == iso))
        if hasattr(plotter, "fig"):
            plotter.fig.canvas.draw_idle()
    except Exception:  # noqa: BLE001 - isolation is cosmetic; keep the caller alive.
        pass
    finally:
        if lock is not None:
            lock.release()
