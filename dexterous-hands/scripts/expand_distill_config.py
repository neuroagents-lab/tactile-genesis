#!/usr/bin/env python
"""Expand one entry from conf/distill_configs.yaml into bash exports.

The YAML carries shared ``defaults`` (the menu of available
resolutions/types/noisy modes/tactile encoders, plus a pinned baseline) and a
``configs:`` map keyed by sweep name (e.g. ``screwdriver-xhand1``). This
script merges defaults with the selected entry, then picks a subset of the
menu based on ``<mode>``:

* ``mode == "types"``: every type at the pinned baseline (clean / pinned
  resolution), prefixed with ``none``. The tactile encoder is auto-picked
  per sensor (grid sensors -> ``tactile_convrnn``; gridless/``none`` ->
  ``rnn``).
* ``mode == "models"``: every tactile_encoder against ``models_sensor_types``
  at the pinned baseline. No ``none`` (encoder is irrelevant when there is
  no tactile signal).
* ``mode == <sensor_type>``: that one type across every resolution and both
  noisy modes, prefixed with ``none``. Encoder auto-picked per sensor.

The script emits **parallel** bash arrays ``SENSORS`` and ``TACTILE_ENCODERS``
(one entry per slurm job). ``submit_distill.sh`` iterates them and assembles
``--run_name=<base>-<tactile_encoder>``; main.py's ``build_run_name`` then
appends task/robot/sensors/config, so the encoder is the only thing the
caller adds to the prefix -- no redundancy with what main.py auto-appends.

Usage:
    eval "$(python scripts/expand_distill_config.py \\
        conf/distill_configs.yaml screwdriver-xhand1 types)"
    eval "$(python scripts/expand_distill_config.py \\
        conf/distill_configs.yaml in_palm_rotate-xhand1 models)"
    eval "$(python scripts/expand_distill_config.py \\
        conf/distill_configs.yaml screwdriver-xhand1 force_torque)"
"""

from __future__ import annotations

import shlex
import sys

import yaml

MODE_TYPES = "types"
MODE_MODELS = "models"


def auto_tactile_encoder(sensor: str) -> str:
    """Pick the default tactile encoder for a sensor name.

    Gridless sensors (`none`, `agg_force`, `agg_bool`, `link_*`) get an `rnn`
    over the flat per-step vector; everything else gets the grid-aware
    `tactile_convrnn`. Mirror this in any consumer that needs to know the
    encoder for an auto-derived run -- the bash submit script no longer carries
    this logic.
    """
    if sensor == "none":
        return "rnn"
    tail = sensor.split("/", 1)[-1]
    if tail.startswith("agg_force") or tail.startswith("agg_bool") or tail.startswith("link_"):
        return "rnn"
    return "tactile_convrnn"


def expand_baseline_sensors(
    placements: list[str],
    sensor_types: list[str],
    pinned_resolution: str,
    pinned_noisy: bool,
    include_none: bool,
) -> list[str]:
    """Sensors at the pinned baseline (one resolution, one noisy mode)."""
    sensors: list[str] = ["none"] if include_none else []
    suffix = "/noisy" if pinned_noisy else ""
    for placement in placements:
        placement_str = f"{pinned_resolution}-{placement}"
        for stype in sensor_types:
            sensors.append(f"{placement_str}/{stype}{suffix}")
    return sensors


def expand_focused_sensors(
    placements: list[str],
    resolutions: list[str],
    noisy_modes: list[bool],
    sensor_type: str,
) -> list[str]:
    """One sensor type across resolutions x noisy_modes (prefixed with `none`)."""
    sensors: list[str] = ["none"]
    for resolution in resolutions:
        for placement in placements:
            placement_str = f"{resolution}-{placement}"
            for noisy in noisy_modes:
                tag = f"{placement_str}/{sensor_type}"
                if noisy:
                    tag += "/noisy"
                sensors.append(tag)
    return sensors


def jobs_for_mode(cfg: dict, mode: str) -> list[tuple[str, str]]:
    """Return the list of (sensor, tactile_encoder) pairs to submit."""
    placements: list[str] = cfg["sensor_placements"]
    sensor_resolutions: list[str] = cfg["sensor_resolutions"]
    sensor_types: list[str] = cfg["sensor_types"]
    noisy_modes: list[bool] = cfg["noisy_modes"]
    tactile_encoders: list[str] = cfg["tactile_encoders"]
    pinned_resolution: str = cfg["pinned_resolution"]
    pinned_noisy: bool = cfg["pinned_noisy"]
    models_sensor_types: list[str] = cfg["models_sensor_types"]

    if pinned_resolution not in sensor_resolutions:
        print(
            f"pinned_resolution={pinned_resolution!r} must be one of "
            f"sensor_resolutions={sensor_resolutions!r}.",
            file=sys.stderr,
        )
        sys.exit(2)
    if pinned_noisy not in noisy_modes:
        print(
            f"pinned_noisy={pinned_noisy!r} must be one of noisy_modes={noisy_modes!r}.",
            file=sys.stderr,
        )
        sys.exit(2)

    if mode == MODE_TYPES:
        sensors = expand_baseline_sensors(
            placements, sensor_types, pinned_resolution, pinned_noisy, include_none=True
        )
        return [(s, auto_tactile_encoder(s)) for s in sensors]

    if mode == MODE_MODELS:
        unknown = [t for t in models_sensor_types if t not in sensor_types]
        if unknown:
            print(
                f"models_sensor_types contains unknown type(s) {unknown!r}; "
                f"sensor_types menu is {sensor_types!r}.",
                file=sys.stderr,
            )
            sys.exit(2)
        sensors = expand_baseline_sensors(
            placements,
            models_sensor_types,
            pinned_resolution,
            pinned_noisy,
            include_none=False,
        )
        return [(s, enc) for s in sensors for enc in tactile_encoders]

    if mode not in sensor_types:
        valid = ", ".join([MODE_TYPES, MODE_MODELS, *sensor_types])
        print(f"Unknown mode {mode!r}. Use one of: {valid}.", file=sys.stderr)
        sys.exit(2)
    sensors = expand_focused_sensors(placements, sensor_resolutions, noisy_modes, mode)
    return [(s, auto_tactile_encoder(s)) for s in sensors]


def main(path: str, name: str, mode: str) -> None:
    with open(path) as f:
        doc = yaml.safe_load(f)

    configs = doc.get("configs") or {}
    if name not in configs:
        available = ", ".join(sorted(configs)) or "(none)"
        print(f"Unknown config {name!r}. Available: {available}", file=sys.stderr)
        sys.exit(2)

    cfg = {**(doc.get("defaults") or {}), **configs[name]}

    task = cfg["task"]
    robot = cfg["robot"]
    checkpoint = cfg["teacher_checkpoint"]
    extra_flags: list[str] = list(cfg.get("extra_flags") or [])
    temporal_reduction = cfg.get("temporal_reduction")

    jobs = jobs_for_mode(cfg, mode)
    sensors = [s for s, _ in jobs]
    encoders = [e for _, e in jobs]

    task_flags_parts = [
        f"--task={task}",
        f"--robot={robot}",
        f"--checkpoint={checkpoint}",
        *extra_flags,
    ]
    if temporal_reduction is not None:
        if temporal_reduction not in {"median", "none", "last"}:
            print(
                f"Invalid temporal_reduction {temporal_reduction!r}; "
                "expected one of median/none/last.",
                file=sys.stderr,
            )
            sys.exit(2)
        task_flags_parts.append(f"--temporal_reduction={temporal_reduction}")

    max_iters = cfg.get("max_iters")
    if max_iters is not None:
        if not isinstance(max_iters, int) or max_iters <= 0:
            print(
                f"Invalid max_iters {max_iters!r}; expected a positive integer.",
                file=sys.stderr,
            )
            sys.exit(2)
        task_flags_parts.append(f"--max_iters={max_iters}")

    task_flags = " ".join(task_flags_parts)

    print(f"TASK={shlex.quote(task)}")
    print(f"ROBOT={shlex.quote(robot)}")
    print(f"CHECKPOINT={shlex.quote(checkpoint)}")
    print(f"TASK_FLAGS={shlex.quote(task_flags)}")
    print(f"SENSORS=({' '.join(shlex.quote(s) for s in sensors)})")
    print(f"TACTILE_ENCODERS=({' '.join(shlex.quote(e) for e in encoders)})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "usage: expand_distill_config.py <configs.yaml> <name> "
            "<mode: types|models|sensor_type>",
            file=sys.stderr,
        )
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
