"""Shared model/encoder cfg registries selected by --tactile_encoder / --encoder.

``DexHandRslRlRunnerMod`` exposes two CLI flags that pick from named per-group
encoder configs:

- ``--tactile_encoder`` -> key of :data:`TACTILE_ENCODER_CFGS` (applied to the
  ``tactile_sensors`` obs group).
- ``--encoder`` -> key of :data:`GROUP_ENCODER_CFGS` (applied to the ``proprio``
  obs group, and reused for any other encoded groups by convention).

The grid-aware tactile encoders (``tactile_cnn`` / ``tactile_convrnn``) need a
``TactileLayout`` describing ``(num_sensors, grid_h, grid_w, features_per_probe,
history_length)``. The runner mod derives it from the live ``sensors_options``
at ``apply`` time (see ``task_mods.derive_tactile_layout``) and injects it into
the chosen tactile cfg, so a single registry entry covers every placement /
resolution / robot whose probes are stored as 2D grids.

Use ``--tactile_encoder=rnn`` (or ``mlp``) for sensor types that have no probe
grid -- ``agg_force`` and the ``link_*`` family aggregate per link, so there is
nothing to convolve over.
"""

from __future__ import annotations

from typing import Any

# ----------------------------------------------------------------------
# Tactile (`tactile_sensors` obs group) encoder cfgs.
#
# ``tactile_layout`` for the grid-aware kinds is injected at runtime by
# ``DexHandRslRlRunnerMod.apply`` -- do not bake it in here.
# ----------------------------------------------------------------------

TACTILE_ENCODER_CFGS: dict[str, dict[str, Any]] = {
    "mlp": {
        # Modest 2-layer MLP for the same-order-of-magnitude size comparison
        # against tactile_convrnn (input layer dominates: 5970 -> 64 -> 64 -> 32).
        "kind": "mlp",
        "hidden_dims": [64, 64],
        "output_dim": 32,
    },
    "rnn": {
        # Single-layer LSTM with hidden=64 (kept 1-layer like the original cfg).
        "kind": "rnn",
        "rnn_type": "lstm",
        "hidden_dim": 64,
        "num_layers": 1,
        "output_dim": 32,
    },
    "tactile_cnn": {
        "kind": "tactile_cnn",
        "output_dim": 32,
        "shared_per_sensor": False,
        "cnn": {"output_channels": [16, 32], "kernel_size": 3, "padding": "zeros"},
    },
    "tactile_convrnn": {
        "kind": "tactile_convrnn",
        "output_dim": 32,
        "shared_per_sensor": False,
        "convrnn": {"out_channels": 16, "ksize": 3, "layernorm": True},
    },
    "tactile_convrnn_lstm": {
        # Same per-sensor structure as tactile_convrnn but with a ConvLSTM cell
        # (pt_tnn LSTMCell). State carries packed (c, h) along the channel axis,
        # so the per-env state is 2x wider than the Intersection variant; the
        # projection still reads the cell's C_h-channel visible output.
        "kind": "tactile_convrnn",
        "output_dim": 32,
        "shared_per_sensor": False,
        "convrnn": {"out_channels": 16, "ksize": 3, "layernorm": True, "cell_type": "lstm"},
    },
    "tactile_convrnn_big": {
        # Per-sensor convrnn with wider hidden channels -- capacity control vs
        # tactile_convrnn at the same 32-d output.
        "kind": "tactile_convrnn",
        "output_dim": 32,
        "shared_per_sensor": False,
        "convrnn": {"out_channels": 48, "ksize": 3, "layernorm": True},
    },
    "tactile_canvas_cnn": {
        "kind": "tactile_canvas_cnn",
        "output_dim": 32,
        "cnn": {"output_channels": [64, 128], "kernel_size": 3, "padding": "zeros"},
    },
    "tactile_canvas_convrnn": {
        "kind": "tactile_canvas_convrnn",
        "output_dim": 32,
        "convrnn": {"out_channels": 64, "ksize": 3, "layernorm": True},
    },
}


# ----------------------------------------------------------------------
# Group encoder cfgs for non-tactile obs groups (proprio, ...).
# ----------------------------------------------------------------------

GROUP_ENCODER_CFGS: dict[str, dict[str, Any]] = {
    "mlp": {
        "kind": "mlp",
        "hidden_dims": [512, 256, 128],
        "output_dim": 64,
    },
    "rnn": {
        "kind": "rnn",
        "rnn_type": "lstm",
        "hidden_dim": 128,
        "num_layers": 1,
        "output_dim": 64,
    },
}
