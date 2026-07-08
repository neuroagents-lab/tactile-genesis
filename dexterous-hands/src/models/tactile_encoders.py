"""Per-sensor tactile encoders for student/teacher distillation.

These slice a flat ``tactile_sensors`` observation (the concatenation produced by
``TactileSensorRead``) back into per-finger grids and run a small CNN or
:class:`pt_tnn.recurrent_cells.IntersectionRNNCell` over each one. Output is a
``(B, output_dim)`` (inference) or ``(T, B, output_dim)`` (batched recurrent
training) latent that can drop into the ``PreEncode*`` model concat-then-head
pattern.

Flat layout convention (must match TactileSensorRead order):
    flat = (B, T * S * F * H * W)  with the innermost grouping being a single
    sensor's flattened (F, H, W) per timestep; sensors in TactileSensorsMod
    dict insertion order; oldest history frame first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import CNN
from rsl_rl.utils import unpad_trajectories


@dataclass(frozen=True)
class TactileLayout:
    """Per-sensor grid metadata derived from TactileSensorsMod + history.

    Each sensor (patch / link) carries its own ``(H_s, W_s)`` so heterogeneous
    placements (e.g. ``low-hand``: fingertips 3x3, palm 4x3, midfingers 2x3)
    fit alongside uniform ones. The grid-aware encoders run a per-sensor
    ConvRNN/CNN stack on each patch's native shape and concatenate the
    projected per-sensor latents.
    """

    num_sensors: int
    grid_hw: tuple[tuple[int, int], ...]
    features_per_probe: int
    history_length: int = 1

    def __post_init__(self) -> None:
        if len(self.grid_hw) != self.num_sensors:
            raise ValueError(
                f"TactileLayout.grid_hw has {len(self.grid_hw)} entries but num_sensors={self.num_sensors}."
            )

    @classmethod
    def from_dict(cls, d: dict) -> "TactileLayout":
        """Accept the old uniform-grid spec (``grid_h``/``grid_w``) or the new per-sensor ``grid_hw``."""
        d = dict(d)
        if "grid_hw" not in d:
            grid_h = d.pop("grid_h")
            grid_w = d.pop("grid_w")
            d["grid_hw"] = tuple((grid_h, grid_w) for _ in range(d["num_sensors"]))
        else:
            d["grid_hw"] = tuple(tuple(hw) for hw in d["grid_hw"])
        return cls(**d)

    @property
    def per_sensor_per_step_dim(self) -> tuple[int, ...]:
        """Flat dim per sensor at a single time step (``F * H_s * W_s``)."""
        return tuple(self.features_per_probe * h * w for h, w in self.grid_hw)

    @property
    def per_step_dim(self) -> int:
        return sum(self.per_sensor_per_step_dim)

    @property
    def flat_dim(self) -> int:
        return self.history_length * self.per_step_dim

    def slice_per_sensor(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Split ``x`` of shape ``(..., flat_dim)`` into per-sensor ``(..., T, F, H_s, W_s)`` tensors.

        Obs layout matches ``TactileSensorRead`` + ``ObservationHistoryLengthMod``:
        the inner step concatenates every sensor's flattened ``(F, H_s, W_s)``,
        and the obs history axis stacks those steps as the outermost flat group.
        """
        if x.shape[-1] != self.flat_dim:
            raise ValueError(
                f"Tactile input feature dim {x.shape[-1]} != layout flat_dim {self.flat_dim}."
            )
        lead = x.shape[:-1]
        per_step = x.reshape(*lead, self.history_length, self.per_step_dim)  # (..., T, per_step)
        parts = per_step.split(list(self.per_sensor_per_step_dim), dim=-1)
        return [
            part.reshape(*lead, self.history_length, self.features_per_probe, h, w)
            for part, (h, w) in zip(parts, self.grid_hw, strict=True)
        ]


def _split_output_dim(output_dim: int, num_sensors: int) -> list[int]:
    """Distribute ``output_dim`` across ``num_sensors`` slots, remainder front-loaded."""
    base = output_dim // num_sensors
    extra = output_dim - base * num_sensors
    return [base + (1 if i < extra else 0) for i in range(num_sensors)]


def _effective_ksize(cfg_ksize: int, h: int, w: int) -> int:
    """Cap ``cfg_ksize`` to a sensor's patch and round down to odd.

    The cfg's ``ksize`` is treated as the maximum kernel size; the per-sensor
    kernel shrinks to fit small patches and stays odd (so "same"-style padding
    works on every backend). 2x3 -> 1; 3x3 -> 3; 4x5 -> 3; 7x5 -> 3 or 5
    depending on cap.
    """
    eff = min(cfg_ksize, h, w)
    if eff % 2 == 0:
        eff -= 1
    return max(eff, 1)


class TactileCNNEncoder(nn.Module):
    """Per-sensor CNN encoder. Non-recurrent; ignores ``masks``/``hidden_state``.

    One ``rsl_rl.modules.cnn.CNN`` per sensor (optionally weight-shared across
    sensors). The per-sensor flat output is run through a per-sensor Linear
    sized to that sensor's own ``(H_s, W_s)``; latents are concatenated to give
    the final ``output_dim`` (split evenly across sensors).
    """

    is_recurrent = False
    state_slots = 0

    def __init__(
        self,
        layout: TactileLayout,
        cnn_cfg: dict[str, Any],
        output_dim: int,
        shared_per_sensor: bool = False,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.shared = shared_per_sensor

        # The cfg's kernel_size is a cap; each sensor's CNN gets a kernel
        # capped to its own patch size and rounded down to odd.
        cnn_cfg = dict(cnn_cfg)
        cfg_ksize = int(cnn_cfg.pop("kernel_size", 3))

        def _make_cnn(h: int, w: int) -> CNN:
            return CNN(
                input_dim=(h, w),
                input_channels=layout.features_per_probe,
                kernel_size=_effective_ksize(cfg_ksize, h, w),
                **cnn_cfg,
            )

        if shared_per_sensor:
            unique = set(layout.grid_hw)
            if len(unique) != 1:
                raise ValueError(
                    f"shared_per_sensor=True requires uniform grid_hw across sensors; got {sorted(unique)}."
                )
            base_cnn = _make_cnn(*layout.grid_hw[0])
            self.cnns = nn.ModuleList([base_cnn for _ in range(layout.num_sensors)])
        else:
            self.cnns = nn.ModuleList([_make_cnn(h, w) for (h, w) in layout.grid_hw])

        if any(cnn.output_channels is not None for cnn in self.cnns):
            raise ValueError("Tactile CNN must use flatten=True so its output is 1D.")

        per_sensor_dims = _split_output_dim(output_dim, layout.num_sensors)
        self._per_sensor_dims = per_sensor_dims
        # Per-sensor proj: each sensor's flat CNN output (over the whole obs-history)
        # projects to its slice of the final latent.
        self.projs = nn.ModuleList([
            nn.Linear(layout.history_length * int(cnn.output_dim), d_s)
            for cnn, d_s in zip(self.cnns, per_sensor_dims, strict=True)
        ])
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        del masks, hidden_state  # CNN is feed-forward
        lead = x.shape[:-1]
        per_sensor_grids = self.layout.slice_per_sensor(x)  # list of (..., T_hist, F, H_s, W_s)

        per_sensor_latents: list[torch.Tensor] = []
        for grid, cnn, proj in zip(per_sensor_grids, self.cnns, self.projs, strict=True):
            T_hist, F = grid.shape[-4], grid.shape[-3]
            H, W = grid.shape[-2], grid.shape[-1]
            flat_in = grid.reshape(-1, F, H, W)
            cnn_out = cnn(flat_in)                    # (N*T_hist, cnn_flat)
            per_sensor_latents.append(
                proj(cnn_out.reshape(*lead, T_hist * cnn_out.shape[-1]))
            )
        return torch.cat(per_sensor_latents, dim=-1)

    # No-op recurrent hooks so the parent model can call uniformly.
    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_hidden_state(self) -> None:
        return None

    def set_hidden_state(self, hidden_state: Any) -> None:
        del hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_state_tuple(self) -> tuple[torch.Tensor, ...]:
        return ()

    def set_state_tuple(self, tup: tuple[torch.Tensor, ...]) -> None:
        del tup


def _resolve_convrnn_cell(cell_type: str) -> tuple[type, int]:
    """Return ``(cell_cls, state_channel_mult)`` for a per-sensor ConvRNN cell.

    ``state_channel_mult`` is the per-env-state channel count divided by the
    cell's visible output channel count: 1 for Intersection/GRU, 2 for pt_tnn's
    LSTM (which packs ``(c, h)`` along the channel axis).
    """
    from pt_tnn.recurrent_cells import GRUCell, IntersectionRNNCell, LSTMCell  # local import: optional dep

    table = {
        "intersection": (IntersectionRNNCell, 1),
        "gru": (GRUCell, 1),
        "lstm": (LSTMCell, 2),
    }
    key = cell_type.lower()
    if key not in table:
        raise ValueError(f"Unknown convrnn cell_type {cell_type!r}; expected one of {sorted(table)}.")
    return table[key]


class TactileConvRNNEncoder(nn.Module):
    """Per-sensor ConvRNN encoder.

    Cell kind is configurable via ``cell_type`` in ``convrnn_cfg``: ``intersection``
    (default, ``IntersectionRNNCell``), ``gru``, or ``lstm``. The LSTM variant
    uses pt_tnn's ``LSTMCell``, whose per-env state packs ``(c, h)`` along the
    channel axis -- so its per-sensor state is ``(N, 2*C_h, H_s, W_s)`` while
    Intersection/GRU keep ``(N, C_h, H_s, W_s)``. The encoder projects from the
    cell's visible output (always ``C_h`` channels) so the per-sensor ``Linear``
    head is identical across cell types and ``shared_per_sensor`` weight-sharing
    works the same way.

    One cell per sensor (optionally weight-shared across sensors when all
    grids are uniform), each carrying its own per-env state at that sensor's
    native ``(H_s, W_s)``. Per-sensor projections produce slices of the final
    latent which are concatenated to ``output_dim``.

    Temporal memory is carried by the cell's per-env state across env steps
    (mirrors ``rsl_rl.modules.rnn.RNN`` state lifecycle). Observation history
    on this group is ignored beyond the most recent frame -- the cell IS the
    temporal model. For batched recurrent training (``masks is not None``)
    the encoder loops over the sequence dim with the provided init state.

    State on the wire: a single 3D tensor shaped
    ``(1, num_envs, sum_s(state_mult * C_h * H_s * W_s))`` -- per-sensor states
    are concatenated along the feature axis. Internally the encoder keeps them
    as a list of per-sensor ``(N, state_mult * C_h, H_s, W_s)`` tensors and
    packs/unpacks on the storage boundary, so ``state_slots == 1`` regardless
    of cell type or grid heterogeneity.
    """

    is_recurrent = True
    state_slots = 1

    def __init__(
        self,
        layout: TactileLayout,
        convrnn_cfg: dict[str, Any],
        output_dim: int,
        shared_per_sensor: bool = False,
    ) -> None:
        super().__init__()

        if layout.history_length != 1:
            raise ValueError(
                "TactileConvRNNEncoder requires layout.history_length == 1: the cell "
                "carries temporal memory across env steps via its persistent state, so "
                "obs-stacked history would be silently discarded. Set the obs group's "
                f"history_length to 1 (got {layout.history_length}) -- e.g. drop the "
                "tactile_sensors entry from ObservationHistoryLengthMod."
            )

        self.layout = layout
        self.shared = shared_per_sensor

        cfg = dict(convrnn_cfg)
        if "out_channels" not in cfg:
            raise ValueError("convrnn config must include 'out_channels'.")
        out_channels = int(cfg.pop("out_channels"))
        cfg_ksize = int(cfg.pop("ksize", 3))   # cap; per-sensor ksize derived below
        cfg.setdefault("layernorm", True)
        cell_type = str(cfg.pop("cell_type", "intersection"))
        cell_cls, state_channel_mult = _resolve_convrnn_cell(cell_type)
        self._cell_type = cell_type.lower()
        self._state_channel_mult = state_channel_mult

        # Cells concat input+state on the channel axis and require equal channel
        # counts. F -> C_h per-sensor with a 1x1 conv so the cell runs in C_h
        # space throughout. 1x1 conv weights are shape-invariant, so sharing is
        # safe even for heterogeneous grids.
        def _make_proj() -> nn.Module:
            return nn.Conv2d(layout.features_per_probe, out_channels, kernel_size=1, bias=True)

        def _make_cell(h: int, w: int) -> nn.Module:
            return cell_cls(
                input_in_channels=out_channels,
                out_channels=out_channels,
                ksize=_effective_ksize(cfg_ksize, h, w),
                **cfg,
            )

        if shared_per_sensor:
            unique = set(layout.grid_hw)
            if len(unique) != 1:
                raise ValueError(
                    f"shared_per_sensor=True requires uniform grid_hw across sensors; got {sorted(unique)}."
                )
            base_proj, base_cell = _make_proj(), _make_cell(*layout.grid_hw[0])
            self.input_projs = nn.ModuleList([base_proj for _ in range(layout.num_sensors)])
            self.cells = nn.ModuleList([base_cell for _ in range(layout.num_sensors)])
        else:
            self.input_projs = nn.ModuleList([_make_proj() for _ in range(layout.num_sensors)])
            self.cells = nn.ModuleList([_make_cell(h, w) for (h, w) in layout.grid_hw])

        self._out_channels = out_channels

        per_sensor_dims = _split_output_dim(output_dim, layout.num_sensors)
        self._per_sensor_dims = per_sensor_dims
        self.projs = nn.ModuleList([
            nn.Linear(out_channels * h * w, d_s)
            for (h, w), d_s in zip(layout.grid_hw, per_sensor_dims, strict=True)
        ])
        self._output_dim = output_dim

        # Persistent per-env state, list of per-sensor tensors (N, C_h, H_s, W_s).
        # None entries mean "not initialised yet for this num_envs".
        self._state: list[torch.Tensor] | None = None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def _state_channels(self) -> int:
        return self._state_channel_mult * self._out_channels

    @property
    def _per_sensor_state_dims(self) -> list[int]:
        return [self._state_channels * h * w for (h, w) in self.layout.grid_hw]

    @property
    def _total_state_dim(self) -> int:
        return sum(self._per_sensor_state_dims)

    # ---- state lifecycle (mirrors rsl_rl.modules.rnn.RNN) ----------------

    def _zero_state(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(num_envs, self._state_channels, h, w, device=device, dtype=dtype)
            for (h, w) in self.layout.grid_hw
        ]

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self._state = None
            return
        if self._state is None:
            return
        idx = (dones == 1)
        for s_state in self._state:
            s_state[idx] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._state is None:
            return
        if dones is None:
            self._state = [s.detach() for s in self._state]
        else:
            mask = (dones == 1)
            for s_state in self._state:
                s_state[mask] = s_state[mask].detach()

    def get_hidden_state(self) -> torch.Tensor | None:
        """Pack per-sensor states into a single ``(1, num_envs, total_state_dim)`` tensor."""
        if self._state is None:
            return None
        n = self._state[0].shape[0]
        packed = torch.cat([s.reshape(n, -1) for s in self._state], dim=-1)
        return packed.unsqueeze(0)

    def set_hidden_state(self, hidden_state: Any) -> None:
        if hidden_state is None:
            self._state = None
            return
        if isinstance(hidden_state, (tuple, list)):
            if len(hidden_state) != 1:
                raise ValueError(
                    f"ConvRNN expects exactly 1 state tensor on the wire, got {len(hidden_state)}."
                )
            hidden_state = hidden_state[0]
        if not isinstance(hidden_state, torch.Tensor):
            raise TypeError(f"Unexpected ConvRNN hidden_state type: {type(hidden_state).__name__}")
        if hidden_state.dim() == 3:
            assert hidden_state.shape[0] == 1, "ConvRNN expects a single 'layer' of state."
            n = hidden_state.shape[1]
            flat = hidden_state.squeeze(0)
            if flat.shape[-1] != self._total_state_dim:
                raise ValueError(
                    f"ConvRNN packed state dim {flat.shape[-1]} != expected {self._total_state_dim}."
                )
            parts = flat.split(self._per_sensor_state_dims, dim=-1)
            self._state = [
                part.reshape(n, self._state_channels, h, w)
                for part, (h, w) in zip(parts, self.layout.grid_hw, strict=True)
            ]
        else:
            raise ValueError(f"Unexpected ConvRNN hidden_state rank: {hidden_state.dim()}")

    def get_state_tuple(self) -> tuple[torch.Tensor, ...]:
        h = self.get_hidden_state()
        return () if h is None else (h,)

    def set_state_tuple(self, tup: tuple[torch.Tensor, ...]) -> None:
        if not tup:
            self.set_hidden_state(None)
            return
        assert len(tup) == 1, "TactileConvRNNEncoder expects exactly 1 state slot."
        self.set_hidden_state(tup[0])

    # ---- forward --------------------------------------------------------

    def _step(
        self, frames: list[torch.Tensor], state: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One temporal step.

        frames[s]: ``(N, F, H_s, W_s)``; state[s]: ``(N, state_channels, H_s, W_s)``.
        Returns ``(outputs, new_state)`` where outputs[s] is the cell's visible
        output ``(N, C_h, H_s, W_s)`` and new_state[s] has the cell's native
        per-env-state channel count (1x C_h for Intersection/GRU, 2x for LSTM).
        """
        outputs: list[torch.Tensor] = []
        new_state: list[torch.Tensor] = []
        for s, (frame, st) in enumerate(zip(frames, state, strict=True)):
            x = self.input_projs[s](frame)               # F -> C_h
            out, ns = self.cells[s](x, st)
            outputs.append(out)
            new_state.append(ns)
        return outputs, new_state

    def _project(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Per-sensor (N, C_h, H_s, W_s) output -> (N, d_s) -> concat (N, output_dim)."""
        outs = [
            proj(o.reshape(o.shape[0], -1))
            for o, proj in zip(outputs, self.projs, strict=True)
        ]
        return torch.cat(outs, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        batch_mode = masks is not None
        per_sensor_grids = self.layout.slice_per_sensor(x)  # list of (..., T_hist, F, H_s, W_s)

        if batch_mode:
            # x: (T, B, flat_dim). Each per_sensor_grids[s] is (T, B, T_hist, F, H_s, W_s).
            T, B = x.shape[0], x.shape[1]
            # Only the most recent obs-history frame matters; the cell carries memory.
            frames_per_t = [g[:, :, -1] for g in per_sensor_grids]   # each (T, B, F, H_s, W_s)
            if hidden_state is None:
                state = self._zero_state(B, x.device, x.dtype)
            else:
                # Stash the packed wire-state into self._state for reuse.
                self.set_hidden_state(hidden_state)
                state = self._state if self._state is not None else self._zero_state(B, x.device, x.dtype)

            outs = []
            for t in range(T):
                step_outputs, state = self._step(
                    [frames_per_t[s][t] for s in range(self.layout.num_sensors)], state
                )
                outs.append(self._project(step_outputs))
            out = torch.stack(outs, dim=0)  # (T, B, output_dim)
            return unpad_trajectories(out, masks)

        # Inference mode: x: (N, flat_dim). Each per_sensor_grids[s]: (N, T_hist, F, H_s, W_s).
        N = x.shape[0]
        frames = [g[:, -1] for g in per_sensor_grids]
        if self._state is None or self._state[0].shape[0] != N:
            self._state = self._zero_state(N, x.device, x.dtype)
        step_outputs, self._state = self._step(frames, self._state)
        return self._project(step_outputs)


class GroupRNNEncoder(nn.Module):
    """Per-group LSTM/GRU encoder that wraps :class:`rsl_rl.modules.rnn.RNN`.

    Use this for arbitrary 1D observation groups (e.g. ``proprio``) when you
    want a small recurrent encoder ahead of the policy head. Its state is
    persistent across env steps and round-trips through rsl_rl's storage the
    same way the top-level policy RNN does: LSTM exposes 2 state tensors
    ``(h, c)``; GRU exposes 1.

    Output shape per the encoder contract:
      - inference: ``(B, output_dim)``
      - batched (``masks`` given): unpadded ``(?, B, output_dim)``
    """

    is_recurrent = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        from rsl_rl.modules import RNN  # local to keep top-level imports light

        self._rnn = RNN(input_dim, hidden_dim, num_layers, rnn_type)
        self._rnn_type = rnn_type.lower()
        self.state_slots = 2 if self._rnn_type == "lstm" else 1
        self.proj = nn.Linear(hidden_dim, output_dim)
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        if masks is not None and hidden_state is None:
            # rsl_rl.RNN's batch-mode forward raises if hidden_state is None; for
            # parity with TactileConvRNNEncoder (and to support fresh-start training
            # rollouts whose first batch has no saved state), zero-init here.
            out, _ = self._rnn.rnn(x, None)  # type: ignore[arg-type]
            out = unpad_trajectories(out, masks)
        else:
            out = self._rnn(x, masks, hidden_state)
            if masks is None:
                # rsl_rl.RNN inference returns (1, B, hidden_dim); collapse leading dim.
                out = out.squeeze(0)
        return self.proj(out)

    # ---- state lifecycle ---------------------------------------------
    def reset(self, dones: torch.Tensor | None = None) -> None:
        self._rnn.reset(dones)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self._rnn.detach_hidden_state(dones)

    def get_hidden_state(self) -> Any:
        return self._rnn.hidden_state

    def set_hidden_state(self, hidden_state: Any) -> None:
        self._rnn.hidden_state = hidden_state

    def get_state_tuple(self) -> tuple[torch.Tensor, ...]:
        h = self._rnn.hidden_state
        if h is None:
            return ()
        if isinstance(h, tuple):
            return tuple(h)
        return (h,)

    def set_state_tuple(self, tup: tuple[torch.Tensor, ...]) -> None:
        if not tup:
            self._rnn.hidden_state = None
            return
        if self._rnn_type == "lstm":
            assert len(tup) == 2, f"LSTM GroupRNNEncoder expects 2 state slots, got {len(tup)}."
            self._rnn.hidden_state = (tup[0], tup[1])
        else:
            assert len(tup) == 1, f"GRU GroupRNNEncoder expects 1 state slot, got {len(tup)}."
            self._rnn.hidden_state = tup[0]


# ---------------------------------------------------------------------------
# Canvas-packed tactile encoders.
#
# Instead of running an independent CNN/ConvRNN per sensor, pack all per-sensor
# grids into a single 2D canvas with zero padding (and a presence-mask channel)
# and run one CNN/ConvRNN over the whole canvas. The canvas layout is read from
# a JSON asset (e.g. ``src/assets/sensors/xhand1/med/canvas_xhand1.json``) and
# wraps the per-sensor TactileLayout, so the existing per-sensor slicing of the
# flat obs is reused unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanvasTactileLayout:
    """Per-sensor TactileLayout + a 2D canvas mapping every sensor to a patch.

    ``placements[i] == (row, col, h, w)`` is where sensor ``i``'s patch sits on
    the canvas, in the same order as ``base.grid_hw`` (which mirrors
    ``sensors_options`` iteration order at layout-derivation time).

    The canvas patch ``(h, w)`` must equal ``base.grid_hw[i]`` or its transpose
    (90° rotation, e.g. to keep finger length along the canvas row axis). The
    transpose flag per sensor is derived in ``__post_init__`` and applied
    automatically by :meth:`scatter_frames`.
    """

    base: TactileLayout
    canvas_hw: tuple[int, int]
    placements: tuple[tuple[int, int, int, int], ...]
    canvas_name: str = ""
    link_names: tuple[str, ...] = field(default_factory=tuple)
    transposed: tuple[bool, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        if len(self.placements) != self.base.num_sensors:
            raise ValueError(
                f"CanvasTactileLayout: {len(self.placements)} placements vs base.num_sensors={self.base.num_sensors}."
            )
        H, W = self.canvas_hw
        occ = [[False] * W for _ in range(H)]
        derived_transposed: list[bool] = []
        for i, (r, c, h, w) in enumerate(self.placements):
            if r < 0 or c < 0 or r + h > H or c + w > W:
                raise ValueError(
                    f"CanvasTactileLayout: sensor {i} placement ({r},{c},{h},{w}) is out of canvas {H}x{W}."
                )
            sh, sw = self.base.grid_hw[i]
            if (h, w) == (sh, sw):
                derived_transposed.append(False)
            elif (h, w) == (sw, sh):
                derived_transposed.append(True)
            else:
                raise ValueError(
                    f"CanvasTactileLayout: sensor {i} placement (h,w)=({h},{w}) does not match base "
                    f"grid {sh}x{sw} or its transpose {sw}x{sh}."
                )
            for rr in range(r, r + h):
                for cc in range(c, c + w):
                    if occ[rr][cc]:
                        raise ValueError(
                            f"CanvasTactileLayout: sensor {i} placement overlaps another sensor at ({rr},{cc})."
                        )
                    occ[rr][cc] = True
        object.__setattr__(self, "transposed", tuple(derived_transposed))

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasTactileLayout":
        base_keys = ("num_sensors", "grid_hw", "features_per_probe", "history_length")
        base = TactileLayout.from_dict({k: d[k] for k in base_keys if k in d})
        canvas_hw = tuple(d["canvas_hw"])
        if len(canvas_hw) != 2:
            raise ValueError(f"CanvasTactileLayout: canvas_hw must have 2 entries, got {canvas_hw}.")
        placements = tuple(tuple(int(v) for v in p) for p in d["placements"])
        link_names = tuple(d.get("link_names", ()))
        return cls(
            base=base,
            canvas_hw=(int(canvas_hw[0]), int(canvas_hw[1])),
            placements=placements,
            canvas_name=str(d.get("canvas_name", "")),
            link_names=link_names,
        )

    @property
    def flat_dim(self) -> int:
        return self.base.flat_dim

    def build_presence_mask(self) -> torch.Tensor:
        """``(1, H_canvas, W_canvas)`` with 1.0 at real sensor cells, 0.0 elsewhere."""
        H, W = self.canvas_hw
        m = torch.zeros(1, H, W)
        for r, c, h, w in self.placements:
            m[0, r:r + h, c:c + w] = 1.0
        return m

    def scatter_frames(self, per_sensor_frames: list[torch.Tensor]) -> torch.Tensor:
        """Place per-sensor frames onto a single canvas, rotating where needed.

        Each ``per_sensor_frames[i]`` has shape ``(N, F, H_s, W_s)`` (matching
        ``base.grid_hw[i]``). If ``self.transposed[i]`` is True, the frame is
        ``.transpose(-1, -2)``'d before being written into the canvas patch.
        Returns ``(N, F, H_canvas, W_canvas)``.
        """
        if not per_sensor_frames:
            raise ValueError("CanvasTactileLayout.scatter_frames: empty frames.")
        return _scatter_to_canvas(per_sensor_frames, self.placements, self.canvas_hw, self.transposed)


def _scatter_to_canvas(
    per_sensor_frames: list[torch.Tensor],
    placements: tuple[tuple[int, int, int, int], ...],
    canvas_hw: tuple[int, int],
    transposed: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Scatter per-sensor ``(N, F, H_s, W_s)`` frames into one ``(N, F, H_canvas, W_canvas)``.

    If ``transposed[i]`` is True the i'th frame is ``.transpose(-1, -2)``'d before
    being written -- callers that pre-validated placements against the layout's
    :attr:`transposed` should pass that tuple through here.
    """
    if not per_sensor_frames:
        raise ValueError("_scatter_to_canvas: empty frames.")
    if transposed is None:
        transposed = (False,) * len(per_sensor_frames)
    ref = per_sensor_frames[0]
    N = ref.shape[0]
    F_in = ref.shape[-3]
    H, W = canvas_hw
    canvas = torch.zeros(N, F_in, H, W, device=ref.device, dtype=ref.dtype)
    for frame, (r, c, h, w), tflag in zip(per_sensor_frames, placements, transposed, strict=True):
        if tflag:
            frame = frame.transpose(-1, -2)
        canvas[:, :, r:r + h, c:c + w] = frame
    return canvas


def _masked_global_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``x``: ``(N, C, H, W)``; ``mask``: ``(1, H, W)``. Returns ``(N, C)``."""
    masked = x * mask  # broadcasts mask across N and C
    valid = mask.sum().clamp(min=1.0)
    return masked.sum(dim=(-2, -1)) / valid


class TactileCanvasCNNEncoder(nn.Module):
    """Feed-forward CNN over a packed-canvas tactile layout.

    Per history frame: scatter per-sensor grids into one canvas, concatenate a
    presence-mask channel, run a small conv stack, masked global-mean-pool over
    real cells, then ``Linear`` to ``output_dim``. History frames are pooled
    independently and concatenated before the final ``Linear`` (mirrors the
    per-sensor encoder's history handling).
    """

    is_recurrent = False
    state_slots = 0

    def __init__(
        self,
        layout: CanvasTactileLayout,
        cnn_cfg: dict[str, Any],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.layout = layout

        cfg = dict(cnn_cfg)
        out_channels: list[int] = list(cfg.pop("output_channels"))
        if not out_channels:
            raise ValueError("TactileCanvasCNNEncoder: cnn_cfg.output_channels must be non-empty.")
        kernel_size = int(cfg.pop("kernel_size", 3))
        if kernel_size % 2 == 0:
            raise ValueError(f"TactileCanvasCNNEncoder: kernel_size must be odd (got {kernel_size}).")
        padding_mode = cfg.pop("padding", "zeros")
        if padding_mode not in ("zeros", "replicate"):
            raise ValueError(f"TactileCanvasCNNEncoder: unsupported padding {padding_mode!r}.")
        activation_name = cfg.pop("activation", "relu")
        cfg.pop("flatten", None)  # ignore rsl_rl-specific flag if passed through
        if cfg:
            raise ValueError(f"TactileCanvasCNNEncoder: unknown cnn_cfg keys: {sorted(cfg.keys())}.")

        self.register_buffer("presence_mask", layout.build_presence_mask(), persistent=False)

        in_channels = layout.base.features_per_probe + 1  # +1 for mask channel
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        c_prev = in_channels
        for c_out in out_channels:
            layers.append(
                nn.Conv2d(
                    c_prev, c_out,
                    kernel_size=kernel_size,
                    padding=pad,
                    padding_mode=padding_mode,
                )
            )
            layers.append(_make_activation(activation_name))
            c_prev = c_out
        self.conv = nn.Sequential(*layers)

        self.head = nn.Linear(layout.base.history_length * c_prev, output_dim)
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        per_sensor_grids = self.layout.base.slice_per_sensor(x)
        lead = x.shape[:-1]
        N = int(torch.tensor(lead).prod().item()) if lead else 1
        T_hist = self.layout.base.history_length

        pooled_history: list[torch.Tensor] = []
        for t in range(T_hist):
            frames = [g[..., t, :, :, :].reshape(N, *g.shape[-3:]) for g in per_sensor_grids]
            canvas = self.layout.scatter_frames(frames)
            mask_bc = self.presence_mask.unsqueeze(0).expand(N, -1, -1, -1)
            canvas = torch.cat([canvas, mask_bc], dim=1)
            conv_out = self.conv(canvas)  # (N, C, H, W)
            pooled = _masked_global_mean(conv_out, self.presence_mask)  # (N, C)
            pooled_history.append(pooled)

        cat = torch.cat(pooled_history, dim=-1)  # (N, T_hist * C)
        out = self.head(cat)  # (N, output_dim)
        return out.reshape(*lead, self._output_dim)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_hidden_state(self) -> None:
        return None

    def set_hidden_state(self, hidden_state: Any) -> None:
        del hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_state_tuple(self) -> tuple[torch.Tensor, ...]:
        return ()

    def set_state_tuple(self, tup: tuple[torch.Tensor, ...]) -> None:
        del tup


class TactileCanvasConvRNNEncoder(nn.Module):
    """ConvRNN over a packed-canvas tactile layout.

    Per step: scatter the most-recent frame into the canvas, concat a
    presence-mask channel, 1x1-project ``(F+1) -> C_h``, run one
    :class:`IntersectionRNNCell` over the whole canvas. The persistent state is
    ``(N, C_h, H_canvas, W_canvas)``. Output is masked global-mean-pool of the
    state followed by ``Linear`` to ``output_dim``.

    Same ``history_length == 1`` constraint as :class:`TactileConvRNNEncoder` --
    temporal memory lives in the cell's state across env steps, not in the
    observation history.
    """

    is_recurrent = True
    state_slots = 1

    def __init__(
        self,
        layout: CanvasTactileLayout,
        convrnn_cfg: dict[str, Any],
        output_dim: int,
    ) -> None:
        super().__init__()
        from pt_tnn.recurrent_cells import IntersectionRNNCell  # local import: optional dep

        if layout.base.history_length != 1:
            raise ValueError(
                "TactileCanvasConvRNNEncoder requires layout.base.history_length == 1: the cell "
                "carries temporal memory across env steps via its persistent state, so obs-stacked "
                f"history would be silently discarded. Got history_length={layout.base.history_length}."
            )

        self.layout = layout

        cfg = dict(convrnn_cfg)
        if "out_channels" not in cfg:
            raise ValueError("TactileCanvasConvRNNEncoder: convrnn_cfg must include 'out_channels'.")
        out_channels = int(cfg.pop("out_channels"))
        cfg_ksize = int(cfg.pop("ksize", 3))
        cfg.setdefault("layernorm", True)

        H, W = layout.canvas_hw
        ksize = _effective_ksize(cfg_ksize, H, W)
        if ksize % 2 == 0 or ksize < 1:
            raise ValueError(f"TactileCanvasConvRNNEncoder: derived ksize must be a positive odd int (got {ksize}).")

        self.register_buffer("presence_mask", layout.build_presence_mask(), persistent=False)

        in_channels = layout.base.features_per_probe + 1  # +1 for mask channel
        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.cell = IntersectionRNNCell(
            input_in_channels=out_channels,
            out_channels=out_channels,
            ksize=ksize,
            **cfg,
        )

        self.head = nn.Linear(out_channels, output_dim)
        self._out_channels = out_channels
        self._output_dim = output_dim

        # Persistent per-env state. None until num_envs is observed for the first time.
        self._state: torch.Tensor | None = None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def _state_dim(self) -> int:
        H, W = self.layout.canvas_hw
        return self._out_channels * H * W

    # ---- state lifecycle (mirrors rsl_rl.modules.rnn.RNN) -----------------

    def _zero_state(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        H, W = self.layout.canvas_hw
        return torch.zeros(num_envs, self._out_channels, H, W, device=device, dtype=dtype)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self._state = None
            return
        if self._state is None:
            return
        idx = (dones == 1)
        self._state[idx] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._state is None:
            return
        if dones is None:
            self._state = self._state.detach()
        else:
            mask = (dones == 1)
            self._state[mask] = self._state[mask].detach()

    def get_hidden_state(self) -> torch.Tensor | None:
        if self._state is None:
            return None
        n = self._state.shape[0]
        return self._state.reshape(n, -1).unsqueeze(0)

    def set_hidden_state(self, hidden_state: Any) -> None:
        if hidden_state is None:
            self._state = None
            return
        if isinstance(hidden_state, (tuple, list)):
            if len(hidden_state) != 1:
                raise ValueError(
                    f"Canvas ConvRNN expects exactly 1 state tensor on the wire, got {len(hidden_state)}."
                )
            hidden_state = hidden_state[0]
        if not isinstance(hidden_state, torch.Tensor):
            raise TypeError(f"Unexpected canvas ConvRNN hidden_state type: {type(hidden_state).__name__}.")
        if hidden_state.dim() != 3 or hidden_state.shape[0] != 1:
            raise ValueError(
                f"Canvas ConvRNN expects state shape (1, N, {self._state_dim}), got {tuple(hidden_state.shape)}."
            )
        if hidden_state.shape[-1] != self._state_dim:
            raise ValueError(
                f"Canvas ConvRNN packed state dim {hidden_state.shape[-1]} != expected {self._state_dim}."
            )
        n = hidden_state.shape[1]
        H, W = self.layout.canvas_hw
        self._state = hidden_state.squeeze(0).reshape(n, self._out_channels, H, W)

    def get_state_tuple(self) -> tuple[torch.Tensor, ...]:
        h = self.get_hidden_state()
        return () if h is None else (h,)

    def set_state_tuple(self, tup: tuple[torch.Tensor, ...]) -> None:
        if not tup:
            self.set_hidden_state(None)
            return
        assert len(tup) == 1, "TactileCanvasConvRNNEncoder expects exactly 1 state slot."
        self.set_hidden_state(tup[0])

    # ---- forward ----------------------------------------------------------

    def _step(self, frame_with_mask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """One temporal step: ``frame_with_mask`` is ``(N, F+1, H, W)``; ``state`` is ``(N, C_h, H, W)``."""
        x = self.input_proj(frame_with_mask)
        _, ns = self.cell(x, state)
        return ns

    def _project(self, state: torch.Tensor) -> torch.Tensor:
        pooled = _masked_global_mean(state, self.presence_mask)  # (N, C_h)
        return self.head(pooled)  # (N, output_dim)

    def _build_canvas(self, per_sensor_frames: list[torch.Tensor]) -> torch.Tensor:
        """Scatter and append mask: returns ``(N, F+1, H, W)``."""
        canvas = self.layout.scatter_frames(per_sensor_frames)
        N = canvas.shape[0]
        mask_bc = self.presence_mask.unsqueeze(0).expand(N, -1, -1, -1)
        return torch.cat([canvas, mask_bc], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        batch_mode = masks is not None
        per_sensor_grids = self.layout.base.slice_per_sensor(x)  # (..., T_hist=1, F, H_s, W_s)

        if batch_mode:
            T, B = x.shape[0], x.shape[1]
            # Only the most recent obs-history frame matters; cell carries memory.
            frames_per_t = [g[:, :, -1] for g in per_sensor_grids]  # (T, B, F, H_s, W_s)
            if hidden_state is None:
                state = self._zero_state(B, x.device, x.dtype)
            else:
                self.set_hidden_state(hidden_state)
                state = self._state if self._state is not None else self._zero_state(B, x.device, x.dtype)

            outs: list[torch.Tensor] = []
            for t in range(T):
                frames_t = [f[t] for f in frames_per_t]  # each (B, F, H_s, W_s)
                canvas = self._build_canvas(frames_t)
                state = self._step(canvas, state)
                outs.append(self._project(state))
            out = torch.stack(outs, dim=0)  # (T, B, output_dim)
            return unpad_trajectories(out, masks)

        # Inference: x: (N, flat_dim). Each per_sensor_grids[s]: (N, T_hist=1, F, H_s, W_s).
        N = x.shape[0]
        frames = [g[:, -1] for g in per_sensor_grids]  # (N, F, H_s, W_s)
        if self._state is None or self._state.shape[0] != N:
            self._state = self._zero_state(N, x.device, x.dtype)
        canvas = self._build_canvas(frames)
        self._state = self._step(canvas, self._state)
        return self._project(self._state)


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}; expected one of relu|elu|gelu|tanh.")
